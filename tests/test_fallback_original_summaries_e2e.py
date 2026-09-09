"""End-to-end: the page-level fallback summarized by the ORIGINAL summary code flow.

Normally a page-level fallback tree is summarized by
FallbackTreeBuilder.summarize_page_level() (pageindex_fallback/tree_builder.py): the page
leaves from their own text, the root rolled up from those leaf summaries. That exists
because the root spans every page, so summarizing it from its own text means putting the
whole document in one prompt.

This file runs the fallback through the *original* flow instead —
pageindex.utils.generate_summaries_for_structure(), the library function every normal tree
uses — so every node including the root is summarized from its own text. That is the only
thing forced here; the rest is the real pipeline.

Genuinely end to end: the real FastAPI route, the real request/response models, the real
MarkdownIndexer, the real FallbackTreeBuilder, and real LLM calls. Only S3 is stubbed, to
a local file.

Run it
------
    .venv/bin/python tests/test_fallback_original_summaries_e2e.py

Edit the CONFIG block below first. This costs real money: one summary call per page, plus
one for the root whose prompt is the entire document, plus the TOC extraction that has to
run and fail before the fallback engages.

`pytest tests/` also collects this file, but only test_original_flow_is_wired_up(), which
checks the wiring offline with no network. The live run happens under __main__ only, so a
test sweep never spends anything.
"""
import asyncio
import json
import logging
import os
import sys
from importlib import import_module

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pageindex.utils as pageindex_utils
from pageindex_fallback import FallbackTreeBuilder, PageLevelIndex

page_index = import_module("pageindex.page_index")


# ══ CONFIG — edit these, then run: python tests/test_fallback_original_summaries_e2e.py ══

# The markdown file to index. Relative paths resolve from the repository root.
MD_PATH = "/Users/hardikjoshi/Desktop/LLM_engine/llm-engine/test_files/ocr/Novature.md"

# Leave as None to use the SAME model PageIndex summarizes with — `model` in
# pageindex/config.yaml, overridable by PAGEINDEX_MODEL, which is the exact `opt.model`
# handed to generate_summaries_for_structure(). That is the default and what you want.
# Set a string only to deliberately run against a different provider.
MODEL = None

# Make the fallback engage by having verification reject the extracted TOC. This is the
# real trigger reproduced, not a shortcut: verify_toc returns exactly (0.0, []) when a
# TOC's last physical_index lands in the first half of the document.
# Set False to let the document fall back on its own merits — it may not.
FORCE_FALLBACK = True

# Pay for the TOC extraction that the fallback then discards. Off by default, and the
# default is what you want: its output is thrown away either way, on a document this size
# it is four sequential ~20k-token calls, and if one of them returns a truncated completion
# process_no_toc RAISES — which routes the run through MarkdownIndexer's retry and lands on
# the extraction-failure fallback (fallback_accuracy None) instead of the verification one
# (0.0). Leaving it live therefore makes "forced" non-deterministic as well as expensive.
# Turn it on only to watch the real extraction fail.
PAY_FOR_DOOMED_EXTRACTION = False

# Also generate the document description (one extra call, over titles + summaries).
ADD_DOC_DESCRIPTION = True

# Keep each node's page text in the saved JSON. Off by default: it duplicates the whole
# document into the tree.
ADD_NODE_TEXT = False

# Virtual page size, used only for markdown WITHOUT <page_number=N> tags.
TOKENS_PER_PAGE = 2000

# Where to write the result JSON.
OUTPUT_DIR = "./results"

# ═══════════════════════════════════════════════════════════════════════════════════════

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Forcing the original summary flow ─────────────────────────────────────────

class OriginalSummaryFlow:
    """Route a page-level tree through generate_summaries_for_structure().

    MarkdownIndexer._add_summaries() calls
    FallbackTreeBuilder.summarize(tree, fallback_used=fallback.used), and
    `fallback_used=True` picks the page-level strategy. Patching `summarize` to ignore that
    flag is the whole override: the tree is then summarized by the library's original
    function, node by node from each node's own text.

    Also records the text size handed to each node's summary, so the cost of doing this to
    a whole-document root is visible rather than implied.
    """

    def __init__(self):
        self.summarized = []          # (title, chars of text prompted with)
        self._real_summarize = None
        self._real_node_summary = None

    def __enter__(self):
        self._real_summarize = FallbackTreeBuilder.summarize
        self._real_node_summary = pageindex_utils.generate_node_summary
        recorder = self

        async def original_flow(self, tree, fallback_used=False):
            # the library's own summary pass, whatever kind of tree this is
            return await pageindex_utils.generate_summaries_for_structure(
                tree, model=self.model
            )

        async def recording_node_summary(node, model=None):
            recorder.summarized.append((node.get("title"), len(node.get("text") or "")))
            return await recorder._real_node_summary(node, model=model)

        FallbackTreeBuilder.summarize = original_flow
        pageindex_utils.generate_node_summary = recording_node_summary
        return self

    def __exit__(self, *exc):
        FallbackTreeBuilder.summarize = self._real_summarize
        pageindex_utils.generate_node_summary = self._real_node_summary
        return False

    @property
    def largest_prompt(self):
        return max((chars for _title, chars in self.summarized), default=0)


class ForcedFallback:
    """Make the page-level index the outcome, deterministically.

    Two patches, and both are load-bearing:

    * verify_toc -> (0.0, []) reproduces the real trigger rather than short-circuiting it.
      That pair is exactly what the library returns when the extracted TOC's last
      physical_index lands in the first half of the document; an empty incorrect_results
      means the repair pass has nothing to fix, so FallbackTreeBuilder finds the extraction
      unusable and builds PageLevelIndex.

    * process_no_toc is replaced by a stub, unless PAY_FOR_DOOMED_EXTRACTION is set. Its
      result is discarded by the fallback in every case, so the real call buys nothing —
      and because generate_toc_init/generate_toc_continue raise on a truncated completion,
      leaving it live means the run can end up on the *extraction-failure* route
      (fallback_accuracy None, attempts > 1) rather than the verification route this file
      exists to exercise. Stubbing it is what makes the force deterministic.

    Both patches are restored on exit, so nothing leaks into a later run or into pytest.
    """

    def __init__(self, enabled=True, stub_extraction=True):
        self.enabled = enabled
        # A natural run needs the real extraction; only stub it when forcing.
        self.stub_extraction = enabled and stub_extraction
        self._real_verify_toc = None
        self._real_process_no_toc = None

    def __enter__(self):
        if self.enabled:
            self._real_verify_toc = page_index.verify_toc

            async def failing_verify_toc(page_list, list_result, start_index=1, N=None,
                                         model=None):
                return 0.0, []

            page_index.verify_toc = failing_verify_toc

        if self.stub_extraction:
            self._real_process_no_toc = page_index.process_no_toc

            async def stub_process_no_toc(page_list, start_index=1, model=None, logger=None):
                # Shaped like a real result so validate_and_truncate_physical_indices and
                # verify_toc see what they expect; rejected by the patch above regardless.
                return [{"structure": "1", "title": "unverified extraction",
                         "physical_index": start_index}]

            page_index.process_no_toc = stub_process_no_toc
        return self

    def __exit__(self, *exc):
        if self.enabled:
            page_index.verify_toc = self._real_verify_toc
        if self.stub_extraction:
            page_index.process_no_toc = self._real_process_no_toc
        return False


# ── S3 stubbed onto the local filesystem ──────────────────────────────────────

class LocalS3:
    """Stands in for S3: reads the markdown off disk, keeps the upload in memory."""

    def __init__(self, md_path):
        self.md_path = md_path
        self.uploaded_key = None
        self.uploaded_data = None

    async def read(self, key, s3_session, bucket):
        with open(self.md_path, encoding="utf-8") as f:
            return f.read()

    async def write(self, key, data, s3_session, bucket):
        self.uploaded_key = key
        self.uploaded_data = data


# ── Tree inspection ───────────────────────────────────────────────────────────

def make_builder(page_list, doc_title="doc"):
    """A builder just for its summary half — nothing here extracts anything."""
    from pageindex.utils import ConfigLoader

    return FallbackTreeBuilder(
        page_list, ConfigLoader().load({}), logging.getLogger(__name__),
        doc_title=doc_title,
    )


def walk(nodes, depth=0):
    for node in nodes:
        yield depth, node
        yield from walk(node.get("nodes") or [], depth + 1)


def count_nodes(nodes):
    return sum(1 for _ in walk(nodes))


def check_page_level_shape(structure, page_count):
    """Invariants a page-level tree must satisfy. Empty list means it is well formed."""
    problems = []
    if len(structure) != 1:
        return [f"expected 1 root node, got {len(structure)}"]

    root = structure[0]
    leaves = root.get("nodes") or []
    if len(leaves) != page_count:
        problems.append(f"expected {page_count} page leaves, got {len(leaves)}")
    if count_nodes(structure) != page_count + 1:
        problems.append(
            f"expected {page_count + 1} nodes total, got {count_nodes(structure)}"
        )
    if (root["start_index"], root["end_index"]) != (1, page_count):
        problems.append(
            f"root should span 1-{page_count}, spans "
            f"{root['start_index']}-{root['end_index']}"
        )
    ranges = [(n["start_index"], n["end_index"]) for n in leaves]
    if ranges != [(i, i) for i in range(1, len(leaves) + 1)]:
        problems.append("page leaves are not one contiguous page each")
    if any(n.get("nodes") for n in leaves):
        problems.append("page leaves should have no children")

    missing = [n["title"] for _d, n in walk(structure) if not n.get("summary")]
    if missing:
        problems.append(f"{len(missing)} node(s) have no summary: {missing[:5]}")
    return problems


# ── The live run ──────────────────────────────────────────────────────────────

def index_markdown_file(md_path, model=None, force_fallback=True, doc_description=True,
                        node_text=False, tokens_per_page=2000,
                        pay_for_doomed_extraction=False):
    """POST a local .md through the real route, with the original summary flow forced.

    Returns (response_body, local_s3, summary_flow).
    """
    from fastapi.testclient import TestClient

    import api.routers.pageindex as router_module
    from api.services import pageindex_service as svc
    from api.server import app

    local_s3 = LocalS3(md_path)
    real_read, real_write = svc.read_markdown_from_s3, svc.upload_tree_to_s3
    real_session, real_bucket = router_module.get_s3_session, router_module.get_s3_bucket
    svc.read_markdown_from_s3 = local_s3.read
    svc.upload_tree_to_s3 = local_s3.write
    router_module.get_s3_session = lambda: None
    router_module.get_s3_bucket = lambda: "test-bucket"

    stem = os.path.splitext(os.path.basename(md_path))[0]
    request = {
        "input_s3_key": f"incoming/{os.path.basename(md_path)}",
        "output_s3_key": f"trees/{stem}.json",
        "tokens_per_page": tokens_per_page,
        "if_add_node_summary": "yes",
        "if_add_doc_description": "yes" if doc_description else "no",
        "if_add_node_text": "yes" if node_text else "no",
    }
    if model:
        request["model"] = model

    try:
        with ForcedFallback(force_fallback,
                            stub_extraction=not pay_for_doomed_extraction), \
                OriginalSummaryFlow() as flow:
            with TestClient(app) as client:
                response = client.post("/api/v1/pageindex/markdown", json=request)
    finally:
        svc.read_markdown_from_s3, svc.upload_tree_to_s3 = real_read, real_write
        router_module.get_s3_session = real_session
        router_module.get_s3_bucket = real_bucket

    if response.status_code != 200:
        raise AssertionError(
            f"POST /api/v1/pageindex/markdown returned {response.status_code}: "
            f"{response.text[:500]}"
        )
    return response.json(), local_s3, flow


def preflight(model):
    """Fail before spending anything if the model is unreachable."""
    import litellm

    try:
        litellm.completion(
            model=(model or "").removeprefix("litellm/"),
            messages=[{"role": "user", "content": "reply with the single word ok"}],
            temperature=0, num_retries=0,
        )
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:200]}"


def main():
    from pageindex.utils import ConfigLoader
    from api.services.pageindex_service import markdown_to_page_list

    md_path = MD_PATH if os.path.isabs(MD_PATH) else os.path.normpath(
        os.path.join(REPO_ROOT, MD_PATH)
    )
    if not os.path.isfile(md_path):
        print(f"MD_PATH does not point at a file: {md_path}")
        return 2

    # PAGEINDEX_MODEL is honoured by the service itself (ENV_MAP in pageindex_service),
    # so resolve it here too — otherwise preflight probes one model while the run uses
    # another, and the banner below names the wrong one.
    model_override = MODEL or os.environ.get("PAGEINDEX_MODEL")
    opt = ConfigLoader().load({"model": model_override} if model_override else {})
    with open(md_path, encoding="utf-8") as f:
        content = f.read()
    tagged = "<page_number=" in content
    page_list = markdown_to_page_list(content, TOKENS_PER_PAGE, opt.model)
    total_chars = sum(len(text) for text, _ in page_list)

    print("=" * 76)
    print("PageIndex fallback, ORIGINAL summary flow — end to end")
    print("=" * 76)
    print(f"file:            {md_path}  ({len(content):,} chars)")
    print(f"paging:          {'<page_number=N> tags' if tagged else f'~{TOKENS_PER_PAGE} token chunks'}")
    print(f"virtual pages:   {len(page_list)}")
    model_source = (
        "MODEL override" if MODEL
        else "PAGEINDEX_MODEL env var" if os.environ.get("PAGEINDEX_MODEL")
        else "pageindex/config.yaml"
    )
    print(f"model:           {opt.model}  (REAL calls, from {model_source})")
    print(f"                 same model PageIndex summarizes with")
    print(f"summary flow:    generate_summaries_for_structure — every node from its own text")
    print(f"fallback:        {'FORCED — verify_toc rejects the TOC' if FORCE_FALLBACK else 'only if it happens naturally'}")
    print(f"extraction:      {'REAL — paid for, then discarded' if PAY_FOR_DOOMED_EXTRACTION else 'stubbed — discarded either way, so not paid for'}")
    print()
    calls = len(page_list) + 1 + (1 if ADD_DOC_DESCRIPTION else 0)
    print(f"Expect {calls} LLM calls: {len(page_list)} page summaries, 1 for the root"
          + (", 1 description" if ADD_DOC_DESCRIPTION else ""))
    print(f"The root's prompt carries the whole document (~{total_chars:,} chars).")
    print("That is the cost the roll-up strategy exists to avoid.")
    print()

    error = preflight(opt.model)
    if error:
        print(f"The model is unreachable: {error}")
        print()
        print("Set MODEL in the CONFIG block to a provider that works, e.g.")
        print('    MODEL = "anthropic/claude-sonnet-4-5-20250929"')
        return 2

    body, local_s3, flow = index_markdown_file(
        md_path, model=MODEL, force_fallback=FORCE_FALLBACK,
        doc_description=ADD_DOC_DESCRIPTION, node_text=ADD_NODE_TEXT,
        tokens_per_page=TOKENS_PER_PAGE,
        pay_for_doomed_extraction=PAY_FOR_DOOMED_EXTRACTION,
    )

    structure = body["structure"]
    print("-" * 76)
    print("RESULT")
    print("-" * 76)
    print(f"fallback_used:     {body['fallback_used']}")
    print(f"fallback_accuracy: {local_s3.uploaded_data['fallback_accuracy']}")
    print(f"attempts:          {body.get('attempts')}")
    print(f"total nodes:       {count_nodes(structure)}")
    print(f"summary calls:     {len(flow.summarized)}")
    print(f"largest prompt:    {flow.largest_prompt:,} chars of node text")
    if body.get("doc_description"):
        print(f"doc_description:   {body['doc_description'][:200]}")
    print()

    if structure:
        root = structure[0]
        print("ROOT")
        print(f"  title:   {root['title']}  (pages {root['start_index']}-{root['end_index']})")
        print(f"  summary: {(root.get('summary') or '')[:400]}")
        print()
        print("FIRST THREE PAGES")
        for leaf in (root.get("nodes") or [])[:3]:
            print(f"  [{leaf.get('node_id')}] {leaf['title']} "
                  f"(pages {leaf['start_index']}-{leaf['end_index']})")
            print(f"      {(leaf.get('summary') or '')[:200]}")
        print()

    problems = []
    if body["fallback_used"]:
        problems = check_page_level_shape(structure, len(page_list))
    elif FORCE_FALLBACK:
        problems = ["the fallback was forced but did not engage"]

    print("-" * 76)
    if problems:
        print(f"FAILED — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print(f"PASSED — {count_nodes(structure)} nodes")
        print(f"  - 1 root spanning 1-{len(page_list)} plus {len(page_list)} contiguous page leaves")
        print("  - every node has a summary from the original flow")
        print(f"  - the root was summarized from its own text ({flow.largest_prompt:,} chars)")
    print("-" * 76)

    out_dir = OUTPUT_DIR if os.path.isabs(OUTPUT_DIR) else os.path.normpath(
        os.path.join(REPO_ROOT, OUTPUT_DIR)
    )
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(md_path))[0]
    out_path = os.path.join(out_dir, f"{stem}_fallback_original_summaries.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(local_s3.uploaded_data, f, indent=2, ensure_ascii=False)
    print(f"\nTree JSON (the object that would be uploaded to {local_s3.uploaded_key}) "
          f"written to:\n  {out_path}")

    return 1 if problems else 0


# ── Offline wiring check (this is what `pytest tests/` runs) ───────────────────

def test_original_flow_is_wired_up():
    """The override must reach generate_summaries_for_structure, root included.

    No network: generate_node_summary is replaced, so nothing calls out. This pins the
    difference from the default page-level strategy — here the root is summarized from its
    own text, not rolled up from its children.
    """
    pages = [(f"body text of page {i}\n", 10) for i in range(1, 6)]
    tree = PageLevelIndex(pages, doc_title="doc").tree()
    from pageindex.utils import add_node_text

    add_node_text(tree, pages)

    prompted = []
    real_node_summary = pageindex_utils.generate_node_summary
    rollups = []
    real_rollup = FallbackTreeBuilder.rollup

    async def fake_node_summary(node, model=None):
        prompted.append(node["title"])
        return f"summary of {node['title']}"

    async def fake_rollup(self, child_summaries):
        rollups.append(child_summaries)
        return "rolled up"

    pageindex_utils.generate_node_summary = fake_node_summary
    FallbackTreeBuilder.rollup = fake_rollup
    try:
        with OriginalSummaryFlow() as flow:
            # fallback_used=True is what would normally select the roll-up strategy
            asyncio.run(make_builder(pages, doc_title="doc").summarize(
                tree, fallback_used=True
            ))
    finally:
        pageindex_utils.generate_node_summary = real_node_summary
        FallbackTreeBuilder.rollup = real_rollup

    # every node, root included, went through the original per-node flow
    assert prompted == ["doc", "Page 1", "Page 2", "Page 3", "Page 4", "Page 5"]
    assert rollups == [], "the roll-up strategy must not run"
    assert all(node.get("summary") for _d, node in walk(tree))
    assert tree[0]["summary"] == "summary of doc"

    # and the recorder saw the root prompted with the whole document
    root_chars = dict(flow.summarized)["doc"]
    assert root_chars == sum(len(text) for text, _ in pages)


def test_forced_fallback_stubs_the_extraction_and_restores_it():
    """Forcing must not pay for the extraction it discards, and must not leak the stub.

    The stub also has to be shaped like a real result: validate_and_truncate_physical_
    indices compares physical_index against the page count, so a None or a bare string
    there would raise before verify_toc ever got a say.
    """
    real_verify, real_extract = page_index.verify_toc, page_index.process_no_toc

    with ForcedFallback(enabled=True) as forced:
        assert forced.stub_extraction is True
        assert page_index.process_no_toc is not real_extract, "extraction was not stubbed"
        assert page_index.verify_toc is not real_verify

        extracted = asyncio.run(page_index.process_no_toc([("page one", 10)], start_index=1))
        assert extracted == [{"structure": "1", "title": "unverified extraction",
                              "physical_index": 1}]
        assert asyncio.run(page_index.verify_toc([("page one", 10)], extracted)) == (0.0, [])

    assert page_index.process_no_toc is real_extract
    assert page_index.verify_toc is real_verify


def test_paying_for_the_extraction_leaves_it_alone():
    """PAY_FOR_DOOMED_EXTRACTION=True keeps the real extraction; only verify_toc is patched."""
    real_extract = page_index.process_no_toc

    with ForcedFallback(enabled=True, stub_extraction=False) as forced:
        assert forced.stub_extraction is False
        assert page_index.process_no_toc is real_extract


def test_a_natural_run_patches_nothing():
    """FORCE_FALLBACK=False must leave both the extraction and the judge real."""
    real_verify, real_extract = page_index.verify_toc, page_index.process_no_toc

    with ForcedFallback(enabled=False) as forced:
        assert forced.stub_extraction is False
        assert page_index.verify_toc is real_verify
        assert page_index.process_no_toc is real_extract


def test_summarize_is_restored_after_the_override():
    """The patch must not leak into other tests or a later live run."""
    before = FallbackTreeBuilder.summarize
    with OriginalSummaryFlow():
        assert FallbackTreeBuilder.summarize is not before
    assert FallbackTreeBuilder.summarize is before
    assert pageindex_utils.generate_node_summary.__name__ == "generate_node_summary"


if __name__ == "__main__":
    sys.exit(main())
