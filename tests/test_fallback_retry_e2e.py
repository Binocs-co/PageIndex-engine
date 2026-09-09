"""End-to-end runs of one real .md document: the original method, the retry budget up to
the third attempt, and the page-level fallback — every one through the API's own
process_markdown(), each saving the tree it produced.

Every scenario calls process_markdown() — the one function the route's handler calls —
with a payload built from the real MarkdownPageIndexRequest, so the request's defaults and
bounds still apply. No HTTP and no TestClient: the transport is the only thing skipped.
Only two things are stubbed, and neither is the code under test: S3 reads/writes map to
the local file (LocalS3) and the LLM is an offline stand-in that builds a real TOC from
the document's own markdown headings. Set USE_REAL_LLM = True to call the model in
pageindex/config.yaml instead.

    original_method                   nothing injected                       -> the baseline
    retry_then_success                attempt 1 raises, attempt 2 runs clean -> == baseline
    third_attempt_succeeds            attempts 1-2 raise, attempt 3 clean    -> == baseline
    all_three_attempts_fail           all 3 attempts raise, budget spent     -> page-level tree
    retry_then_fallback               both attempts raise (budget of 2)      -> page-level tree
    verification_fallback             the TOC fails verification             -> page-level tree
    retry_then_verification_fallback  attempt 1 raises, then the TOC fails   -> page-level tree

The first three inject only *raised* failures, which the retry is supposed to absorb, so
none of them asserts a fixed outcome: they are compared against `original_method`. Whether
a document yields a verifiable TOC at all belongs to the document and the model — the
invariant being tested is that retrying lands on the same tree an uninjected run reaches,
differing only in `attempts`.

`third_attempt_succeeds` and `all_three_attempts_fail` sit at the API's ceiling —
max_tree_attempts is capped at 3 by both the request field and
MarkdownIndexer.MAX_TREE_ATTEMPTS_CEILING, so a third attempt is as far as the retry goes.
`retry_then_verification_fallback` is the only scenario where `attempts > 1` and
`fallback_accuracy` is a real number at the same time.

## Running it on your own document

Point MD_PATH at any .md file — or drop yours over tests/fixtures/sample_document.md — and
both modes index it.

Mind what the offline stand-in can and cannot read. It builds its TOC from ATX markdown
headings (`# Heading`) and nothing else, so a document that marks its structure some other
way — an OCR dump using `**bold**` lines, say — gives it no headings to work with. It
extracts an empty TOC, verify_toc takes its early exit (accuracy 0, nothing to repair) and
every scenario lands on the page-level fallback. That is a limit of the stand-in, not of
the pipeline: set USE_REAL_LLM = True to let a real model read the prose. It is also
exactly the case the fallback exists for — upstream `meta_processor` raises
`Processing failed` on that same early exit, which the route turns into a 500.

Script mode saves every scenario's tree plus a summary.json under ./results/fallback_retry:

    python tests/test_fallback_retry_e2e.py

Suite mode asserts the same expectations and writes to a tmp dir, so nothing lands in the
repo:

    pytest tests/test_fallback_retry_e2e.py -v
"""
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from importlib import import_module

import pytest

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(_TESTS_DIR)
sys.path.insert(0, REPO_ROOT)    # for `api`, `pageindex`, `pageindex_fallback`
sys.path.insert(0, _TESTS_DIR)   # for the harness module below

# The offline harness lives in the end-to-end test: LocalS3, the stand-in LLM,
# fail_extraction() and the tree checks. Reused rather than duplicated.
import test_md_end_to_end as harness  # noqa: E402

# `pageindex/__init__.py` does `from .page_index import *`, so the package attribute
# `pageindex.page_index` is the *function*, not the module — import the module explicitly.
pi = import_module("pageindex.page_index")  # noqa: E402


# ══ CONFIG — script mode only; pytest ignores this block ══════════════════════

# The markdown file to index. Relative paths resolve from the repository root.
MD_PATH = "tests/fixtures/sample_document.md"

# False: offline stand-in LLM (no API key or credits needed).
# True:  call the real model — real extraction, real verification, real summaries.
#        This costs money. Needs the provider key for MODEL in the environment.
USE_REAL_LLM = True

# Which model to run against. None uses pageindex/config.yaml (gpt-4o-2024-11-20, which
# needs OPENAI_API_KEY). Set it to override, e.g. "anthropic/claude-sonnet-4-6".
MODEL = None

# The three trees script mode produces, in order. Everything in SCENARIOS is still
# available to pytest; this is just what a script run indexes.
TREES = ("original_method", "retry_then_success", "verification_fallback")

# Summaries exercise the builder's summary half, root roll-up included on a fallback tree.
ADD_SUMMARIES = True

# Include each node's raw page text in the saved JSON.
ADD_NODE_TEXT = False

# Token budget per virtual page. Ignored for a document that carries <page_number=N>
# tags — each tag block is one page then.
TOKENS_PER_PAGE = 2000

# Where the saved trees go.
OUTPUT_DIR = "./results/fallback_retry"

# ══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class Scenario:
    """One request's worth of injected failure, plus what the response must look like."""

    name: str
    what: str
    # How many extraction attempts raise before one is allowed through.
    # None means every attempt raises, so the retry budget is spent.
    fail_first: int | None
    # Make verify_toc report a failing accuracy with nothing to repair — the real
    # trigger for the verification fallback.
    force_fallback: bool
    max_tree_attempts: int
    expect_attempts: int
    # True/False when the scenario forces the outcome. None means "whatever this document
    # naturally yields" — checked against the baseline run instead of a fixed value,
    # because whether a document produces a verifiable TOC is the document's business.
    expect_fallback: bool | None
    # "score": fallback_accuracy is a real number, so verification ran and failed.
    # "null":  fallback_accuracy is null — either no fallback, or extraction never
    #          got far enough to be scored.
    # None:    no claim; the baseline comparison covers it.
    expect_accuracy: str | None


SCENARIOS = (
    # The baseline. Nothing is injected, so whatever comes out is what this document and
    # this model produce on their own — a real TOC, or the fallback if the TOC cannot be
    # verified. Every scenario below that injects only *raised* failures is compared
    # against it rather than against a hard-coded outcome.
    Scenario(
        name="original_method",
        what="nothing injected — the untouched pipeline on this document",
        fail_first=0,
        force_fallback=False,
        max_tree_attempts=3,
        expect_attempts=1,
        expect_fallback=None,
        expect_accuracy=None,
    ),
    Scenario(
        name="retry_then_success",
        what="attempt 1 raises, attempt 2 runs clean — must match the baseline",
        fail_first=1,
        force_fallback=False,
        max_tree_attempts=2,
        expect_attempts=2,
        expect_fallback=None,
        expect_accuracy=None,
    ),
    Scenario(
        name="third_attempt_succeeds",
        what="attempts 1-2 raise, attempt 3 runs clean — the budget ceiling",
        fail_first=2,
        force_fallback=False,
        max_tree_attempts=3,
        expect_attempts=3,
        expect_fallback=None,
        expect_accuracy=None,
    ),
    Scenario(
        name="all_three_attempts_fail",
        what="all 3 attempts raise, budget spent -> page-level index",
        fail_first=None,
        force_fallback=False,
        max_tree_attempts=3,
        expect_attempts=3,
        expect_fallback=True,
        expect_accuracy="null",
    ),
    Scenario(
        name="retry_then_fallback",
        what="both attempts raise on a budget of 2 -> page-level index",
        fail_first=None,
        force_fallback=False,
        max_tree_attempts=2,
        expect_attempts=2,
        expect_fallback=True,
        expect_accuracy="null",
    ),
    Scenario(
        name="verification_fallback",
        what="nothing raises, but the TOC fails verification",
        fail_first=0,
        force_fallback=True,
        max_tree_attempts=2,
        expect_attempts=1,
        expect_fallback=True,
        expect_accuracy="score",
    ),
    Scenario(
        name="retry_then_verification_fallback",
        what="attempt 1 raises, attempt 2's TOC fails verification",
        fail_first=1,
        force_fallback=True,
        max_tree_attempts=3,
        expect_attempts=2,
        expect_fallback=True,
        expect_accuracy="score",
    ),
)


def configured_md_path():
    """The .md both modes index: whatever MD_PATH points at, or None if it is missing."""
    path = MD_PATH if os.path.isabs(MD_PATH) else os.path.join(REPO_ROOT, MD_PATH)
    return path if os.path.isfile(path) else None


@dataclass
class ScenarioResult:
    scenario: Scenario
    body: dict
    payload: dict          # exactly the object that would be uploaded to S3
    page_count: int
    node_count: int
    extraction_calls: int  # how many times process_no_toc was entered
    llm_calls: dict
    out_path: str
    real_llm: bool = False
    problems: list = field(default_factory=list)


def virtual_page_count(md_path):
    """How many virtual pages the service will split this file into."""
    from api.services.pageindex_service import markdown_to_page_list
    from pageindex.utils import ConfigLoader

    with open(md_path, encoding="utf-8") as f:
        content = f.read()
    model = ConfigLoader().load({}).model
    return len(markdown_to_page_list(content, TOKENS_PER_PAGE, model))


# Which pipeline stage a prompt belongs to. Same names the offline stand-in counts under,
# so an offline run and a real run are directly comparable.
_PROMPT_KINDS = (
    ("toc_detected", "toc_detector"),
    ("continue the tree structure", "generate_toc"),
    ("generate the tree structure of the document", "generate_toc"),
    ("check if the given section appears or starts in the given page_text", "verify_title"),
    ("starts in the beginning of the given page_text", "title_at_start"),
    ("find the physical index of the start page of the section", "fix_index"),
    ("generate a single description of what the whole document covers", "rollup_summary"),
    ("generate a description of the partial document", "node_summary"),
    ("generate a one-sentence description for the document", "doc_description"),
)


def count_real_llm_calls(monkeypatch):
    """Count real provider calls by stage, without changing a single answer.

    USE_REAL_LLM leaves the model untouched, which also left `llm_calls` empty — there was
    no way to see that the summaries really came from the provider, or how many calls each
    tree cost. This wraps the two entry points and delegates every call straight through,
    so the numbers are the real run's.

    Returns an object with `.calls`, shaped like StandInLLM's.
    """
    from pageindex import utils as pageindex_utils

    class Counter:
        def __init__(self):
            self.calls = {}

        def tally(self, prompt):
            for needle, kind in _PROMPT_KINDS:
                if needle in prompt:
                    self.calls[kind] = self.calls.get(kind, 0) + 1
                    return
            self.calls["other"] = self.calls.get("other", 0) + 1

    counter = Counter()
    real_sync, real_async = pageindex_utils.llm_completion, pageindex_utils.llm_acompletion

    def counting_sync(model, prompt, chat_history=None, return_finish_reason=False):
        counter.tally(prompt)
        return real_sync(model, prompt, chat_history, return_finish_reason)

    async def counting_async(model, prompt, return_finish_reason=False):
        counter.tally(prompt)
        return await real_async(model, prompt, return_finish_reason)

    for module in (pageindex_utils, pi):
        harness._patch(module, "llm_completion", counting_sync, monkeypatch)
        harness._patch(module, "llm_acompletion", counting_async, monkeypatch)
    return counter


def build_payload(md_path, *, scenario, summaries, node_text, model=None):
    """The request object the route would have handed to process_markdown().

    Built with the real MarkdownPageIndexRequest so field defaults, bounds and coercion
    are the API's own — the only thing skipped is the HTTP hop.
    """
    from api.routers.pageindex import MarkdownPageIndexRequest

    stem = os.path.splitext(os.path.basename(md_path))[0]
    return MarkdownPageIndexRequest(
        input_s3_key=f"incoming/{os.path.basename(md_path)}",
        output_s3_key=f"trees/{stem}.json",
        tokens_per_page=TOKENS_PER_PAGE,
        if_add_node_summary="yes" if summaries else "no",
        if_add_doc_description="yes" if summaries else "no",
        if_add_node_text="yes" if node_text else "no",
        max_tree_attempts=scenario.max_tree_attempts,
        **({"model": model} if model else {}),
    )


def call_service(md_path, monkeypatch, *, scenario, real_llm, summaries, node_text,
                 model=None):
    """Call the function the route calls — process_markdown() — and nothing above it.

    No TestClient and no HTTP: the route's body is one line, `await process_markdown(
    payload, get_s3_session(), get_s3_bucket())`, so calling that directly exercises the
    same pipeline without the transport. One consequence worth knowing: the route filters
    its return through `response_model`, and this does not, so `fallback_accuracy` is
    present here even though a real HTTP caller would not see it.

    Two things are still substituted, neither of them the code under test:
      * S3 — read_markdown_from_s3/upload_tree_to_s3 are swapped for LocalS3, so the
        markdown comes off disk and the tree is captured in memory. `s3_session` is
        therefore never used and is passed as None.
      * the LLM — the offline stand-in, unless real_llm is set.

    Returns (result, local_s3, stand_in_llm_or_None).
    """
    from api.services import pageindex_service as svc

    # Real runs get a pass-through counter so the provider calls are still visible.
    llm = (count_real_llm_calls(monkeypatch) if real_llm
           else harness.install_stand_in_llm(monkeypatch))

    if scenario.force_fallback:
        # What the real verify_toc returns from its early exit when the extracted TOC
        # never reaches halfway through the document: a failing accuracy with no
        # mismatches to hand the repair pass, so the builder goes to the page-level index.
        async def failing_verify_toc(page_list, list_result, start_index=1, N=None, model=None):
            return 0.0, []

        harness._patch(pi, "verify_toc", failing_verify_toc, monkeypatch)

    local_s3 = harness.LocalS3(md_path)
    harness._patch(svc, "read_markdown_from_s3", local_s3.read, monkeypatch)
    harness._patch(svc, "upload_tree_to_s3", local_s3.write, monkeypatch)

    payload = build_payload(
        md_path, scenario=scenario, summaries=summaries, node_text=node_text,
        model=model,
    )
    result = asyncio.run(svc.process_markdown(payload, None, "test-bucket"))
    return result, local_s3, llm


def run_scenario(scenario, md_path, out_dir, monkeypatch, *,
                 real_llm=False, summaries=True, node_text=False, model=None):
    """Drive one scenario through process_markdown() and save the tree it produced."""
    # Injected before call_service so it captures the real process_no_toc.
    injector = harness.fail_extraction(monkeypatch, times=scenario.fail_first)

    body, local_s3, llm = call_service(
        md_path, monkeypatch,
        scenario=scenario, real_llm=real_llm, summaries=summaries,
        node_text=node_text, model=model,
    )

    page_count = virtual_page_count(md_path)
    structure = body["structure"]

    problems = harness.check_tree_invariants(structure, page_count=page_count)
    if body["fallback_used"]:
        problems += harness.check_page_level_shape(structure, page_count=page_count)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{scenario.name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(local_s3.uploaded_data, f, indent=2, ensure_ascii=False)

    return ScenarioResult(
        scenario=scenario,
        body=body,
        payload=local_s3.uploaded_data,
        page_count=page_count,
        node_count=harness.count_nodes(structure),
        extraction_calls=injector["calls"],
        llm_calls=dict(sorted(llm.calls.items())) if llm else {},
        out_path=out_path,
        real_llm=real_llm,
        problems=problems,
    )


BASELINE = next(s for s in SCENARIOS if s.expect_fallback is None)


def tree_skeleton(structure):
    """Titles and page ranges, depth-first — the part of a tree a retry must reproduce.

    Summaries are left out deliberately: they are prose, and with a real model they are
    not reproducible run to run. What a retried attempt has to reproduce is the structure.
    """
    return [
        (depth, node["title"], node["start_index"], node["end_index"])
        for depth, node in harness.walk(structure)
    ]


def render_tree_with_summaries(nodes, width=96):
    """The tree with each node's summary underneath it — the point of a real-LLM run."""
    lines = []
    for depth, node in harness.walk(nodes):
        pad = "  " * depth
        node_id = node.get("node_id", "----")
        lines.append(f"{pad}[{node_id}] {node['title']}  "
                     f"(pages {node['start_index']}-{node['end_index']})")
        summary = (node.get("summary") or "").strip()
        if not summary:
            lines.append(f"{pad}      (no summary)")
            continue
        text = " ".join(summary.split())
        indent = pad + "      "
        while text:
            lines.append(indent + text[:width])
            text = text[width:]
    return "\n".join(lines)


def check_result(result, baseline=None):
    """Everything the scenario promised, as a list of problems. Empty means it held.

    A scenario with `expect_fallback=None` claims nothing about the outcome — whether this
    document yields a verifiable TOC depends on the document and the model, not on the
    retry logic. Those scenarios are checked against the baseline instead, which is the
    real invariant: injecting failures that are then retried must land on the same outcome
    an uninjected run reaches, differing only in how many attempts it took.
    """
    s, body = result.scenario, result.body
    problems = list(result.problems)

    def want(label, actual, expected):
        if actual != expected:
            problems.append(f"{label}: expected {expected!r}, got {actual!r}")

    want("attempts", body["attempts"], s.expect_attempts)
    want("extraction calls", result.extraction_calls, s.expect_attempts)

    # process_markdown() returns this; only the HTTP response_model strips it.
    accuracy = result.body["fallback_accuracy"]

    if s.expect_fallback is None:
        if baseline is not None:
            want("fallback_used vs baseline",
                 body["fallback_used"], baseline.body["fallback_used"])
            want("node count vs baseline", result.node_count, baseline.node_count)
            # Only under the offline stand-in, which is deterministic. A real model is not
            # guaranteed to re-extract an identical tree.
            if not result.real_llm and not baseline.real_llm:
                want("tree skeleton vs baseline",
                     tree_skeleton(body["structure"]),
                     tree_skeleton(baseline.body["structure"]))
    else:
        want("fallback_used", body["fallback_used"], s.expect_fallback)
        if s.expect_accuracy == "null":
            want("fallback_accuracy", accuracy, None)
        elif s.expect_accuracy == "score" and not isinstance(accuracy, (int, float)):
            problems.append(f"fallback_accuracy: expected a number, got {accuracy!r}")

    # Shape invariants follow the tree that actually came back, not the scenario's guess.
    if body["fallback_used"]:
        want("node count", result.node_count, result.page_count + 1)
    elif result.node_count <= 1:
        problems.append("node count: a real TOC should give more than one node")

    # What went to S3 must be what the caller got back, minus the key the service adds
    # on the way out. This is the only place the two are compared.
    want("s3 object vs returned result",
         result.payload, {k: v for k, v in body.items() if k != "output_s3_key"})

    # The saved file is the deliverable — it has to round-trip.
    with open(result.out_path, encoding="utf-8") as f:
        saved = json.load(f)
    want("saved structure", saved["structure"], body["structure"])
    want("saved attempts", saved["attempts"], body["attempts"])

    return problems


# ── pytest ────────────────────────────────────────────────────────────────────

@pytest.fixture
def md_document(tmp_path):
    """The document every scenario is run against — the same one script mode uses.

    Point MD_PATH at your own .md (or drop it over tests/fixtures/sample_document.md) and
    the whole suite indexes it. Every expectation below is derived from the document at
    run time — page count, node count, summary counts — so nothing here is tied to one
    file. If the configured path is missing, this falls back to the synthetic eight-page
    document so the suite still runs in a bare checkout.
    """
    configured = configured_md_path()
    if configured:
        return configured

    path = tmp_path / "quarterly-report.md"
    path.write_text(harness.PAGE_TAGGED_MD, encoding="utf-8")
    return str(path)


@pytest.mark.parametrize("scenario", SCENARIOS, ids=[s.name for s in SCENARIOS])
def test_scenario(scenario, md_document, tmp_path, monkeypatch):
    # A scenario that claims no outcome of its own needs the uninjected run to compare
    # against. Its own monkeypatch context, so the injector cannot leak between the two.
    baseline = None
    if scenario.expect_fallback is None and scenario is not BASELINE:
        with pytest.MonkeyPatch.context() as mp:
            baseline = run_scenario(
                BASELINE, md_document, str(tmp_path / "baseline"), mp, summaries=True
            )

    result = run_scenario(
        scenario, md_document, str(tmp_path / "out"), monkeypatch, summaries=True
    )

    assert check_result(result, baseline) == [], f"{scenario.name}: {scenario.what}"


def test_summaries_survive_a_retried_fallback(md_document, tmp_path, monkeypatch):
    """The combined case still gets page-level summaries: leaves from their own text,
    root rolled up from the leaves rather than from the whole document."""
    scenario = next(s for s in SCENARIOS if s.name == "retry_then_verification_fallback")
    result = run_scenario(
        scenario, md_document, str(tmp_path / "out"), monkeypatch, summaries=True
    )

    root = result.body["structure"][0]
    leaves = root["nodes"]
    assert result.llm_calls.get("node_summary") == len(leaves)
    assert result.llm_calls.get("rollup_summary") == 1
    assert "Page-level index" not in root["summary"], "the template is only a last resort"
    assert all(node["summary"] for _, node in harness.walk(result.body["structure"]))
    assert result.body["doc_description"]


def test_every_scenario_returns_200_and_a_tree(md_document, tmp_path, monkeypatch):
    """No injected failure may escape as an exception — call_service does not catch."""
    for scenario in SCENARIOS:
        with pytest.MonkeyPatch.context() as mp:
            result = run_scenario(
                scenario, md_document, str(tmp_path / scenario.name), mp,
                summaries=False,
            )
        assert result.body["structure"], scenario.name


# ── script mode ───────────────────────────────────────────────────────────────

def main():
    md_path = configured_md_path()
    if md_path is None:
        print(f"MD_PATH not found: {MD_PATH}")
        print("Point MD_PATH at the .md file you want indexed.")
        return 2

    out_dir = (OUTPUT_DIR if os.path.isabs(OUTPUT_DIR)
               else os.path.normpath(os.path.join(REPO_ROOT, OUTPUT_DIR)))

    print("=" * 78)
    print("RETRY + PAGE-LEVEL FALLBACK — end-to-end scenarios")
    print("=" * 78)
    print(f"document:  {md_path}")
    print(f"pages:     {virtual_page_count(md_path)} virtual pages")
    print(f"llm:       {'real model from config.yaml' if USE_REAL_LLM else 'offline stand-in'}")
    print(f"summaries: {'on' if ADD_SUMMARIES else 'off'}")
    print(f"output:    {out_dir}")
    print()

    scenarios = [s for name in TREES for s in SCENARIOS if s.name == name]

    results = []
    baseline = None
    for scenario in scenarios:
        print(f"--- {scenario.name} — {scenario.what}")
        # A fresh context per scenario: force_fallback patches verify_toc and
        # fail_extraction patches process_no_toc, and neither may leak into the next run.
        with pytest.MonkeyPatch.context() as mp:
            result = run_scenario(
                scenario, md_path, out_dir, mp,
                real_llm=USE_REAL_LLM, summaries=ADD_SUMMARIES, node_text=ADD_NODE_TEXT,
                model=MODEL,
            )
        # BASELINE is first in SCENARIOS, so it is already in hand by the time the
        # scenarios that compare against it run.
        if scenario is BASELINE:
            baseline = result
        result.problems = check_result(result, baseline)
        results.append(result)
        print(f"    attempts={result.body['attempts']} "
              f"fallback_used={result.body['fallback_used']} "
              f"fallback_accuracy={result.body['fallback_accuracy']} "
              f"nodes={result.node_count}")
        print(f"    saved -> {os.path.relpath(result.out_path, REPO_ROOT)}")
        print()

    print("=" * 78)
    header = f"{'scenario':34} {'att':>3} {'fallback':>8} {'accuracy':>9} {'nodes':>5}  result"
    print(header)
    print("-" * 78)
    for r in results:
        accuracy = r.body["fallback_accuracy"]
        print(f"{r.scenario.name:34} {r.body['attempts']:>3} "
              f"{str(r.body['fallback_used']):>8} "
              f"{'null' if accuracy is None else f'{accuracy:.2f}':>9} "
              f"{r.node_count:>5}  {'PASS' if not r.problems else 'FAIL'}")
    print("=" * 78)

    failed = [r for r in results if r.problems]
    if failed:
        print()
        for r in failed:
            print(f"{r.scenario.name}:")
            for p in r.problems:
                print(f"  - {p}")
        return 1

    # Every tree, with the summary each node actually got. On a fallback tree the leaves
    # are summarized from their own page and the root is rolled up from those leaf
    # summaries, so the root text is the one worth reading.
    for run in results:
        print()
        print("=" * 78)
        print(f"{run.scenario.name} — {run.node_count} nodes, "
              f"fallback={run.body['fallback_used']}, attempts={run.body['attempts']}")
        print("=" * 78)
        print(render_tree_with_summaries(run.body["structure"]))
        if run.body.get("doc_description"):
            print(f"\ndoc_description: {run.body['doc_description']}")
    print()

    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "scenario": r.scenario.name,
                "what": r.scenario.what,
                "max_tree_attempts": r.scenario.max_tree_attempts,
                "attempts": r.body["attempts"],
                "extraction_calls": r.extraction_calls,
                "fallback_used": r.body["fallback_used"],
                "fallback_accuracy": r.body["fallback_accuracy"],
                "page_count": r.page_count,
                "node_count": r.node_count,
                "llm_calls": r.llm_calls,
                "tree": os.path.relpath(r.out_path, out_dir),
            }
            for r in results
        ], f, indent=2)
    print(f"summary -> {os.path.relpath(summary_path, REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
