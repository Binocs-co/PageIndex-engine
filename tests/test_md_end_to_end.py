"""End-to-end test for indexing a markdown file through the PageIndex API.

Point it at any .md file and it runs the real pipeline: process_markdown(), the request
model, markdown_to_page_list(), meta_processor()'s verification + repair + mode cascade,
the service's extraction retry, tree assembly, and the JSON that would be written to S3.
Only two things are stubbed: S3 (mapped to local files) and, optionally, the LLM.

Run it on your own file
-----------------------
Set MD_PATH in the CONFIG block below to your .md file, adjust the other constants
if you want, then:

    python tests/test_md_end_to_end.py

An out-of-the-box run indexes the document twice and saves each tree to OUTPUT_DIR
as <name>_<tree>.json:

    original_method        nothing injected — the tree the pipeline builds on its own
    retry_then_success     attempt 1 raises, so this is what the first retry produced

For each tree it prints every node with the summary that node actually got, the provider
calls it cost broken down by stage, and the tree's structural invariants. USE_REAL_LLM
defaults to True, so the extraction, the verification and the summaries are all real.
Set USE_REAL_LLM = False for a free offline run, or trim TREES to run fewer.

Run it as a test suite
----------------------
    pytest tests/test_md_end_to_end.py -v

The pytest cases build their own markdown fixtures and always use the stand-in LLM,
so they need no network access and ignore the CONFIG block entirely.

Note: this exercises the *API* markdown path by calling the function the route calls,
process_markdown(), rather than going over HTTP — the route body is a single call to it.
That path splits markdown into virtual pages and runs the full PDF tree-generation
pipeline over them. The library's other markdown
entry point, pageindex.page_index_md.md_to_tree(), is a separate deterministic
header-regex parser with no verification step and is not covered here.
"""
import asyncio
import importlib
import json
import os
import re
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pageindex.utils as pageindex_utils

pi = importlib.import_module("pageindex.page_index")


# ══ CONFIG — edit these, then run: python tests/test_md_end_to_end.py ══════════

# The markdown file to index — replace this with your own file.
# Relative paths resolve from the repository root.
MD_PATH = "/Users/hardikjoshi/Desktop/LLM_engine/llm-engine/test_files/ocr/Novature.md"

# False: use the offline stand-in LLM (no API key or credits needed).
# True:  call the real model configured in pageindex/config.yaml — real extraction, real
#        verification, real summaries. This costs money.
USE_REAL_LLM = True

# Model override. Setting this implies USE_REAL_LLM. None keeps the configured model.
MODEL = None

# Virtual page size, used only for markdown WITHOUT <page_number=N> tags.
# Tagged markdown always gets one virtual page per tag. Valid range 500-10000.
TOKENS_PER_PAGE = 2000

# Generate a summary for every node plus a document description (costs one LLM
# call per node, so set it False for a quick structural check).
ADD_SUMMARIES = True

# Include each node's raw page text in the saved JSON. Off by default because it
# duplicates the whole document into the tree.
ADD_NODE_TEXT = False

# The trees a script run produces, in order, as (name, fail_first):
#   fail_first  how many extraction attempts raise before one is let through, so
#               `1` is "the tree the first retry produced"
# Trim this tuple to run fewer. Each entry is one full pass over the document.
TREES = (
    ("original_method",    0),
    ("retry_then_success", 1),
)

# Where to write the result JSON.
OUTPUT_DIR = "./results"

# ══════════════════════════════════════════════════════════════════════════════


# ── A deterministic stand-in for the LLM ──────────────────────────────────────
#
# Every prompt in the pipeline is answered by doing the work the model would do,
# using regex over the text the prompt itself carries. That keeps the run offline
# while still driving the real code paths — including the verification judge, which
# genuinely checks whether a title is on the page it was assigned to.

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_PAGE_SPLIT_RE = re.compile(r"<physical_index_(\d+)>")
_FENCE_RE = re.compile(r"^\s*```")


def _normalize(text):
    return re.sub(r"\s+", " ", text or "").strip().lower()


def _pages_in_prompt(text):
    """Split prompt text tagged with <physical_index_N> into {page_number: page_text}."""
    parts = _PAGE_SPLIT_RE.split(text)
    pages = {}
    # parts alternates: [pre, "1", body, "1", between, "2", body, "2", ...]
    for i in range(1, len(parts) - 1, 2):
        page_number = int(parts[i])
        body = parts[i + 1]
        # The closing tag repeats the number, so the first body wins.
        pages.setdefault(page_number, body)
    return pages


def _headings_outside_code_fences(page_text):
    """Markdown headings on a page, skipping anything inside a fenced code block."""
    headings = []
    in_fence = False
    for line in page_text.split("\n"):
        if _FENCE_RE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        match = _HEADING_RE.match(line)
        if match:
            headings.append((len(match.group(1)), match.group(2).strip()))
    return headings


class StandInLLM:
    """Answers each pipeline prompt deterministically. Counts calls by kind.

    Handlers that ignore their prompt still keep the parameter where the pipeline
    passes it by keyword — the signatures below have to match the real functions
    they stand in for, so unused parameters here are deliberate.
    """

    def __init__(self):
        self.calls = {}

    def _count(self, kind):
        self.calls[kind] = self.calls.get(kind, 0) + 1

    # -- individual prompt handlers ------------------------------------------

    def _toc_detected(self):
        self._count("toc_detector")
        return '{"thinking": "stand-in", "toc_detected": "no"}'

    def _generate_toc(self, prompt, previous=None):
        """Build a real nested TOC from the markdown headings in the given pages."""
        self._count("generate_toc")
        given = prompt.split("Given text\n:", 1)[-1]
        # generate_toc_continue appends the previous tree after the text
        given = given.split("\nPrevious tree structure\n:", 1)[0]

        already = set()
        if previous:
            already = {(_normalize(item.get("title")), item.get("physical_index")) for item in previous}

        counters = []
        items = []
        for page_number in sorted(_pages_in_prompt(given)):
            for level, title in _headings_outside_code_fences(_pages_in_prompt(given)[page_number]):
                # maintain a dotted structure code from the heading depth
                if level > len(counters):
                    counters.extend([0] * (level - len(counters)))
                del counters[level:]
                counters[level - 1] += 1
                structure = ".".join(str(c) for c in counters if c)
                physical = f"<physical_index_{page_number}>"
                if (_normalize(title), physical) in already:
                    continue
                items.append({
                    "structure": structure,
                    "title": title,
                    "physical_index": physical,
                })
        return json.dumps(items)

    def _title_appears(self, prompt):
        """The verification judge: is this title actually on the page it points at?"""
        self._count("verify_title")
        title, page_text = _split_title_and_page(prompt)
        answer = "yes" if _normalize(title) and _normalize(title) in _normalize(page_text) else "no"
        return json.dumps({"thinking": "stand-in substring match", "answer": answer})

    def _title_at_start(self, prompt):
        self._count("title_at_start")
        title, page_text = _split_title_and_page(prompt)
        head = _normalize(page_text)[:200]
        return json.dumps({
            "thinking": "stand-in prefix match",
            "start_begin": "yes" if _normalize(title) and _normalize(title) in head else "no",
        })

    def _fix_index(self, prompt):
        """Find the page that really does contain this title."""
        self._count("fix_index")
        title = prompt.split("Section Title:\n", 1)[-1].split("\nDocument pages:\n", 1)[0]
        content = prompt.split("\nDocument pages:\n", 1)[-1]
        pages = _pages_in_prompt(content)
        for page_number in sorted(pages):
            if _normalize(title) and _normalize(title) in _normalize(pages[page_number]):
                return json.dumps({"thinking": "found", "physical_index": f"<physical_index_{page_number}>"})
        first = min(pages) if pages else 1
        return json.dumps({"thinking": "not found, using first page",
                           "physical_index": f"<physical_index_{first}>"})

    def _node_summary(self, prompt):
        self._count("node_summary")
        text = prompt.split("Partial Document Text:", 1)[-1]
        text = text.split("Directly return the description", 1)[0]
        snippet = _normalize(text)[:160]
        return f"[stand-in summary] {snippet}"

    def _doc_description(self):
        self._count("doc_description")
        return "[stand-in description] A markdown document indexed end to end by the test harness."

    # -- dispatch -----------------------------------------------------------

    def respond(self, prompt):
        if "toc_detected" in prompt:
            return self._toc_detected()
        if "continue the tree structure" in prompt:
            previous = _extract_previous_tree(prompt)
            return self._generate_toc(prompt, previous)
        if "generate the tree structure of the document" in prompt:
            return self._generate_toc(prompt)
        if "check if the given section appears or starts in the given page_text" in prompt:
            return self._title_appears(prompt)
        if "starts in the beginning of the given page_text" in prompt:
            return self._title_at_start(prompt)
        if "find the physical index of the start page of the section" in prompt:
            return self._fix_index(prompt)
        if "generate a description of the partial document" in prompt:
            return self._node_summary(prompt)
        if "generate a one-sentence description for the document" in prompt:
            return self._doc_description()
        self._count("unhandled")
        return "{}"


def _split_title_and_page(prompt):
    match = re.search(
        r"The given section title is (?P<title>.*?)\.\s*\n\s*The given page_text is (?P<page>.*?)\.\s*\n\s*\n\s*(?:Reply format|reply format)",
        prompt,
        re.DOTALL,
    )
    if not match:
        return "", ""
    return match.group("title").strip(), match.group("page")


def _extract_previous_tree(prompt):
    tail = prompt.split("\nPrevious tree structure\n:", 1)
    if len(tail) < 2:
        return None
    try:
        return json.loads(tail[1])
    except json.JSONDecodeError:
        return None


def install_stand_in_llm(monkeypatch=None):
    """Replace the two LLM entry points everywhere they are referenced.

    Returns the StandInLLM so callers can inspect its call counts.
    """
    llm = StandInLLM()

    def sync_completion(model, prompt, chat_history=None, return_finish_reason=False):
        out = llm.respond(prompt)
        return (out, "finished") if return_finish_reason else out

    async def async_completion(model, prompt, return_finish_reason=False):
        out = llm.respond(prompt)
        return (out, "finished") if return_finish_reason else out

    targets = [pageindex_utils, pi]
    for module in targets:
        if monkeypatch is not None:
            monkeypatch.setattr(module, "llm_completion", sync_completion, raising=False)
            monkeypatch.setattr(module, "llm_acompletion", async_completion, raising=False)
        else:
            module.llm_completion = sync_completion
            module.llm_acompletion = async_completion
    return llm


# ── Counting real provider calls ──────────────────────────────────────────────

# Which pipeline stage a prompt belongs to. The names match StandInLLM's counters, so an
# offline run and a real run are directly comparable.
_PROMPT_KINDS = (
    ("toc_detected", "toc_detector"),
    ("continue the tree structure", "generate_toc"),
    ("generate the tree structure of the document", "generate_toc"),
    ("check if the given section appears or starts in the given page_text", "verify_title"),
    ("starts in the beginning of the given page_text", "title_at_start"),
    ("find the physical index of the start page of the section", "fix_index"),
    ("generate a description of the partial document", "node_summary"),
    ("generate a one-sentence description for the document", "doc_description"),
)


class RealLLMCounter:
    """Counts real provider calls by stage without changing a single answer.

    A real run leaves the model untouched, which also left the call counts empty — there
    was no way to see that the summaries came from the provider, or what each tree cost.
    Both entry points are wrapped and every call delegates straight through, so the
    numbers belong to the real run. `.calls` is shaped like StandInLLM's.
    """

    def __init__(self):
        self.calls = {}

    def tally(self, prompt):
        for needle, kind in _PROMPT_KINDS:
            if needle in prompt:
                self.calls[kind] = self.calls.get(kind, 0) + 1
                return
        self.calls["other"] = self.calls.get("other", 0) + 1


def count_real_llm_calls(monkeypatch=None):
    """Install the pass-through counter over the real LLM entry points."""
    counter = RealLLMCounter()
    real_sync = pageindex_utils.llm_completion
    real_async = pageindex_utils.llm_acompletion

    def counting_sync(model, prompt, chat_history=None, return_finish_reason=False):
        counter.tally(prompt)
        return real_sync(model, prompt, chat_history, return_finish_reason)

    async def counting_async(model, prompt, return_finish_reason=False):
        counter.tally(prompt)
        return await real_async(model, prompt, return_finish_reason)

    for module in (pageindex_utils, pi):
        _patch(module, "llm_completion", counting_sync, monkeypatch)
        _patch(module, "llm_acompletion", counting_async, monkeypatch)
    return counter


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


# ── The end-to-end run ────────────────────────────────────────────────────────

def index_markdown_file(
    md_path,
    monkeypatch=None,
    real_llm=False,
    tokens_per_page=2000,
    summaries=False,
    node_text=False,
    model=None,
    max_tree_attempts=None,
):
    """Index a local .md file by calling the function the route calls.

    No TestClient and no HTTP. The route's body is one line —
    `await process_markdown(payload, get_s3_session(), get_s3_bucket())` — so calling
    process_markdown() directly exercises the same pipeline without the transport. The
    payload is built with the real MarkdownPageIndexRequest, so field defaults, bounds and
    coercion are still the API's own.

    One consequence: a failure inside the pipeline raises out of here instead of arriving
    as a 500 the way it would through the route.

    Two things are substituted, neither of them the code under test — S3 (LocalS3, so the
    markdown comes off disk and the tree is captured in memory; `s3_session` is unused and
    passed as None) and the LLM (the offline stand-in, unless real_llm).

    Returns (result, local_s3, stand_in_llm_or_None).
    """
    from api.routers.pageindex import MarkdownPageIndexRequest
    from api.services import pageindex_service as svc

    # Real runs get a pass-through counter, so provider calls stay visible either way.
    llm = (count_real_llm_calls(monkeypatch) if real_llm
           else install_stand_in_llm(monkeypatch))

    local_s3 = LocalS3(md_path)
    _patch(svc, "read_markdown_from_s3", local_s3.read, monkeypatch)
    _patch(svc, "upload_tree_to_s3", local_s3.write, monkeypatch)

    stem = os.path.splitext(os.path.basename(md_path))[0]
    fields = {
        "input_s3_key": f"incoming/{os.path.basename(md_path)}",
        "output_s3_key": f"trees/{stem}.json",
        "tokens_per_page": tokens_per_page,
        "if_add_node_summary": "yes" if summaries else "no",
        "if_add_doc_description": "yes" if summaries else "no",
        "if_add_node_text": "yes" if node_text else "no",
    }
    if model:
        fields["model"] = model
    if max_tree_attempts is not None:
        fields["max_tree_attempts"] = max_tree_attempts

    payload = MarkdownPageIndexRequest(**fields)
    result = asyncio.run(svc.process_markdown(payload, None, "test-bucket"))
    return result, local_s3, llm


def _patch(target, name, value, monkeypatch):
    if monkeypatch is not None:
        monkeypatch.setattr(target, name, value)
    else:
        setattr(target, name, value)


def fail_extraction(monkeypatch, times=None):
    """Make process_no_toc raise, so the service's retry path is exercised.

    `times=None` fails every attempt; an integer fails only the first N calls and then
    delegates to the real function. The exception mirrors the real failure class the retry
    exists for: generate_toc_init / generate_toc_continue raise on a truncated completion
    (page_index.py:544, :578).

    Returns a dict whose "calls" key counts how many extraction attempts were made — that
    is what proves the attempt budget was honoured.
    """
    real_process_no_toc = pi.process_no_toc
    state = {"calls": 0}

    async def flaky_process_no_toc(page_list, start_index=1, model=None, logger=None):
        state["calls"] += 1
        if times is None or state["calls"] <= times:
            raise Exception("finish reason: max_output_reached")
        return await real_process_no_toc(
            page_list, start_index=start_index, model=model, logger=logger
        )

    _patch(pi, "process_no_toc", flaky_process_no_toc, monkeypatch)
    return state


# ── Tree inspection ───────────────────────────────────────────────────────────

def walk(nodes, depth=0):
    for node in nodes:
        yield depth, node
        yield from walk(node.get("nodes") or [], depth + 1)


def count_nodes(nodes):
    return sum(1 for _ in walk(nodes))


def node_key_sets(body):
    """Union of the field names appearing on any node in the tree."""
    keys = set()
    for _depth, node in walk(body["structure"]):
        keys.update(k for k in node if k != "nodes")
    return keys


def render_toc(nodes, max_lines=60):
    lines = []
    for depth, node in walk(nodes):
        if len(lines) >= max_lines:
            lines.append(f"    ... ({count_nodes(nodes) - max_lines} more nodes)")
            break
        node_id = node.get("node_id", "----")
        lines.append(
            "  " * depth
            + f"[{node_id}] {node['title']}  (pages {node['start_index']}-{node['end_index']})"
        )
    return "\n".join(lines)


def render_tree_with_summaries(nodes, width=96):
    """The tree with each node's summary underneath it — the point of a real-LLM run."""
    lines = []
    for depth, node in walk(nodes):
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


def check_tree_invariants(structure, page_count):
    """Returns a list of problems; empty means the tree is well formed."""
    problems = []
    if not structure:
        return ["tree is empty"]

    seen_ids = []
    for _depth, node in walk(structure):
        label = f"{node.get('title')!r}"
        for field in ("title", "start_index", "end_index"):
            if node.get(field) is None:
                problems.append(f"{label}: missing {field}")
        start, end = node.get("start_index"), node.get("end_index")
        if isinstance(start, int) and isinstance(end, int):
            if start > end:
                problems.append(f"{label}: start_index {start} > end_index {end}")
            if start < 1 or end > page_count:
                problems.append(
                    f"{label}: pages {start}-{end} outside the document (1-{page_count})"
                )
        if "node_id" in node:
            seen_ids.append(node["node_id"])

    if seen_ids:
        if len(set(seen_ids)) != len(seen_ids):
            problems.append("duplicate node_id values")
        expected = [str(i).zfill(4) for i in range(len(seen_ids))]
        if seen_ids != expected:
            problems.append("node_id values are not a pre-order 0000..N sequence")
    return problems


# ── pytest cases (self-contained fixtures, always offline) ────────────────────

PAGE_TAGGED_MD = "\n".join(
    f"<page_number={i}>\n" + body + f"\n</page_number={i}>"
    for i, body in enumerate(
        [
            "# Quarterly Report\n\nOpening remarks for the quarter.",
            "## Revenue\n\nRevenue grew across all segments.",
            "### North America\n\nStrong performance in the north.",
            "### Europe\n\nSteady growth in Europe.",
            "## Costs\n\nOperating costs were contained.",
            "## Outlook\n\nWe expect continued growth.",
            "# Appendix\n\nSupporting tables follow.",
            "## Methodology\n\nHow the figures were computed.",
        ],
        start=1,
    )
)

PLAIN_MD = "\n\n".join(
    [f"# Section {i}\n\n" + ("filler sentence for this section. " * 40) for i in range(1, 7)]
)


@pytest.fixture
def page_tagged_md(tmp_path):
    path = tmp_path / "quarterly-report.md"
    path.write_text(PAGE_TAGGED_MD, encoding="utf-8")
    return str(path)


@pytest.fixture
def plain_md(tmp_path):
    path = tmp_path / "plain-notes.md"
    path.write_text(PLAIN_MD, encoding="utf-8")
    return str(path)


def test_page_tagged_markdown_indexes_end_to_end(page_tagged_md, monkeypatch):
    body, s3, llm = index_markdown_file(page_tagged_md, monkeypatch=monkeypatch)

    assert body["attempts"] == 1, "a clean run should need one attempt"
    assert check_tree_invariants(body["structure"], page_count=8) == []
    assert count_nodes(body["structure"]) > 1

    titles = [node["title"] for _, node in walk(body["structure"])]
    assert "Quarterly Report" in titles
    assert "Revenue" in titles

    # nesting survived: at least one node has children
    assert any(node.get("nodes") for _, node in walk(body["structure"]))

    # the verification judge really ran
    assert llm.calls.get("verify_title", 0) > 0


def test_output_is_written_to_s3_key(page_tagged_md, monkeypatch):
    body, s3, _ = index_markdown_file(page_tagged_md, monkeypatch=monkeypatch)

    assert s3.uploaded_key == "trees/quarterly-report.json"
    assert s3.uploaded_data["structure"] == body["structure"]
    assert set(s3.uploaded_data) == {"doc_description", "structure", "attempts"}
    assert s3.uploaded_data["attempts"] == 1


def test_markdown_without_page_tags_is_chunked(plain_md, monkeypatch):
    body, _, _ = index_markdown_file(plain_md, monkeypatch=monkeypatch, tokens_per_page=500)

    assert body["structure"], "plain markdown should still produce a tree"
    # every page index must be inside the chunk count the splitter produced
    from api.services.pageindex_service import markdown_to_page_list

    page_count = len(markdown_to_page_list(PLAIN_MD, 500, "gpt-4o-2024-11-20"))
    assert check_tree_invariants(body["structure"], page_count=page_count) == []


def test_summaries_and_description(page_tagged_md, monkeypatch):
    body, _, llm = index_markdown_file(page_tagged_md, monkeypatch=monkeypatch, summaries=True)

    assert body["doc_description"], "a description should be generated"
    assert all("summary" in node for _, node in walk(body["structure"]))
    # text was requested off, so it must not leak into the output
    assert all("text" not in node for _, node in walk(body["structure"]))
    assert llm.calls.get("node_summary", 0) == count_nodes(body["structure"])


# ── Extraction retry ─────────────────────────────────────────────────────────

def test_extraction_retry_succeeds_on_second_attempt(page_tagged_md, monkeypatch):
    """A raised extraction failure is retried, and the retry produces a normal tree."""
    state = fail_extraction(monkeypatch, times=1)

    body, s3, _ = index_markdown_file(page_tagged_md, monkeypatch=monkeypatch)

    assert state["calls"] == 2, "the first attempt failed, so a second should have run"
    assert body["attempts"] == 2
    assert s3.uploaded_data["attempts"] == 2
    assert check_tree_invariants(body["structure"], page_count=8) == []
    titles = [node["title"] for _, node in walk(body["structure"])]
    assert "Revenue" in titles, "the retry built a real TOC, not a page-level one"


def test_all_attempts_failing_propagates(page_tagged_md, monkeypatch):
    """Once the budget is spent the last failure propagates — the route turns it into a
    500, exactly as it did before the retry existed."""
    state = fail_extraction(monkeypatch)

    with pytest.raises(Exception, match="max_output_reached"):
        index_markdown_file(page_tagged_md, monkeypatch=monkeypatch)

    assert state["calls"] == 2, "the default budget is two attempts"


def test_max_tree_attempts_one_disables_the_retry(page_tagged_md, monkeypatch):
    state = fail_extraction(monkeypatch)

    with pytest.raises(Exception, match="max_output_reached"):
        index_markdown_file(page_tagged_md, monkeypatch=monkeypatch, max_tree_attempts=1)

    assert state["calls"] == 1, "the budget was one attempt, so nothing should be retried"


def test_max_tree_attempts_is_capped_at_the_ceiling(page_tagged_md, monkeypatch):
    """The request field is bounded by the model (le=3), so the ceiling is reachable but
    never exceeded."""
    state = fail_extraction(monkeypatch)

    with pytest.raises(Exception, match="max_output_reached"):
        index_markdown_file(page_tagged_md, monkeypatch=monkeypatch, max_tree_attempts=3)

    assert state["calls"] == 3


def test_env_var_sets_the_default_budget(page_tagged_md, monkeypatch):
    """PAGEINDEX_MAX_TREE_ATTEMPTS is the deploy-time default, used when the request is
    silent; a garbage value is ignored rather than fatal."""
    monkeypatch.setenv("PAGEINDEX_MAX_TREE_ATTEMPTS", "3")
    state = fail_extraction(monkeypatch)

    with pytest.raises(Exception, match="max_output_reached"):
        index_markdown_file(page_tagged_md, monkeypatch=monkeypatch)

    assert state["calls"] == 3

    monkeypatch.setenv("PAGEINDEX_MAX_TREE_ATTEMPTS", "not-a-number")
    state = fail_extraction(monkeypatch)

    with pytest.raises(Exception, match="max_output_reached"):
        index_markdown_file(page_tagged_md, monkeypatch=monkeypatch)

    assert state["calls"] == 2, "an unparseable env value falls back to the default"


# ── Runner (reads the CONFIG block above) ─────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _preflight_real_llm(model):
    """Fail early with a clear message if the configured model is unreachable."""
    import litellm

    try:
        litellm.completion(
            model=(model or "").removeprefix("litellm/"),
            messages=[{"role": "user", "content": "reply with the single word ok"}],
            temperature=0,
            num_retries=0,
        )
        return None
    except Exception as e:
        return f"{type(e).__name__}: {str(e)[:200]}"


def main():
    md_path = MD_PATH if os.path.isabs(MD_PATH) else os.path.normpath(os.path.join(REPO_ROOT, MD_PATH))

    if not os.path.isfile(md_path):
        print(f"MD_PATH does not point at a file: {md_path}")
        print("Edit MD_PATH in the CONFIG block at the top of this file.")
        return 2
    if not md_path.lower().endswith((".md", ".markdown")):
        print(f"MD_PATH must be a .md or .markdown file: {md_path}")
        return 2

    real_llm = USE_REAL_LLM or bool(MODEL)

    from api.services.pageindex_service import markdown_to_page_list
    from pageindex.utils import ConfigLoader

    opt = ConfigLoader().load({"model": MODEL} if MODEL else {})

    with open(md_path, encoding="utf-8") as f:
        content = f.read()
    tagged = "<page_number=" in content
    page_list = markdown_to_page_list(content, TOKENS_PER_PAGE, opt.model)

    print("=" * 72)
    print("PageIndex end-to-end markdown run")
    print("=" * 72)
    print(f"file:          {md_path}  ({len(content):,} chars)")
    print(f"paging:        {'<page_number=N> tags' if tagged else f'token chunks of ~{TOKENS_PER_PAGE}'}")
    print(f"virtual pages: {len(page_list)}")
    print(f"llm:           {'real — ' + str(opt.model) if real_llm else 'offline stand-in'}")
    print(f"summaries:     {'on' if ADD_SUMMARIES else 'off'}")
    print(f"node text:     {'on' if ADD_NODE_TEXT else 'off'}")
    print(f"trees:         {', '.join(name for name, _ in TREES)}")
    print()

    if real_llm:
        error = _preflight_real_llm(opt.model)
        if error:
            print(f"The configured model is unreachable: {error}")
            print("Set USE_REAL_LLM = False to use the offline stand-in LLM.")
            return 2

    out_dir = OUTPUT_DIR if os.path.isabs(OUTPUT_DIR) else os.path.normpath(os.path.join(REPO_ROOT, OUTPUT_DIR))
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(os.path.basename(md_path))[0]

    all_problems = []
    for name, fail_first in TREES:
        # A fresh patch context per tree: fail_extraction patches process_no_toc, and that
        # must not leak into the next run.
        with pytest.MonkeyPatch.context() as mp:
            injector = fail_extraction(mp, times=fail_first) if fail_first else None
            body, local_s3, llm = index_markdown_file(
                md_path,
                monkeypatch=mp,
                real_llm=real_llm,
                tokens_per_page=TOKENS_PER_PAGE,
                summaries=ADD_SUMMARIES,
                node_text=ADD_NODE_TEXT,
                model=MODEL,
            )

        structure = body["structure"]
        total = count_nodes(structure)

        problems = check_tree_invariants(structure, page_count=len(page_list))
        all_problems += [f"{name}: {p}" for p in problems]

        print("=" * 72)
        print(f"{name}  —  {total} nodes")
        print("=" * 72)
        print(f"attempts:          {body['attempts']}"
              + (f"  (extraction entered {injector['calls']}x)" if injector else ""))
        if llm:
            print(f"llm calls:         {dict(sorted(llm.calls.items()))}")
        print()
        # Each node with the summary it actually got.
        print(render_tree_with_summaries(structure))
        if body.get("doc_description"):
            print(f"\ndoc_description: {body['doc_description']}")
        if problems:
            print(f"\nFAILED — {len(problems)} problem(s):")
            for problem in problems:
                print(f"  - {problem}")

        out_path = os.path.join(out_dir, f"{stem}_{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(local_s3.uploaded_data, f, indent=2, ensure_ascii=False)
        print(f"\nsaved -> {os.path.relpath(out_path, REPO_ROOT)}")
        print()

    print("-" * 72)
    if all_problems:
        print(f"FAILED — {len(all_problems)} problem(s) across {len(TREES)} tree(s):")
        for problem in all_problems:
            print(f"  - {problem}")
    else:
        print(f"PASSED — {len(TREES)} tree(s), every one structurally valid:")
        print("  - every node has a title and a valid page range")
        print(f"  - all page indices inside 1-{len(page_list)}")
        print("  - node_id values are a unique pre-order sequence")
    print("-" * 72)

    return 1 if all_problems else 0


if __name__ == "__main__":
    sys.exit(main())
