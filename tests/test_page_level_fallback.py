"""Tests for pageindex_fallback: PageLevelIndex and FallbackTreeBuilder.

`pageindex` itself is unmodified, so it still does what it always did — cascade to weaker
extraction modes and finally raise `Processing failed`. pageindex_fallback re-orchestrates
the library's own building blocks (process_no_toc, verify_toc, fix_incorrect_toc_with_
retries) and substitutes a one-node-per-page index instead of raising. The fallback is
unconditional: there is no threshold and no way to opt out.

No network / LLM calls: PageLevelIndex is pure, and the builder tests stub the library
functions it calls.
"""
import asyncio
import importlib
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# `pageindex/__init__.py` does `from .page_index import *`, so the package attribute
# `pageindex.page_index` is the *function*, not the module — import the module explicitly.
pi = importlib.import_module("pageindex.page_index")
from pageindex.utils import ConfigLoader, format_structure, write_node_id

from pageindex_fallback import (
    FallbackState,
    FallbackTreeBuilder,
    PageLevelIndex,
)


def make_pages(n):
    return [(f"text of page {i}", 10) for i in range(1, n + 1)]


def walk(nodes):
    for node in nodes:
        yield node
        yield from walk(node.get("nodes") or [])


class StubLogger:
    def __init__(self):
        self.records = []

    def info(self, msg, *args):
        self.records.append(msg % args if args else msg)

    warning = error = debug = info


# ── PageLevelIndex: the flat list ─────────────────────────────────────────────

def test_flat_list_has_one_item_per_page_plus_a_root():
    flat = PageLevelIndex(make_pages(10), "doc").flat_toc()

    assert len(flat) == 11
    assert flat[0] == {"structure": "1", "title": "doc", "physical_index": 1}
    assert [item["structure"] for item in flat[1:]] == [f"1.{i}" for i in range(1, 11)]
    assert [item["title"] for item in flat[1:]] == [f"Page {i}" for i in range(1, 11)]


def test_appear_start_is_preset_so_no_llm_is_needed():
    """post_processing derives each page's range from the next item's appear_start."""
    flat = PageLevelIndex(make_pages(4), "doc").flat_toc()

    assert "appear_start" not in flat[0], "the root's range is widened, not derived"
    # 'no' on page 1 only: that flag sets the ROOT's end_index and nothing else.
    assert [item["appear_start"] for item in flat[1:]] == ["no", "yes", "yes", "yes"]


def test_flat_list_keeps_absolute_page_numbers():
    flat = PageLevelIndex(make_pages(3), "sub", start_index=7).flat_toc()

    assert flat[0]["physical_index"] == 7
    assert [item["physical_index"] for item in flat[1:]] == [7, 8, 9]
    assert [item["title"] for item in flat[1:]] == ["Page 7", "Page 8", "Page 9"]


def test_empty_page_list_gives_an_empty_index():
    index = PageLevelIndex([], "doc")

    assert index.flat_toc() == []
    assert index.tree() == []
    assert index.page_count == 0


# ── PageLevelIndex: the assembled tree ────────────────────────────────────────

def test_tree_is_a_root_spanning_the_document_over_single_page_leaves():
    index = PageLevelIndex(make_pages(10), "doc")
    tree = index.tree()

    assert index.page_count == 10
    assert index.end_index == 10
    assert len(tree) == 1
    root = tree[0]
    assert (root["start_index"], root["end_index"]) == (1, 10)

    leaves = root["nodes"]
    assert len(leaves) == 10
    assert [(n["start_index"], n["end_index"]) for n in leaves] == [(i, i) for i in range(1, 11)]
    assert not any(n.get("nodes") for n in leaves)
    assert sum(1 for _ in walk(tree)) == 11


def test_tree_keeps_absolute_page_numbers_for_a_subrange():
    index = PageLevelIndex(make_pages(3), "sub", start_index=7)
    tree = index.tree()

    assert index.end_index == 9
    assert (tree[0]["start_index"], tree[0]["end_index"]) == (7, 9)
    assert [(n["start_index"], n["end_index"]) for n in tree[0]["nodes"]] == [(7, 7), (8, 8), (9, 9)]


def test_node_ids_and_field_order_work_on_a_page_level_tree():
    tree = PageLevelIndex(make_pages(3), "doc").tree()
    write_node_id(tree)
    tree = format_structure(tree, order=["title", "node_id", "start_index", "end_index", "nodes"])

    ids = [node["node_id"] for node in walk(tree)]
    assert ids == ["0000", "0001", "0002", "0003"]
    assert list(tree[0]) == ["title", "node_id", "start_index", "end_index", "nodes"]
    assert list(tree[0]["nodes"][0]) == ["title", "node_id", "start_index", "end_index"]


def test_widen_root_is_a_no_op_on_an_empty_tree():
    assert PageLevelIndex.widen_root([], 10) == []


def test_root_summary_template_is_available_without_a_page_list():
    """The summary step needs the text but does not hold the pages."""
    text = PageLevelIndex.root_summary_for("report", 12)

    assert "report" in text and "12 pages" in text
    assert text == PageLevelIndex(make_pages(12), "report").root_summary()


# ── FallbackTreeBuilder ───────────────────────────────────────────────────────

def stub_library(monkeypatch, accuracy, unfixable=False, nothing_to_fix=False):
    """Stub the library functions the builder calls, and count the calls.

    The stubs keep every parameter of the functions they replace, because the builder
    passes several of them by keyword — the unused ones are deliberate.

    incorrect_results mirrors what the real verify_toc returns: empty only at 100%
    accuracy, otherwise at least one failing item, which routes the run into the repair
    pass. unfixable=True makes the repair pass hand its mismatches back unfixed — the
    trigger for the fallback now that there is no threshold. nothing_to_fix=True reports a
    failing accuracy with an *empty* mismatch list, which is what the real verify_toc
    returns from its early exit when the extraction never reached halfway through the
    document; there is nothing to repair, so that falls back too.
    """
    calls = {"process_no_toc": 0, "verify_toc": 0, "fix": 0, "boundary_check": 0}
    incorrect_results = (
        []
        if accuracy == 1.0 or nothing_to_fix
        else [{"list_index": 0, "answer": "no", "title": "Section One", "page_number": 1}]
    )

    async def fake_process_no_toc(page_list, start_index=1, model=None, logger=None):
        calls["process_no_toc"] += 1
        return [{"structure": "1", "title": "Section One", "physical_index": start_index}]

    async def fake_verify_toc(page_list, list_result, start_index=1, N=None, model=None):
        calls["verify_toc"] += 1
        return accuracy, list(incorrect_results)

    async def fake_fix_with_retries(toc_with_page_number, page_list, incorrect,
                                    start_index=1, max_attempts=3, model=None, logger=None):
        calls["fix"] += 1
        return toc_with_page_number, list(incorrect) if unfixable else []

    async def fake_boundary_check(structure, page_list, model=None, logger=None):
        calls["boundary_check"] += 1
        for item in structure:
            item.setdefault("appear_start", "no")
        return structure

    monkeypatch.setattr(pi, "process_no_toc", fake_process_no_toc)
    monkeypatch.setattr(pi, "verify_toc", fake_verify_toc)
    monkeypatch.setattr(pi, "fix_incorrect_toc_with_retries", fake_fix_with_retries)
    monkeypatch.setattr(pi, "check_title_appearance_in_start_concurrent", fake_boundary_check)
    return calls


def make_builder(page_list, logger=None, doc_title="doc", start_index=1):
    return FallbackTreeBuilder(
        page_list, ConfigLoader().load({}), logger or StubLogger(),
        doc_title=doc_title, start_index=start_index,
    )


def test_a_clean_verification_is_accepted(monkeypatch):
    stub_library(monkeypatch, accuracy=1.0)

    extraction = asyncio.run(make_builder(make_pages(10)).extract_verified_toc())

    assert extraction.usable
    assert [item["title"] for item in extraction.flat_toc] == ["Section One"]
    assert (extraction.accuracy, extraction.unfixed_items) == (1.0, [])


def test_a_repaired_toc_is_accepted_however_low_the_score_was(monkeypatch):
    """No threshold: what matters is that the repair pass placed every item."""
    calls = stub_library(monkeypatch, accuracy=0.1)

    extraction = asyncio.run(make_builder(make_pages(10)).extract_verified_toc())

    assert extraction.usable, "a fully repaired TOC is not a fallback"
    assert calls["fix"] == 1
    assert (extraction.accuracy, extraction.unfixed_items) == (0.1, [])


def test_no_usable_toc_when_the_repair_pass_falls_short(monkeypatch):
    stub_library(monkeypatch, accuracy=0.8, unfixable=True)

    extraction = asyncio.run(make_builder(make_pages(10)).extract_verified_toc())

    assert not extraction.usable
    assert extraction.flat_toc is None
    assert extraction.accuracy == 0.8
    assert len(extraction.unfixed_items) == 1


def test_no_usable_toc_when_there_is_nothing_to_repair(monkeypatch):
    """verify_toc's early exit: a failing accuracy with no mismatches to hand the repairer."""
    calls = stub_library(monkeypatch, accuracy=0.0, nothing_to_fix=True)

    extraction = asyncio.run(make_builder(make_pages(10)).extract_verified_toc())

    assert not extraction.usable
    assert (extraction.accuracy, extraction.unfixed_items) == (0.0, [])
    assert calls["fix"] == 0, "there was nothing to repair"


def test_accepted_toc_becomes_a_normal_tree(monkeypatch):
    calls = stub_library(monkeypatch, accuracy=1.0)

    tree, fallback = asyncio.run(make_builder(make_pages(10)).build())

    assert fallback == FallbackState()
    assert fallback.used is False
    assert [node["title"] for node in walk(tree)] == ["Section One"]
    assert calls["boundary_check"] == 1, "the normal path still asks where titles start"


def test_failed_verification_becomes_a_page_level_tree(monkeypatch):
    stub_library(monkeypatch, accuracy=0.4, unfixable=True)
    logger = StubLogger()

    tree, fallback = asyncio.run(
        make_builder(make_pages(10), logger=logger, doc_title="report").build()
    )

    assert fallback == FallbackState(used=True, accuracy=0.4)
    assert tree[0]["title"] == "report"
    assert (tree[0]["start_index"], tree[0]["end_index"]) == (1, 10)
    assert [n["title"] for n in tree[0]["nodes"]] == [f"Page {i}" for i in range(1, 11)]
    assert any(
        isinstance(r, dict) and r.get("event") == "page_level_fallback" for r in logger.records
    )


def test_the_page_level_path_never_runs_the_boundary_check(monkeypatch):
    """It would ask the model whether 'Page 7' appears on page 7 and overwrite appear_start,
    which is what gives each leaf its single-page range."""
    calls = stub_library(monkeypatch, accuracy=0.4, unfixable=True)

    tree, _ = asyncio.run(make_builder(make_pages(6)).build())

    assert calls["boundary_check"] == 0
    assert [(n["start_index"], n["end_index"]) for n in tree[0]["nodes"]] == [
        (i, i) for i in range(1, 7)
    ]


def test_fallback_keeps_absolute_page_numbers_for_a_subrange(monkeypatch):
    stub_library(monkeypatch, accuracy=0.4, unfixable=True)

    tree, fallback = asyncio.run(make_builder(make_pages(3), start_index=7).build())

    assert fallback.used is True
    assert (tree[0]["start_index"], tree[0]["end_index"]) == (7, 9)


def test_the_builder_can_be_rebuilt_after_a_failure(monkeypatch):
    """The API's retry loop reuses one builder, so build() must keep no state."""
    stub_library(monkeypatch, accuracy=0.4, unfixable=True)
    builder = make_builder(make_pages(5))

    first, first_state = asyncio.run(builder.build())
    second, second_state = asyncio.run(builder.build())

    assert first == second
    assert first_state == second_state == FallbackState(used=True, accuracy=0.4)


def test_page_level_index_is_reachable_for_the_extraction_failure_case(monkeypatch):
    """What the API calls when every attempt raised and there is no accuracy to report."""
    builder = make_builder(make_pages(4), doc_title="report")

    tree = builder.page_level_index().tree()

    assert tree[0]["title"] == "report"
    assert sum(1 for _ in walk(tree)) == 5


# ── The summary half of FallbackTreeBuilder ───────────────────────────────────

def test_summarizer_falls_back_to_the_template_when_the_rollup_is_empty(monkeypatch):
    """llm_acompletion returns '' once it has exhausted its retries."""
    tree_builder = importlib.import_module("pageindex_fallback.tree_builder")

    async def empty_llm(model, prompt, return_finish_reason=False):
        return ("", "error") if return_finish_reason else ""

    monkeypatch.setattr(tree_builder.utils, "llm_acompletion", empty_llm)

    pages = make_pages(3)
    tree = PageLevelIndex(pages, "report").tree()
    for leaf in tree[0]["nodes"]:
        leaf["text"] = "page text"
    builder = make_builder(pages, doc_title="report")
    asyncio.run(builder.summarize(tree, fallback_used=True))

    assert tree[0]["summary"] == PageLevelIndex.root_summary_for("report", 3)


# ── the library is untouched ──────────────────────────────────────────────────

def test_the_library_still_fails_hard_on_the_same_input(monkeypatch):
    """pageindex/ is unmodified: meta_processor raises where the package falls back."""
    stub_library(monkeypatch, accuracy=0.4, unfixable=True)
    opt, logger = ConfigLoader().load({}), StubLogger()

    with pytest.raises(Exception, match="Processing failed"):
        asyncio.run(pi.meta_processor(
            make_pages(10), mode="process_no_toc", start_index=1, opt=opt, logger=logger
        ))

    # and the package, given the same stubs, returns a tree instead
    tree, fallback = asyncio.run(make_builder(make_pages(10)).build())
    assert fallback.used is True
    assert sum(1 for _ in walk(tree)) == 11


def test_the_library_config_has_no_fallback_knobs():
    """Neither knob exists in pageindex/config.yaml, so passing either is an error."""
    for key, value in (("if_page_fallback", "no"), ("fallback_accuracy_threshold", 0.9)):
        with pytest.raises(ValueError, match="Unknown config keys"):
            ConfigLoader().load({key: value})
