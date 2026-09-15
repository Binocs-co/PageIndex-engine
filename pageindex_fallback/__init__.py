"""Page-level fallback for PageIndex.

Everything here sits on top of an unmodified `pageindex` package: when a document's table
of contents cannot be extracted and verified, the library raises and gives the caller
nothing, so this package re-orchestrates the library's own building blocks and substitutes
a deterministic one-node-per-page index instead.

    builder = FallbackTreeBuilder(page_list, opt, logger, doc_title="report")
    tree, fallback = await builder.build()
    if opt.if_add_node_summary == "yes":
        await builder.summarize(tree, fallback_used=fallback.used)

One class per module — `tree_builder.py` builds and summarizes, `page_level.py` holds the
structure it falls back to, `schemas.py` the values they pass around. This file is only the
public surface.
"""
from pageindex_fallback.page_level import PageLevelIndex
from pageindex_fallback.schemas import FallbackState, TocExtraction
from pageindex_fallback.tree_builder import FallbackTreeBuilder

__all__ = [
    "FallbackState",
    "FallbackTreeBuilder",
    "PageLevelIndex",
    "TocExtraction",
]
