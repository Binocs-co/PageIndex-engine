"""A fallback-aware stand-in for `meta_processor`'s `process_no_toc` path.

`pageindex.page_index.meta_processor()` is a thin orchestrator: it runs one extraction
mode, validates the physical indices, verifies the result, and repairs what it can. What
it does when that is not enough is the part we need to change — it downgrades to a weaker
extraction mode and finally raises `Processing failed`, so the caller gets nothing.

`FallbackTreeBuilder` below calls the same library building blocks in the same order and
makes the same accept decisions, but where the library gives up it substitutes a
`PageLevelIndex` — and it summarizes whichever of the two trees came out. The library is
used unmodified: nothing in `pageindex/` knows this package exists.

Like the markdown route it serves, this enters `process_no_toc` directly and never runs
`check_toc()`: a markdown document's structure is already in the text, so there is no
contents page to scan for.

The library functions are called through their module (`page_index.verify_toc(...)`,
`utils.llm_acompletion(...)`) rather than imported by name, so a test that patches
`pageindex.page_index.<fn>` or `pageindex.utils.<fn>` patches what this module calls too.
A `from ... import` would bind the real function and quietly reach the network.

`pageindex.page_index` has to be reached with import_module: `pageindex/__init__.py` does
`from .page_index import *`, and page_index.py defines a *function* named page_index, so
the package attribute `pageindex.page_index` is that function — `from pageindex import
page_index` and `import pageindex.page_index as pi` both hand back the function.
"""
import asyncio
from importlib import import_module

from pageindex.utils import post_processing

from pageindex_fallback.page_level import PageLevelIndex
from pageindex_fallback.schemas import FallbackState, TocExtraction

page_index = import_module("pageindex.page_index")
utils = import_module("pageindex.utils")


class FallbackTreeBuilder:
    """Builds a tree for one document and summarizes it, falling back when needed.

        builder = FallbackTreeBuilder(page_list, opt, logger, doc_title="report")
        tree, fallback = await builder.build()
        await builder.summarize(tree, fallback_used=fallback.used)

    Building and summarizing are one class because they answer to the same fact. A
    page-level root spans every page, so it cannot be summarized from its own text the way
    a section node can — which means the fallback decision *is* the choice of summary
    strategy. Both halves also need the same four things: the page list, the model, the
    document title and the page count, so splitting them only meant passing those twice.

    Neither `build()` nor `summarize()` holds state on the instance, so the same builder
    can be re-run — which is what the API's retry loop does after a raised extraction
    failure.
    """

    MAX_REPAIR_ATTEMPTS = 3

    def __init__(self, page_list, opt, logger, doc_title="Document", start_index=1):
        self.page_list = page_list
        self.opt = opt
        self.logger = logger
        self.doc_title = doc_title
        self.start_index = start_index

    @property
    def page_count(self):
        return len(self.page_list)

    @property
    def model(self):
        return self.opt.model

    async def build(self):
        """Returns (tree, FallbackState).

        The fallback is unconditional: a page-level index is always preferable to a
        structure known to be wrong, so there is nothing to configure and no weaker
        extraction mode to try.
        """
        extraction = await self.extract_verified_toc()
        if not extraction.usable:
            return self._page_level_tree(extraction)
        return await self._assemble(extraction.flat_toc), FallbackState()

    async def extract_verified_toc(self):
        """Extract a TOC and verify it, returning a TocExtraction.

        There is no accuracy threshold. A score is not evidence that a TOC is usable, only
        a clean verification is, so every imperfect TOC goes through the repair pass and is
        accepted only if nothing is left misplaced.
        """
        flat_toc = await page_index.process_no_toc(
            self.page_list, start_index=self.start_index, model=self.opt.model,
            logger=self.logger,
        )
        flat_toc = [item for item in flat_toc if item.get('physical_index') is not None]
        flat_toc = page_index.validate_and_truncate_physical_indices(
            flat_toc, self.page_count, start_index=self.start_index, logger=self.logger
        )

        accuracy, incorrect_results = await page_index.verify_toc(
            self.page_list, flat_toc, start_index=self.start_index, model=self.opt.model
        )
        self.logger.info({
            'mode': 'process_no_toc',
            'accuracy': accuracy,
            'incorrect_results': incorrect_results,
        })

        if accuracy == 1.0 and not incorrect_results:
            return TocExtraction(flat_toc, accuracy, [])

        if incorrect_results:
            flat_toc, incorrect_results = await page_index.fix_incorrect_toc_with_retries(
                flat_toc, self.page_list, incorrect_results, start_index=self.start_index,
                max_attempts=self.MAX_REPAIR_ATTEMPTS, model=self.opt.model,
                logger=self.logger,
            )
            if not incorrect_results:
                return TocExtraction(flat_toc, accuracy, [])

        return TocExtraction(None, accuracy, incorrect_results)

    async def _assemble(self, flat_toc):
        """Turn a verified flat TOC into a tree, the way the library's tree_parser does."""
        flat_toc = page_index.add_preface_if_needed(flat_toc)
        flat_toc = await page_index.check_title_appearance_in_start_concurrent(
            flat_toc, self.page_list, model=self.opt.model, logger=self.logger
        )
        valid_items = [item for item in flat_toc if item.get('physical_index') is not None]
        return post_processing(valid_items, self.start_index + self.page_count - 1)

    def _page_level_tree(self, extraction):
        self.logger.info({
            'event': 'page_level_fallback',
            'accuracy': extraction.accuracy,
            'unfixed_items': len(extraction.unfixed_items),
            'page_count': self.page_count,
            'start_index': self.start_index,
        })
        self.logger.warning(
            "TOC verification failed (accuracy %.2f%%, %d unfixed item(s)) — indexing "
            "%d pages as one node each",
            extraction.accuracy * 100, len(extraction.unfixed_items), self.page_count,
        )
        tree = self.page_level_index().tree()
        return tree, FallbackState(used=True, accuracy=extraction.accuracy)

    def page_level_index(self):
        """The page-level index for this document, whatever the reason for needing it."""
        return PageLevelIndex(
            self.page_list, doc_title=self.doc_title, start_index=self.start_index
        )

    # ── summaries ────────────────────────────────────────────────────────────

    async def summarize(self, tree, fallback_used=False):
        """Add `summary` to every node, choosing the strategy from the fallback flag."""
        if fallback_used:
            return await self.summarize_page_level(tree)
        return await utils.generate_summaries_for_structure(tree, model=self.model)

    async def summarize_page_level(self, tree):
        """Leaves from their page text, root rolled up from those leaf summaries.

        A page-level root is why this path exists: it spans every page, so
        pageindex.utils.generate_node_summary() would put the whole document in one
        prompt. It is summarized from its children's summaries instead.
        """
        if not tree:
            return tree

        root = tree[0]
        leaves = root.get('nodes') or []
        if leaves:
            summaries = await asyncio.gather(
                *[utils.generate_node_summary(leaf, model=self.model) for leaf in leaves]
            )
            for leaf, summary in zip(leaves, summaries):
                leaf['summary'] = summary

        rolled_up = await self.rollup([leaf.get('summary') for leaf in leaves])
        # llm_acompletion returns '' once it has exhausted its retries, and a blank
        # summary is worse than a deterministic one.
        root['summary'] = rolled_up or self.page_level_index().root_summary()
        return tree

    async def rollup(self, child_summaries):
        """Summarize a parent from its children's summaries instead of its own text.

        Keeps the prompt proportional to the number of children rather than to the
        document's length.
        """
        joined = "\n".join(f"- {s}" for s in child_summaries if s)
        if not joined:
            return ""
        prompt = f"""You are given the section-by-section descriptions of a document, your task is to generate a single description of what the whole document covers.

    Section descriptions:
    {joined}

    Directly return the description, do not include any other text.
    """
        return await utils.llm_acompletion(self.model, prompt)
