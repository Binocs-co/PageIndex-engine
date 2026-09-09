import asyncio
import json
import logging
import os
import re

_PAGE_TAG_RE = re.compile(r"(<page_number=\d+>.*?</page_number=\d+>)", re.DOTALL)

from pageindex.page_index import process_large_node_recursively
from pageindex.utils import (
    ConfigLoader,
    add_node_text,
    agenerate_doc_description,
    count_tokens,
    create_clean_structure_for_description,
    format_structure,
    remove_structure_text,
    write_node_id,
)
from pageindex_fallback import FallbackState, FallbackTreeBuilder

logger = logging.getLogger(__name__)
# ── S3 helpers ────────────────────────────────────────────────────────────────

async def read_markdown_from_s3(key: str, s3_session, bucket: str) -> str:
    async with s3_session.client("s3") as s3:
        resp = await s3.get_object(Bucket=bucket, Key=key)
        return (await resp["Body"].read()).decode("utf-8")


async def upload_tree_to_s3(key: str, data: dict, s3_session, bucket: str) -> None:
    async with s3_session.client("s3") as s3:
        await s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2, ensure_ascii=False),
            ContentType="application/json",
        )


# ── Markdown → page_list ──────────────────────────────────────────────────────

def markdown_to_page_list(content: str, tokens_per_page: int, model: str) -> list[tuple[str, int]]:
    """Split markdown text into virtual pages for the pipeline.

    When the content contains <page_number=N>...</page_number=N> tags (produced
    by the OCR-to-markdown converter), each tag block becomes exactly one virtual
    page. This keeps physical_index values accurate to real document page numbers.

    Falls back to token-budget chunking for plain markdown without page tags.
    tokens_per_page is unused in the tagged path.
    """
    tag_matches = _PAGE_TAG_RE.findall(content)
    if tag_matches:
        return [(page, count_tokens(page, model)) for page in tag_matches if page.strip()]

    lines = content.split("\n")
    pages: list[tuple[str, int]] = []
    current_lines: list[str] = []
    current_tokens = 0

    for line in lines:
        line_tokens = count_tokens(line, model) if line.strip() else 1
        if current_tokens + line_tokens > tokens_per_page and current_lines:
            chunk = "\n".join(current_lines)
            pages.append((chunk, count_tokens(chunk, model)))
            current_lines = []
            current_tokens = 0
        current_lines.append(line)
        current_tokens += line_tokens

    if current_lines:
        chunk = "\n".join(current_lines)
        pages.append((chunk, count_tokens(chunk, model)))

    return pages


# ── Tree building: attempt, retry, fallback ───────────────────────────────────

class MarkdownIndexer:
    """Indexes one markdown document: config, retry, tree, summaries.

        indexer = MarkdownIndexer(payload)
        tree, description, fallback, attempts = await indexer.build_tree(page_list)

    Two independent recoveries, and the difference matters:

    * A TOC that fails verification is handled by FallbackTreeBuilder: it repairs what it
      can and, if any item is still misplaced, returns the page-level tree and reports it
      through the FallbackState it hands back. Nothing is retried — at temperature 0 a
      re-run reproduces the same extraction.
    * A *raised* extraction failure (a truncated completion, or an unparseable one, which
      makes extract_json return {} and the following extend() raise AttributeError) is
      retried up to `max_attempts` times. If every attempt raises, the page-level index is
      built here instead — unconditionally, so a failure that would otherwise reach the
      router and become a 500 returns a usable tree.

    Only extraction and tree assembly are retried. Node text, summaries and the
    description run once, on whichever tree survives, so a retry never pays for summaries
    twice.
    """

    DEFAULT_MAX_TREE_ATTEMPTS = 2
    MAX_TREE_ATTEMPTS_CEILING = 3

    # Deploy-time defaults for the library config, overridden per request.
    ENV_MAP = {
        "model": "PAGEINDEX_MODEL",
        "if_add_node_id": "PAGEINDEX_IF_ADD_NODE_ID",
        "if_add_node_summary": "PAGEINDEX_IF_ADD_NODE_SUMMARY",
        "if_add_node_text": "PAGEINDEX_IF_ADD_NODE_TEXT",
        "if_add_doc_description": "PAGEINDEX_IF_ADD_DOC_DESCRIPTION",
    }

    def __init__(self, payload):
        self.payload = payload
        self.opt = ConfigLoader().load(self.config_overrides())
        self.max_attempts = self.resolve_max_tree_attempts()
        # No doc_name on this path — name the root after the input file.
        self.doc_title = (
            os.path.splitext(os.path.basename(payload.input_s3_key))[0] or "Document"
        )

    # ── configuration ────────────────────────────────────────────────────────

    def config_overrides(self):
        """Library config overrides with three-tier priority.

        Priority (highest → lowest):
          1. Request fields — explicit per-call values
          2. PAGEINDEX_* env vars — deploy-time defaults
          3. pageindex/config.yaml — library defaults (handled by ConfigLoader)
        """
        payload = self.payload
        overrides = {
            key: os.environ[env] for key, env in self.ENV_MAP.items() if env in os.environ
        }
        request_fields = {
            "model": payload.model,
            "if_add_node_id": payload.if_add_node_id,
            "if_add_node_summary": payload.if_add_node_summary,
            "if_add_node_text": payload.if_add_node_text,
            "if_add_doc_description": payload.if_add_doc_description,
            **(payload.extra_config or {}),
        }
        overrides.update({k: v for k, v in request_fields.items() if v is not None})
        return overrides

    def resolve_max_tree_attempts(self):
        """Attempt budget: request field > env var > default.

        Deliberately kept out of `opt`: ConfigLoader._validate_keys() rejects any key
        absent from pageindex/config.yaml, so routing this through config_overrides() — or
        through `extra_config` — would fail the request outright.
        """
        attempts = getattr(self.payload, "max_tree_attempts", None)
        if attempts is None:
            raw = os.environ.get("PAGEINDEX_MAX_TREE_ATTEMPTS")
            if raw:
                try:
                    attempts = int(raw)
                except ValueError:
                    logger.warning(
                        "ignoring non-integer PAGEINDEX_MAX_TREE_ATTEMPTS=%r", raw
                    )
        if attempts is None:
            attempts = self.DEFAULT_MAX_TREE_ATTEMPTS
        return max(1, min(int(attempts), self.MAX_TREE_ATTEMPTS_CEILING))

    # ── tree building ────────────────────────────────────────────────────────

    def page_list(self, content: str) -> list[tuple[str, int]]:
        return markdown_to_page_list(content, self.payload.tokens_per_page, self.opt.model)

    async def build_tree(
        self, page_list: list[tuple[str, int]]
    ) -> tuple[list, str, FallbackState, int]:
        """Returns (tree, doc_description, fallback, attempts)."""
        logger.info({"total_page_number": len(page_list)})
        logger.info({"total_token": sum(t for _, t in page_list)})

        # One builder for the whole document: it both extracts the tree and summarizes
        # whichever tree came out, so it is created here rather than inside the retry loop.
        builder = FallbackTreeBuilder(page_list, self.opt, logger, doc_title=self.doc_title)
        tree, fallback, attempts = await self._extract_with_retries(builder)

        if not fallback.used:
            await self._split_large_nodes(tree, page_list)

        if self.opt.if_add_node_id == "yes":
            write_node_id(tree)
        if self.opt.if_add_node_text == "yes" or self.opt.if_add_node_summary == "yes":
            add_node_text(tree, page_list)

        description = await self._add_summaries(builder, tree, fallback)

        tree = format_structure(
            tree,
            order=["title", "node_id", "start_index", "end_index", "summary", "text", "nodes"],
        )
        return tree, description, fallback, attempts

    async def _extract_with_retries(self, builder):
        """Retry a *raised* extraction, then fall back. Returns (tree, fallback, attempts).

        max_attempts is clamped to at least 1, so the loop always returns — either a tree
        an attempt produced, or the page-level index.
        """
        for attempt in range(1, self.max_attempts + 1):
            try:
                tree, fallback = await builder.build()
                if attempt > 1:
                    logger.info(
                        "tree extraction succeeded on attempt %d/%d",
                        attempt, self.max_attempts,
                    )
                return tree, fallback, attempt
            except Exception as exc:
                # No backoff: llm_acompletion already retried the provider 10x with a 1s
                # sleep, so what reaches here is a post-transport failure a pause cannot
                # help.
                logger.warning(
                    "tree extraction attempt %d/%d failed: %s",
                    attempt, self.max_attempts, exc, exc_info=True,
                )
                if attempt == self.max_attempts:
                    # The page-level index is unconditional: an exhausted retry budget
                    # always yields a tree, never a 500. No verification ever ran, so
                    # there is no accuracy to report — accuracy=None is what tells this
                    # apart from a failed verification.
                    logger.warning(
                        "all %d tree extraction attempts failed (last error: %s) — "
                        "building the page-level index for %d pages instead",
                        self.max_attempts, exc, builder.page_count,
                    )
                    tree = builder.page_level_index().tree()
                    return tree, FallbackState(used=True, accuracy=None), self.max_attempts

    async def _split_large_nodes(self, tree, page_list):
        """Kept out of the retry scope on purpose: this re-enters the library's
        meta_processor, and re-running the whole document extraction to fix one unsplit
        subtree is not worth it. return_exceptions keeps one bad node from cancelling its
        siblings."""
        results = await asyncio.gather(
            *[
                process_large_node_recursively(node, page_list, self.opt, logger=logger)
                for node in tree
            ],
            return_exceptions=True,
        )
        for failure in (r for r in results if isinstance(r, Exception)):
            logger.warning(
                "large-node splitting failed for one root node, keeping it unsplit: %s",
                failure, exc_info=failure,
            )

    async def _add_summaries(self, builder, tree, fallback) -> str:
        """Summaries and, nested inside them, the document description.

        The nesting matches the library's page_index_builder(): a description is only
        generated when summaries are also enabled.
        """
        if self.opt.if_add_node_summary != "yes":
            return ""

        # A page-level root spans the whole document, so it is summarized from its page
        # summaries rather than from its own text; the builder picks the strategy.
        await builder.summarize(tree, fallback_used=fallback.used)
        if self.opt.if_add_node_text == "no":
            remove_structure_text(tree)

        if self.opt.if_add_doc_description != "yes":
            return ""
        clean = create_clean_structure_for_description(tree)
        return await agenerate_doc_description(clean, model=self.opt.model)


# ── Public orchestrator ───────────────────────────────────────────────────────

async def process_markdown(payload, s3_session, bucket: str) -> dict:
    """Read the markdown from S3, index it, write the tree back."""
    indexer = MarkdownIndexer(payload)
    content = await read_markdown_from_s3(payload.input_s3_key, s3_session, bucket)

    tree, description, fallback, attempts = await indexer.build_tree(
        indexer.page_list(content)
    )

    output = {
        "doc_description": description,
        "structure": tree,
        "fallback_used": fallback.used,
        # None when the fallback came from a failed extraction rather than a failed
        # verification.
        "fallback_accuracy": fallback.accuracy,
        # Persisted too, so a stored tree can be triaged after the fact.
        "attempts": attempts,
    }
    await upload_tree_to_s3(payload.output_s3_key, output, s3_session, bucket)

    return {**output, "output_s3_key": payload.output_s3_key}
