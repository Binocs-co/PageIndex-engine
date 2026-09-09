import logging

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from api.dependencies import get_s3_bucket, get_s3_session
from api.services.pageindex_service import process_markdown

logger = logging.getLogger(__name__)

pageindex_router = APIRouter()


_CONFIG_YAML_NOTE = "Defaults to the value in pageindex/config.yaml when not provided."


class MarkdownPageIndexRequest(BaseModel):
    input_s3_key: str = Field(..., description="S3 key of the markdown file to index.")
    output_s3_key: str = Field(..., description="S3 key where the output tree JSON will be written.")
    tokens_per_page: int = Field(
        default=2000,
        ge=500,
        le=10000,
        description="Target token budget per virtual page. Controls section granularity.",
    )

    # Pipeline options — all optional; unset fields fall back to pageindex/config.yaml defaults
    model: str | None = Field(default=None, description=f"LLM model name for all pipeline stages. {_CONFIG_YAML_NOTE}")
    if_add_node_id: str | None = Field(default=None, description=f'"yes" or "no". {_CONFIG_YAML_NOTE}')
    if_add_node_summary: str | None = Field(default=None, description=f'"yes" or "no". {_CONFIG_YAML_NOTE}')
    if_add_node_text: str | None = Field(default=None, description=f'"yes" or "no". {_CONFIG_YAML_NOTE}')
    if_add_doc_description: str | None = Field(default=None, description=f'"yes" or "no". {_CONFIG_YAML_NOTE}')
    max_tree_attempts: int | None = Field(
        default=None,
        ge=1,
        le=3,
        description=(
            "How many times to attempt tree extraction before the request fails. Only "
            "*raised* failures are retried (a truncated or unparseable LLM completion); a "
            "low-accuracy result is not, since the extraction is deterministic at "
            "temperature 0. Defaults to PAGEINDEX_MAX_TREE_ATTEMPTS, else 2."
        ),
    )

    extra_config: dict | None = Field(
        default=None,
        description="Escape hatch for any other config.yaml key not exposed above.",
    )
    # TODO: add `content: str | None` to accept raw markdown inline, skipping the S3 read


class MarkdownPageIndexResponse(BaseModel):
    output_s3_key: str
    doc_description: str
    structure: list
    attempts: int = Field(
        default=1,
        description=(
            "How many tree extraction attempts were made. Greater than 1 means an attempt "
            "raised and was retried; see the server logs for the failure."
        ),
    )


@pageindex_router.post("/markdown", response_model=MarkdownPageIndexResponse)
async def index_markdown(payload: MarkdownPageIndexRequest) -> MarkdownPageIndexResponse:
    """Index a markdown document stored on S3 using the full PDF pipeline (process_no_toc path).

    Reads the markdown from S3, splits it into virtual pages, runs tree generation
    + verification + retry logic, then writes the resulting tree JSON back to S3.
    """
    try:
        result = await process_markdown(payload, get_s3_session(), get_s3_bucket())
    except Exception as e:
        logger.error(f"PageIndex pipeline failed for '{payload.input_s3_key}': {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
    return result
