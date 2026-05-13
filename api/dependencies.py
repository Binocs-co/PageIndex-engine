import os
from functools import lru_cache

import aioboto3

# TODO: replace with a dedicated async S3 client that manages its own connection pool
# and lifecycle (e.g. FastAPI lifespan event) instead of creating a new client per call.
@lru_cache(maxsize=1)
def get_s3_session() -> aioboto3.Session:
    return aioboto3.Session()


def get_s3_bucket() -> str:
    bucket = os.environ.get("PAGEINDEX_S3_BUCKET")
    if not bucket:
        raise RuntimeError("PAGEINDEX_S3_BUCKET environment variable is not set")
    return bucket
