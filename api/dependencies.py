import os
from functools import lru_cache

import boto3


@lru_cache(maxsize=1)
def get_s3_client():
    return boto3.client("s3")


def get_s3_bucket() -> str:
    bucket = os.environ.get("PAGEINDEX_S3_BUCKET")
    if not bucket:
        raise RuntimeError("PAGEINDEX_S3_BUCKET environment variable is not set")
    return bucket
