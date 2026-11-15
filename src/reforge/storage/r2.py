"""Cloudflare R2 storage provider (STUB). Implement in Phase 1+ with boto3 S3 client.

TODO(builder):
  - boto3.client('s3', endpoint_url=f'https://{account_id}.r2.cloudflarestorage.com', ...)
  - ensure_local(): download to a local cache dir (settings.local_data_dir/cache) and return path
  - upload(): put_object to settings.r2_bucket
  - cache downloads so repeated reads don't re-egress (egress is free on R2, but disk cache is faster)
"""
from __future__ import annotations
from .base import StorageProvider

class R2Storage(StorageProvider):
    def __init__(self):
        raise NotImplementedError("R2Storage not implemented yet; see TODO in this file.")

    def ensure_local(self, key: str) -> str: ...
    def upload(self, local_path: str, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def list(self, prefix: str = "") -> list[str]: ...
