"""Local-disk storage provider (implemented)."""
from __future__ import annotations
import os
import glob
from .base import StorageProvider

class LocalStorage(StorageProvider):
    def __init__(self, root: str | None = None):
        self.root = os.path.abspath(root or ".")

    def _abs(self, key: str) -> str:
        return key if os.path.isabs(key) else os.path.join(self.root, key)

    def ensure_local(self, key: str) -> str:
        p = self._abs(key)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return p

    def upload(self, local_path: str, key: str) -> None:
        # already local; no-op (copy if root differs)
        dst = self._abs(key)
        if os.path.abspath(local_path) != dst:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            import shutil
            shutil.copy2(local_path, dst)

    def exists(self, key: str) -> bool:
        return os.path.exists(self._abs(key))

    def list(self, prefix: str = "") -> list[str]:
        base = self._abs(prefix)
        return sorted(glob.glob(os.path.join(base, "**", "*"), recursive=True))
