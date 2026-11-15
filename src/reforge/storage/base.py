"""Storage abstraction so originals can live local now and on R2 later (config flip)."""
from __future__ import annotations
from abc import ABC, abstractmethod

class StorageProvider(ABC):
    """Key = logical path of a clip (e.g. 'library/C2046.MP4')."""

    @abstractmethod
    def ensure_local(self, key: str) -> str:
        """Return an absolute local path to the asset, downloading if remote."""

    @abstractmethod
    def upload(self, local_path: str, key: str) -> None: ...

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]: ...

def get_storage() -> "StorageProvider":
    """Factory: returns provider based on settings.storage_backend."""
    from ..config import settings
    if settings.storage_backend == "r2":
        from .r2 import R2Storage
        return R2Storage()
    from .local import LocalStorage
    return LocalStorage()
