"""Filesystem helpers for handling uploaded files.

Keeps low-level file concerns (naming, validation, saving) out of the service
and route layers.
"""

from __future__ import annotations

import re
import uuid
from pathlib import Path

from app.core.config import settings

# Matches characters that are unsafe or awkward in filenames.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize_filename(filename: str) -> str:
    """Return a filesystem-safe version of ``filename``.

    Strips directory components and replaces unsafe characters with ``_``.
    """
    # Take only the final path component to defend against traversal.
    name = Path(filename).name
    return _UNSAFE_CHARS.sub("_", name).strip("._") or "dataset.csv"


def has_allowed_extension(filename: str) -> bool:
    """Check whether ``filename`` uses an allowed extension (case-insensitive)."""
    suffix = Path(filename).suffix.lower()
    return suffix in settings.ALLOWED_EXTENSIONS


def build_unique_path(filename: str) -> Path:
    """Build a collision-free absolute path inside the uploads directory.

    A short UUID prefix guarantees uniqueness even if two files share a name.
    """
    safe_name = sanitize_filename(filename)
    unique_name = f"{uuid.uuid4().hex[:8]}_{safe_name}"
    return settings.upload_dir / unique_name


def save_bytes(content: bytes, destination: Path) -> int:
    """Write ``content`` to ``destination`` and return the number of bytes written."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(content)
    return len(content)
