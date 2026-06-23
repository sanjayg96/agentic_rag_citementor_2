"""Resolve the on-disk location of the Chroma vector store at runtime.

The corpus Chroma DB is bundled into the container image as a read-only
artifact under ``storage/chroma_db``. ChromaDB's ``PersistentClient`` opens the
underlying SQLite database read-write (it takes locks and may create WAL/journal
files), but on AWS Lambda the image filesystem is read-only — only ``/tmp`` is
writable. So on Lambda we copy the bundled DB into ``/tmp`` once per cold start
and point every client at that writable copy.

Both the retriever and the semantic answer cache must resolve to the *same*
path: the cache collection lives in the same SQLite database as the corpus
collection, so they have to share one copy. The resolved path is therefore
memoised at module level — the copy happens exactly once per process and every
caller gets the identical directory.

Offline build scripts (``src/utils/ingestion.py``, ``backfill_genre.py``) keep
writing straight to ``storage/chroma_db`` and do not use this resolver.
"""

import os
import shutil
from pathlib import Path

# Source bundle (read-only on Lambda). Env-overridable so tests / alternate
# layouts can point elsewhere without touching code.
_DEFAULT_SOURCE = "storage/chroma_db"

# Memoised resolved path: the copy runs once per process and all callers share it.
_resolved_path: str | None = None


def _should_relocate() -> bool:
    """True when the bundled DB must be copied to a writable location.

    ``AWS_LAMBDA_FUNCTION_NAME`` is set automatically in every Lambda runtime, so
    the relocation is zero-config in production. ``CHROMA_COPY_TO_TMP`` forces the
    same code path in a local container so Phase 2 parity can be verified offline.
    """
    on_lambda = bool(os.getenv("AWS_LAMBDA_FUNCTION_NAME"))
    forced = os.getenv("CHROMA_COPY_TO_TMP", "").lower() in ("1", "true", "yes")
    return on_lambda or forced


def resolve_chroma_path() -> str:
    """Return the Chroma directory to open, relocating to /tmp on Lambda.

    Locally this is a no-op and returns the bundled (or ``CHROMA_DIR``) path. On
    Lambda it copies that bundle to ``/tmp`` once and returns the copy. Memoised
    so the (potentially large) copy runs exactly once and every client — corpus
    retriever and answer cache alike — shares the same writable database.
    """
    global _resolved_path
    if _resolved_path is not None:
        return _resolved_path

    source = Path(os.getenv("CHROMA_DIR", _DEFAULT_SOURCE))
    if not _should_relocate():
        _resolved_path = str(source)
        return _resolved_path

    target = Path(os.getenv("CHROMA_TMP_DIR", "/tmp/chroma_db"))
    # Guard on existence to stay idempotent across warm invocations that reuse
    # the same /tmp (only the cold start pays the copy cost).
    if not target.exists():
        shutil.copytree(source, target)
    _resolved_path = str(target)
    return _resolved_path
