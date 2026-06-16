"""One-off backfill: add per-book `genre` to existing chunk metadata.

Earlier ingestion stored only {book_id, title, author, chunk_index} on each
chunk, so genre-scoped retrieval had nothing to filter on. Genre is a static
`book_id -> genre` mapping from catalog.json, so we can patch the existing Chroma
collections and the BM25 index in place without re-embedding anything.

    uv run python -m src.utils.backfill_genre
"""

import json
import pickle
from pathlib import Path

import chromadb
import yaml

CONFIG_PATH = Path("config/retrieval.yaml")
CATALOG_PATH = Path("catalog.json")
CHROMA_DIR = Path("storage/chroma_db")
BM25_DIR = Path("storage/bm25/bm25_index.pkl")
BATCH = 500


def _book_genres() -> dict[str, str]:
    catalog = json.loads(CATALOG_PATH.read_text())
    return {book_id: meta.get("genre", "unknown") for book_id, meta in catalog.items()}


def _backfill_chroma(collection, genres: dict[str, str]) -> int:
    """Adds `genre` to every chunk's metadata, in batches, idempotently."""
    total = collection.count()
    patched = 0
    offset = 0
    while offset < total:
        batch = collection.get(limit=BATCH, offset=offset, include=["metadatas"])
        ids = batch.get("ids", [])
        metadatas = batch.get("metadatas", [])
        if not ids:
            break

        update_ids, update_metas = [], []
        for chunk_id, meta in zip(ids, metadatas):
            meta = dict(meta or {})
            desired = genres.get(meta.get("book_id"), "unknown")
            if meta.get("genre") != desired:
                meta["genre"] = desired
                update_ids.append(chunk_id)
                update_metas.append(meta)

        if update_ids:
            collection.update(ids=update_ids, metadatas=update_metas)
            patched += len(update_ids)

        offset += len(ids)
    return patched


def _backfill_bm25(genres: dict[str, str]) -> int:
    if not BM25_DIR.exists():
        return 0
    with open(BM25_DIR, "rb") as f:
        data = pickle.load(f)

    patched = 0
    for meta in data.get("metadata", []):
        desired = genres.get(meta.get("book_id"), "unknown")
        if meta.get("genre") != desired:
            meta["genre"] = desired
            patched += 1

    if patched:
        with open(BM25_DIR, "wb") as f:
            pickle.dump(data, f)
    return patched


def main() -> None:
    genres = _book_genres()
    config = yaml.safe_load(CONFIG_PATH.read_text())
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing = {c.name for c in client.list_collections()}
    for key in ("local_collection", "openai_collection"):
        name = config["vector_stores"][key]
        if name not in existing:
            print(f"Skipping {name} (not present).")
            continue
        # Genre metadata is patched without touching embeddings, so we don't need
        # the embedding function attached here.
        collection = client.get_collection(name)
        patched = _backfill_chroma(collection, genres)
        print(f"Chroma '{name}': patched {patched} / {collection.count()} chunks.")

    bm25_patched = _backfill_bm25(genres)
    print(f"BM25 index: patched {bm25_patched} chunk metadata entries.")


if __name__ == "__main__":
    main()
