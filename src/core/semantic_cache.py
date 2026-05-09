import hashlib
import json
import time
from pathlib import Path
from typing import Any

import chromadb
import yaml
from dotenv import load_dotenv

load_dotenv()

CONFIG_PATH = Path("config/retrieval.yaml")
CHROMA_DIR = Path("storage/chroma_db")


class SemanticAnswerCache:
    def __init__(self):
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.safe_load(f)

        self.inference_mode = self.config["system"].get("inference_mode", "local")
        self.threshold = float(self.config.get("cache", {}).get("similarity_threshold", 0.88))
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.embedding_function = self._build_embedding_function()
        self.collection = self.client.get_or_create_collection(
            name=f"citementor_answer_cache_{self.inference_mode}",
            embedding_function=self.embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def _build_embedding_function(self):
        from chromadb.utils import embedding_functions

        if self.inference_mode == "openai":
            import os

            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
            if not api_key:
                raise RuntimeError("OPENAI_API_KEY is required for the OpenAI semantic cache.")
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=self.config["openai"]["embedding_model"],
            )

        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config["system"]["embedding_model"],
            trust_remote_code=True,
        )

    def lookup(self, query: str) -> dict[str, Any] | None:
        if self.collection.count() == 0:
            return None

        result = self.collection.query(
            query_texts=[query],
            n_results=1,
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        if not ids:
            return None

        distance = float(result.get("distances", [[1.0]])[0][0])
        similarity = max(0.0, min(1.0, 1.0 - distance))
        if similarity < self.threshold:
            return None

        metadata = result.get("metadatas", [[{}]])[0][0] or {}
        try:
            sources = json.loads(metadata.get("sources_json", "[]"))
        except json.JSONDecodeError:
            sources = []

        return {
            "answer": metadata.get("answer", ""),
            "sources": sources,
            "similarity": similarity,
            "matched_query": result.get("documents", [[""]])[0][0],
        }

    def store(self, query: str, answer: str, sources: list[dict[str, Any]]) -> None:
        cache_id = hashlib.sha256(query.strip().lower().encode("utf-8")).hexdigest()
        source_payload = [
            {
                "id": source.get("id", ""),
                "text": source.get("text", ""),
                "book_id": source.get("book_id", ""),
                "author": source.get("author", ""),
                "cross_score": source.get("cross_score", 0.0),
            }
            for source in sources
        ]
        self.collection.upsert(
            ids=[cache_id],
            documents=[query],
            metadatas=[{
                "answer": answer,
                "sources_json": json.dumps(source_payload),
                "created_at": time.time(),
            }],
        )
