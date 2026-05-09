import json
import re
import time
import yaml
import pickle
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from dotenv import load_dotenv

load_dotenv()

# Paths
CONFIG_PATH = Path("config/retrieval.yaml")
CHROMA_DIR = Path("storage/chroma_db")
BM25_DIR = Path("storage/bm25/bm25_index.pkl")

class HybridRetriever:
    def __init__(self):
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.retrieval_cfg = self.config["retrieval"]
        self.inference_mode = self.config["system"].get("inference_mode", "local")
        
        # 1. Connect to the collection that matches the active embedding space.
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        collection_name = self.config["vector_stores"]["local_collection"]
        if self.inference_mode == "openai":
            collection_name = self.config["vector_stores"]["openai_collection"]

        collection_kwargs = {"name": collection_name}
        if self.inference_mode == "local":
            from chromadb.utils import embedding_functions

            self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config["system"]["embedding_model"],
                trust_remote_code=True
            )
            collection_kwargs["embedding_function"] = self.emb_fn
        elif self.inference_mode == "openai":
            from chromadb.utils import embedding_functions

            api_key = self._get_openai_api_key()
            self.emb_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name=self.config["openai"]["embedding_model"]
            )
            collection_kwargs["embedding_function"] = self.emb_fn

        self.collection = self.chroma_client.get_or_create_collection(
            **collection_kwargs
        )
        
        # 2. Load BM25 Index
        with open(BM25_DIR, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25_model = bm25_data["model"]
            self.bm25_metadata = bm25_data["metadata"]
            
        # 3. Local cross-encoder is loaded lazily and only in local mode.
        self._cross_encoder = None
        self._openai_reranker = None
        self.last_timings = {}

    @property
    def cross_encoder(self):
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder

            print(f"Loading Cross-Encoder: {self.config['system']['reranker_model']}...")
            self._cross_encoder = CrossEncoder(self.config["system"]["reranker_model"])
        return self._cross_encoder

    @property
    def openai_reranker(self):
        if self._openai_reranker is None:
            from langchain_openai import ChatOpenAI

            self._openai_reranker = ChatOpenAI(
                model=self.config["openai"]["reranker_model"],
                temperature=0
            )
        return self._openai_reranker

    def _get_openai_api_key(self) -> str:
        import os

        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is required when inference_mode is set to openai.")
        return api_key

    def _reciprocal_rank_fusion(self, vector_results: List[Dict], bm25_results: List[Dict]) -> List[Dict]:
        """Fuses ranked lists using the RRF formula."""
        rrf_k = self.retrieval_cfg["rrf_k"]
        scores = {}
        chunk_map = {}

        # Process Vector Results
        for rank, item in enumerate(vector_results):
            chunk_id = item["id"]
            chunk_map[chunk_id] = item
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Process BM25 Results
        for rank, item in enumerate(bm25_results):
            chunk_id = item["id"]
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = item
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank + 1)

        # Sort by cumulative RRF score
        fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [chunk_map[chunk_id] for chunk_id, _ in fused]

    def retrieve(self, queries: List[str]) -> List[Dict[str, Any]]:
        """Executes Hybrid Search across multiple expanded queries."""
        started_total = time.perf_counter()
        semantic_ms = 0.0
        lexical_ms = 0.0
        fusion_ms = 0.0
        reranker_ms = 0.0
        all_vector_results = []
        all_bm25_results = []
        
        seen_vector_ids = set()
        seen_bm25_ids = set()

        for query in queries:
            # --- A. Semantic Search (Chroma) ---
            started = time.perf_counter()
            vector_res = self.collection.query(
                query_texts=[query],
                n_results=self.retrieval_cfg["semantic_top_k"]
            )
            semantic_ms += (time.perf_counter() - started) * 1000
            
            for idx, chunk_id in enumerate(vector_res["ids"][0]):
                if chunk_id not in seen_vector_ids:
                    all_vector_results.append({
                        "id": chunk_id,
                        "text": vector_res["documents"][0][idx],
                        "book_id": vector_res["metadatas"][0][idx]["book_id"],
                        "author": vector_res["metadatas"][0][idx]["author"]
                    })
                    seen_vector_ids.add(chunk_id)

            # --- B. Lexical Search (BM25) ---
            started = time.perf_counter()
            tokenized_query = query.split()
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            top_n_indices = bm25_scores.argsort()[::-1][:self.retrieval_cfg["lexical_top_k"]]

            candidate_metadata = []
            missing_ids = []
            for idx in top_n_indices:
                meta = self.bm25_metadata[idx]
                chunk_id = f"{meta['book_id']}_chunk_{meta['chunk_index']}"

                if chunk_id not in seen_bm25_ids:
                    candidate_metadata.append((chunk_id, meta))
                    missing_ids.append(chunk_id)

            # Fetch BM25 documents from Chroma in one batch instead of one DB call per hit.
            if missing_ids:
                fetch_res = self.collection.get(ids=missing_ids)
                documents_by_id = {
                    chunk_id: document
                    for chunk_id, document in zip(fetch_res.get("ids", []), fetch_res.get("documents", []))
                }

                for chunk_id, meta in candidate_metadata:
                    document = documents_by_id.get(chunk_id)
                    if document:
                        all_bm25_results.append({
                            "id": chunk_id,
                            "text": document,
                            "book_id": meta["book_id"],
                            "author": meta["author"]
                        })
                        seen_bm25_ids.add(chunk_id)
            lexical_ms += (time.perf_counter() - started) * 1000

        # --- C. Reciprocal Rank Fusion ---
        started = time.perf_counter()
        fused_results = self._reciprocal_rank_fusion(all_vector_results, all_bm25_results)
        fusion_ms = (time.perf_counter() - started) * 1000
        
        if not fused_results:
            self.last_timings = {
                "semantic_search": round(semantic_ms, 2),
                "lexical_search": round(lexical_ms, 2),
                "fusion": round(fusion_ms, 2),
                "reranker": 0.0,
                "retriever_total": round((time.perf_counter() - started_total) * 1000, 2),
            }
            return []

        # --- D. Reranking ---
        # Compare every fused chunk against the primary (first) original query
        primary_query = queries[0] 
        use_openai_reranker = bool(self.config["openai"].get("use_llm_reranker", False))
        if self.inference_mode == "openai" and use_openai_reranker:
            started = time.perf_counter()
            results = self._rerank_with_openai(primary_query, fused_results)
            reranker_ms = (time.perf_counter() - started) * 1000
            self.last_timings = {
                "semantic_search": round(semantic_ms, 2),
                "lexical_search": round(lexical_ms, 2),
                "fusion": round(fusion_ms, 2),
                "reranker": round(reranker_ms, 2),
                "retriever_total": round((time.perf_counter() - started_total) * 1000, 2),
            }
            return results

        if self.inference_mode == "openai":
            for item in fused_results:
                item["cross_score"] = 0.0
            self.last_timings = {
                "semantic_search": round(semantic_ms, 2),
                "lexical_search": round(lexical_ms, 2),
                "fusion": round(fusion_ms, 2),
                "reranker": 0.0,
                "retriever_total": round((time.perf_counter() - started_total) * 1000, 2),
            }
            return fused_results[:self.retrieval_cfg["final_top_n"]]

        started = time.perf_counter()
        cross_inp = [[primary_query, item["text"]] for item in fused_results]
        cross_scores = self.cross_encoder.predict(cross_inp, show_progress_bar=False)
        reranker_ms = (time.perf_counter() - started) * 1000
        
        # Attach scores and sort highest to lowest
        for i, score in enumerate(cross_scores):
            fused_results[i]["cross_score"] = float(score)
            
        reranked_results = sorted(fused_results, key=lambda x: x["cross_score"], reverse=True)
        
        # Return exact Top N defined in config
        self.last_timings = {
            "semantic_search": round(semantic_ms, 2),
            "lexical_search": round(lexical_ms, 2),
            "fusion": round(fusion_ms, 2),
            "reranker": round(reranker_ms, 2),
            "retriever_total": round((time.perf_counter() - started_total) * 1000, 2),
        }
        return reranked_results[:self.retrieval_cfg["final_top_n"]]

    def _rerank_with_openai(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Uses an OpenAI model as the reranking agent in API-only mode."""
        final_top_n = self.retrieval_cfg["final_top_n"]
        candidate_lines = []
        for item in candidates:
            snippet = item["text"].replace("\n", " ")[:900]
            candidate_lines.append(f"- id: {item['id']}\n  text: {snippet}")

        prompt = (
            "You are the CiteMentor reranking agent. Select the most relevant source chunks "
            "for answering the user query. Return only valid JSON in this shape: "
            "{\"ranked_ids\": [\"chunk_id_1\", \"chunk_id_2\"]}.\n\n"
            f"User query: {query}\n\n"
            f"Return exactly {min(final_top_n, len(candidates))} ids, ordered from most to least relevant.\n\n"
            "Candidate chunks:\n"
            + "\n".join(candidate_lines)
        )

        try:
            response = self.openai_reranker.invoke(prompt).content
            ranked_ids = self._parse_ranked_ids(response)
        except Exception:
            ranked_ids = []

        candidate_map = {item["id"]: item for item in candidates}
        reranked = []
        for score, chunk_id in enumerate(ranked_ids[::-1], start=1):
            item = candidate_map.get(chunk_id)
            if item:
                item["cross_score"] = float(score)
                reranked.insert(0, item)

        if len(reranked) < final_top_n:
            seen = {item["id"] for item in reranked}
            for item in candidates:
                if item["id"] not in seen:
                    item["cross_score"] = 0.0
                    reranked.append(item)
                if len(reranked) >= final_top_n:
                    break

        return reranked[:final_top_n]

    def _parse_ranked_ids(self, response: str) -> List[str]:
        try:
            return json.loads(response).get("ranked_ids", [])
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", response, flags=re.DOTALL)
            if not match:
                return []
            return json.loads(match.group(0)).get("ranked_ids", [])
