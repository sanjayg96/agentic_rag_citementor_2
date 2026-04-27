import yaml
import pickle
from pathlib import Path
from typing import List, Dict, Any
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import CrossEncoder

# Paths
CONFIG_PATH = Path("config/retrieval.yaml")
CHROMA_DIR = Path("storage/chroma_db")
BM25_DIR = Path("storage/bm25/bm25_index.pkl")

class HybridRetriever:
    def __init__(self):
        with open(CONFIG_PATH, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.retrieval_cfg = self.config["retrieval"]
        
        # 1. Initialize Embeddings (Must match ingestion model)
        self.emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=self.config["system"]["embedding_model"],
            trust_remote_code=True
        )
        
        # 2. Connect to ChromaDB
        self.chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.chroma_client.get_or_create_collection(
            name="citementor_library",
            embedding_function=self.emb_fn
        )
        
        # 3. Load BM25 Index
        with open(BM25_DIR, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25_model = bm25_data["model"]
            self.bm25_metadata = bm25_data["metadata"]
            
        # 4. Load Cross-Encoder for Reranking
        # This small local model re-scores chunks for maximum relevance
        print(f"Loading Cross-Encoder: {self.config['system']['reranker_model']}...")
        self.cross_encoder = CrossEncoder(self.config["system"]["reranker_model"])

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
        all_vector_results = []
        all_bm25_results = []
        
        seen_vector_ids = set()
        seen_bm25_ids = set()

        for query in queries:
            # --- A. Semantic Search (Chroma) ---
            vector_res = self.collection.query(
                query_texts=[query],
                n_results=self.retrieval_cfg["semantic_top_k"]
            )
            
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
            tokenized_query = query.split()
            bm25_scores = self.bm25_model.get_scores(tokenized_query)
            top_n_indices = bm25_scores.argsort()[::-1][:self.retrieval_cfg["lexical_top_k"]]
            
            for idx in top_n_indices:
                meta = self.bm25_metadata[idx]
                chunk_id = f"{meta['book_id']}_chunk_{meta['chunk_index']}"
                
                if chunk_id not in seen_bm25_ids:
                    # Fetch actual text from Chroma to avoid storing it twice on disk
                    fetch_res = self.collection.get(ids=[chunk_id])
                    if fetch_res and fetch_res["documents"]:
                        all_bm25_results.append({
                            "id": chunk_id,
                            "text": fetch_res["documents"][0],
                            "book_id": meta["book_id"],
                            "author": meta["author"]
                        })
                        seen_bm25_ids.add(chunk_id)

        # --- C. Reciprocal Rank Fusion ---
        fused_results = self._reciprocal_rank_fusion(all_vector_results, all_bm25_results)
        
        if not fused_results:
            return []

        # --- D. Cross-Encoder Reranking ---
        # Compare every fused chunk against the primary (first) original query
        primary_query = queries[0] 
        cross_inp = [[primary_query, item["text"]] for item in fused_results]
        cross_scores = self.cross_encoder.predict(cross_inp)
        
        # Attach scores and sort highest to lowest
        for i, score in enumerate(cross_scores):
            fused_results[i]["cross_score"] = float(score)
            
        reranked_results = sorted(fused_results, key=lambda x: x["cross_score"], reverse=True)
        
        # Return exact Top N defined in config
        return reranked_results[:self.retrieval_cfg["final_top_n"]]