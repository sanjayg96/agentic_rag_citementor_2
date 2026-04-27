import os
from dotenv import load_dotenv
import json
import pickle
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from mlx_lm import load, generate
import chromadb
from chromadb.utils import embedding_functions
from rank_bm25 import BM25Okapi
import yaml
from tqdm.auto import tqdm

# Load environment variables from .env file
load_dotenv()

# Paths
BOOKS_DIR = Path("data/books")
CHROMA_DIR = Path("storage/chroma_db")
BM25_DIR = Path("storage/bm25")
CATALOG_PATH = Path("catalog.json")
CONFIG_PATH = Path("config/retrieval.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def generate_contextual_summary(prev_text: str, current_text: str, next_text: str, model, tokenizer) -> str:
    """Uses quantized Mistral via MLX to summarize the chunk within its broader context."""
    prompt = (
        f"You are an AI tasked with providing a single-sentence contextual summary of a 'Current Chunk' of text. "
        f"Use the 'Previous' and 'Next' context to understand the broader narrative, but summarize ONLY the current chunk.\n\n"
        f"Previous Context:\n{prev_text}\n\n"
        f"Current Chunk:\n{current_text}\n\n"
        f"Next Context:\n{next_text}\n\n"
        f"Provide only a single-sentence summary of the Current Chunk's main idea. Do not add commentary:\nSummary:"
    )
    # MLX generation optimized for Apple Silicon
    response = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)
    return response.strip()

def process_and_ingest():
    print("🚀 Starting Local MLX Ingestion Pipeline...")
    config = load_config()
    
    # 1. Load MLX Model for Contextual Chunking
    print(f"Loading local LLM: {config['system']['local_llm']}...")
    llm_model, llm_tokenizer = load(config["system"]["local_llm"])
    
    # 2. Setup Vector Store (Chroma)
    print(f"Loading Embedding Model: {config['system']['embedding_model']}...")
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=config["system"]["embedding_model"],
        trust_remote_code=True
    )
    
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # We must explicitly pass the emb_fn here so it doesn't use the default
    collection = chroma_client.get_or_create_collection(
        name="citementor_library",
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 3. Load Catalog
    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len
    )

    all_bm25_corpus = []
    bm25_metadata = []

    # 4. Process Each Book
    for book_id, metadata in catalog.items():
        # Look for the exact file name, fallback to book_id.pdf if missing
        file_name = metadata.get("file_name", f"{book_id}.pdf")
        pdf_path = BOOKS_DIR / file_name
        
        if not pdf_path.exists():
            print(f"⚠️ Warning: PDF for {metadata['title']} not found at {pdf_path}. Skipping.")
            continue
            
        print(f"\n📚 Processing {metadata['title']}...")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)
        
        chunk_texts = []
        chunk_metadatas = []
        chunk_ids = []

        total_chunks = len(chunks)

        for i, chunk in tqdm(enumerate(chunks), total=total_chunks, desc="Processing chunks"):
            # Isolate neighboring chunks (handling boundaries)
            prev_text = chunks[i-1].page_content if i > 0 else ""
            next_text = chunks[i+1].page_content if i < total_chunks - 1 else ""
            
            # Contextual Chunking: Prepend AI summary
            summary = generate_contextual_summary(prev_text, chunk.page_content, next_text, llm_model, llm_tokenizer)
            enriched_text = f"[Context: {summary}]\n{chunk.page_content}"
            
            chunk_texts.append(enriched_text)
            all_bm25_corpus.append(enriched_text.split()) # Tokenize for BM25
            
            meta = {
                "book_id": book_id,
                "title": metadata["title"],
                "author": metadata["author"],
                "chunk_index": i
            }
            chunk_metadatas.append(meta)
            bm25_metadata.append(meta)
            chunk_ids.append(f"{book_id}_chunk_{i}")

        # Add to Chroma
        print(f"Adding {len(chunk_texts)} chunks to Vector Store...")
        collection.add(
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids
        )
        
        # Update Catalog Chunk Count
        catalog[book_id]["total_chunks"] = len(chunk_texts)

    # 5. Build and Save BM25 Lexical Index
    print("\n🔨 Building BM25 Lexical Index...")
    if all_bm25_corpus:
        bm25 = BM25Okapi(all_bm25_corpus)
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        with open(BM25_DIR / "bm25_index.pkl", "wb") as f:
            pickle.dump({"model": bm25, "metadata": bm25_metadata, "corpus": all_bm25_corpus}, f)

    # 6. Save Updated Catalog
    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)
        
    print("\n✅ Ingestion Complete! Catalog updated with total chunk counts.")

if __name__ == "__main__":
    process_and_ingest()