import argparse
import json
import os
import pickle
import re
from pathlib import Path

import chromadb
import yaml
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi
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


def get_openai_api_key() -> str:
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("CHROMA_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for OpenAI ingestion.")
    return api_key


def load_context_model(profile: str, config: dict):
    if profile == "local":
        from mlx_lm import load

        print(f"Loading local LLM: {config['system']['local_llm']}...")
        return load(config["system"]["local_llm"])

    from langchain_openai import ChatOpenAI

    print(f"Using OpenAI context model: {config['openai']['ingestion_context_model']}...")
    return ChatOpenAI(model=config["openai"]["ingestion_context_model"], temperature=0)


def generate_contextual_summary(
    profile: str,
    prev_text: str,
    current_text: str,
    next_text: str,
    context_model,
) -> str:
    """Summarizes the current chunk using the configured local or OpenAI model."""
    prompt = (
        "You are an AI tasked with providing a single-sentence contextual summary "
        "of a 'Current Chunk' of text. Use the 'Previous' and 'Next' context to "
        "understand the broader narrative, but summarize ONLY the current chunk.\n\n"
        f"Previous Context:\n{prev_text}\n\n"
        f"Current Chunk:\n{current_text}\n\n"
        f"Next Context:\n{next_text}\n\n"
        "Provide only a single-sentence summary of the Current Chunk's main idea. "
        "Do not add commentary:\nSummary:"
    )

    if profile == "local":
        from mlx_lm import generate

        model, tokenizer = context_model
        response = generate(model, tokenizer, prompt=prompt, max_tokens=60, verbose=False)
        return response.strip()

    return context_model.invoke(prompt).content.strip()


def parse_json_object(text: str) -> dict:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def generate_openai_contextual_summaries(chunks: list, context_model, batch_size: int) -> list[str]:
    summaries = []

    for start in tqdm(range(0, len(chunks), batch_size), desc="Summarizing batches"):
        batch = chunks[start:start + batch_size]
        items = []
        for offset, chunk in enumerate(batch):
            index = start + offset
            prev_text = chunks[index - 1].page_content if index > 0 else ""
            next_text = chunks[index + 1].page_content if index < len(chunks) - 1 else ""
            items.append({
                "index": index,
                "previous": prev_text[:1200],
                "current": chunk.page_content[:2400],
                "next": next_text[:1200],
            })

        prompt = (
            "You are creating contextual summaries for retrieval-augmented generation. "
            "For each item, write exactly one concise sentence summarizing ONLY the current chunk, "
            "using previous and next only for context. Return only valid JSON in this exact shape: "
            "{\"summaries\": [{\"index\": 0, \"summary\": \"...\"}]}.\n\n"
            f"Items:\n{json.dumps(items, ensure_ascii=False)}"
        )

        response = context_model.invoke(prompt).content
        parsed = parse_json_object(response)
        by_index = {
            int(item["index"]): item["summary"].strip()
            for item in parsed.get("summaries", [])
            if "index" in item and "summary" in item
        }

        for offset, chunk in enumerate(batch):
            index = start + offset
            summaries.append(by_index.get(index, chunk.page_content[:180].replace("\n", " ")))

    return summaries


def create_embedding_function(profile: str, config: dict):
    if profile == "local":
        print(f"Loading embedding model: {config['system']['embedding_model']}...")
        return embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=config["system"]["embedding_model"],
            trust_remote_code=True,
        )

    print(f"Using OpenAI embedding model: {config['openai']['embedding_model']}...")
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=get_openai_api_key(),
        model_name=config["openai"]["embedding_model"],
    )


def get_collection_name(profile: str, config: dict) -> str:
    if profile == "local":
        return config["vector_stores"]["local_collection"]
    return config["vector_stores"]["openai_collection"]


def prepare_collection(profile: str, config: dict, reset: bool):
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection_name = get_collection_name(profile, config)

    if reset:
        try:
            chroma_client.delete_collection(collection_name)
            print(f"Deleted existing collection: {collection_name}")
        except Exception:
            pass

    emb_fn = create_embedding_function(profile, config)
    return chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )


def rebuild_openai_from_existing(config: dict, reset: bool):
    """Re-embeds already enriched local documents into the OpenAI collection."""
    print("Rebuilding OpenAI collection from existing enriched local Chroma documents...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    source_name = config["vector_stores"]["local_collection"]
    target = prepare_collection("openai", config, reset)

    source = chroma_client.get_collection(name=source_name)
    total = source.count()
    if total == 0:
        raise RuntimeError(f"Source collection {source_name} is empty. Run local ingestion first.")

    all_bm25_corpus = []
    bm25_metadata = []
    chunk_counts = {}
    batch_size = 100

    for offset in tqdm(range(0, total, batch_size), desc="Re-embedding batches"):
        batch = source.get(
            limit=batch_size,
            offset=offset,
            include=["documents", "metadatas"],
        )
        ids = batch["ids"]
        documents = batch["documents"]
        metadatas = batch["metadatas"]

        target.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas,
        )

        for document, metadata in zip(documents, metadatas):
            all_bm25_corpus.append(document.split())
            bm25_metadata.append(metadata)
            book_id = metadata["book_id"]
            chunk_counts[book_id] = chunk_counts.get(book_id, 0) + 1

    print("\nBuilding BM25 lexical index...")
    if all_bm25_corpus:
        bm25 = BM25Okapi(all_bm25_corpus)
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        with open(BM25_DIR / "bm25_index.pkl", "wb") as f:
            pickle.dump({"model": bm25, "metadata": bm25_metadata, "corpus": all_bm25_corpus}, f)

    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)
    for book_id, count in chunk_counts.items():
        if book_id in catalog:
            catalog[book_id]["total_chunks"] = count
    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"\nOpenAI collection rebuilt with {total} chunks.")


def process_and_ingest(profile: str = "local", reset: bool = False, source: str = "auto"):
    print(f"Starting {profile.upper()} ingestion pipeline...")
    config = load_config()

    if profile not in {"local", "openai"}:
        raise ValueError("profile must be either 'local' or 'openai'")

    if source == "auto":
        source = "existing" if profile == "openai" else "pdf"

    if profile == "openai" and source == "existing":
        rebuild_openai_from_existing(config, reset)
        return

    context_model = load_context_model(profile, config)
    collection = prepare_collection(profile, config, reset)

    with open(CATALOG_PATH, "r") as f:
        catalog = json.load(f)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
    )

    all_bm25_corpus = []
    bm25_metadata = []

    for book_id, metadata in catalog.items():
        file_name = metadata.get("file_name", f"{book_id}.pdf")
        pdf_path = BOOKS_DIR / file_name

        if not pdf_path.exists():
            print(f"Warning: PDF for {metadata['title']} not found at {pdf_path}. Skipping.")
            continue

        print(f"\nProcessing {metadata['title']}...")
        loader = PyPDFLoader(str(pdf_path))
        docs = loader.load()
        chunks = text_splitter.split_documents(docs)

        chunk_texts = []
        chunk_metadatas = []
        chunk_ids = []

        total_chunks = len(chunks)
        openai_summaries = None
        if profile == "openai":
            batch_size = int(config["openai"].get("ingestion_summary_batch_size", 20))
            openai_summaries = generate_openai_contextual_summaries(chunks, context_model, batch_size)

        for i, chunk in tqdm(enumerate(chunks), total=total_chunks, desc="Processing chunks"):
            if openai_summaries is not None:
                summary = openai_summaries[i]
            else:
                prev_text = chunks[i - 1].page_content if i > 0 else ""
                next_text = chunks[i + 1].page_content if i < total_chunks - 1 else ""
                summary = generate_contextual_summary(
                    profile,
                    prev_text,
                    chunk.page_content,
                    next_text,
                    context_model,
                )
            enriched_text = f"[Context: {summary}]\n{chunk.page_content}"

            chunk_texts.append(enriched_text)
            all_bm25_corpus.append(enriched_text.split())

            meta = {
                "book_id": book_id,
                "title": metadata["title"],
                "author": metadata["author"],
                "chunk_index": i,
            }
            chunk_metadatas.append(meta)
            bm25_metadata.append(meta)
            chunk_ids.append(f"{book_id}_chunk_{i}")

        print(f"Upserting {len(chunk_texts)} chunks into {collection.name}...")
        collection.upsert(
            documents=chunk_texts,
            metadatas=chunk_metadatas,
            ids=chunk_ids,
        )

        catalog[book_id]["total_chunks"] = len(chunk_texts)

    print("\nBuilding BM25 lexical index...")
    if all_bm25_corpus:
        bm25 = BM25Okapi(all_bm25_corpus)
        BM25_DIR.mkdir(parents=True, exist_ok=True)
        with open(BM25_DIR / "bm25_index.pkl", "wb") as f:
            pickle.dump({"model": bm25, "metadata": bm25_metadata, "corpus": all_bm25_corpus}, f)

    with open(CATALOG_PATH, "w") as f:
        json.dump(catalog, f, indent=2)

    print("\nIngestion complete.")


def parse_args():
    parser = argparse.ArgumentParser(description="Build CiteMentor Chroma and BM25 indices.")
    parser.add_argument(
        "--profile",
        choices=["local", "openai"],
        default="local",
        help="Embedding and contextualization profile to use.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the target Chroma collection before ingesting.",
    )
    parser.add_argument(
        "--source",
        choices=["auto", "pdf", "existing"],
        default="auto",
        help="Use PDFs, or re-embed existing enriched local Chroma documents.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    process_and_ingest(profile=args.profile, reset=args.reset, source=args.source)
