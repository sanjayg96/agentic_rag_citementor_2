import json
import os
from functools import lru_cache
from typing import Dict, Any

CATALOG_PATH = "catalog.json"

@lru_cache(maxsize=1)
def load_catalog() -> Dict[str, Any]:
    """Loads the book metadata and pricing catalog."""
    if not os.path.exists(CATALOG_PATH):
        raise FileNotFoundError(f"Missing {CATALOG_PATH}. Please ensure it exists in the root.")
    with open(CATALOG_PATH, "r") as f:
        return json.load(f)

def calculate_snippet_cost(book_id: str) -> float:
    """Calculates the micro-royalty cost for a single retrieved chunk."""
    catalog = load_catalog()
    book = catalog.get(book_id)
    
    # Return 0 if book is missing or ingestion hasn't counted chunks yet
    if not book or book.get("total_chunks", 0) == 0:
        return 0.0
        
    p_book = float(book["retail_price"])
    n_chunks = int(book["total_chunks"])
    
    # The Core Formula
    c_s = (1.0 / n_chunks) * p_book
    return round(c_s, 6)

def record_transaction(session_state: dict, book_id: str) -> float:
    """
    Updates the tracking ledger in the current session state.
    Pass st.session_state directly into this function from the UI.
    """
    if "royalties" not in session_state:
        session_state["royalties"] = 0.0
    if "ledger_details" not in session_state:
        session_state["ledger_details"] = []
        
    cost = calculate_snippet_cost(book_id)
    session_state["royalties"] += cost
    
    # Log the specific transaction for the dashboard
    if cost > 0:
        session_state["ledger_details"].append({
            "book_id": book_id,
            "cost": cost
        })
        
    return cost
