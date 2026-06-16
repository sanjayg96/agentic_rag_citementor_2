"""Headless CLI for exercising the compiled CiteMentor LangGraph.

Usage:
    uv run python main.py "How do I handle a toxic boss?"

Runs the same compiled graph the Streamlit app drives, then prints the chosen
route, the grounded answer, and the source titles. Useful for smoke-testing the
orchestration without launching the UI.
"""

import json
import sys

from src.core.graph import app_graph

with open("catalog.json", "r") as f:
    CATALOG = json.load(f)


def main() -> None:
    query = " ".join(sys.argv[1:]).strip()
    if not query:
        print('Usage: uv run python main.py "your question here"')
        raise SystemExit(1)

    final_state = app_graph.invoke({"query": query, "timings": {}})

    print(f"\nQuery: {query}")
    print(f"Route: {final_state.get('route', 'n/a')}")
    print(f"\nAnswer:\n{final_state.get('answer', '')}\n")

    sources = final_state.get("retrieved_chunks", [])
    if sources:
        print("Sources:")
        for chunk in sources:
            title = CATALOG.get(chunk["book_id"], {}).get("title", chunk["book_id"])
            print(f"  - {title} ({chunk.get('author', 'Unknown')})")


if __name__ == "__main__":
    main()
