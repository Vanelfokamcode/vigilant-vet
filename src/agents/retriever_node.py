"""
Agent 01 — Retriever

Responsibility: find the most relevant chunks from ANMV notices.
Uses HybridSearcher (ChromaDB + BM25 + RRF).
Single responsibility: populate state["chunks"]. Nothing else.
"""

from src.agents.state import VetState
from src.retriever.hybrid_search import HybridSearcher

# loaded once at module level — shared across all graph invocations
_searcher = HybridSearcher()


def retriever_node(state: VetState) -> VetState:
    """
    LangGraph node. Receives state, returns updated state.
    Runs hybrid search on state["query"] and stores top-5 chunks.
    """
    query   = state["query"]
    results = _searcher.search(query, k=5)

    print(f"[Retriever] query='{query}' → {len(results)} chunks retrieved")
    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r['score']} | {r['metadata']['product_name'][:50]}")

    return {**state, "chunks": results}
