from src.agents.state import VetState
from src.retriever.hybrid_search import HybridSearcher

_searcher = None

def get_searcher():
    global _searcher
    if _searcher is None:
        _searcher = HybridSearcher()
    return _searcher

def retriever_node(state: VetState) -> VetState:
    query   = state["query"]
    results = get_searcher().search(query, k=5)
    print(f"[Retriever] query='{query}' → {len(results)} chunks retrieved")
    for i, r in enumerate(results):
        print(f"  [{i+1}] score={r['score']} | {r['metadata']['product_name'][:50]}")
    return {**state, "chunks": results}
