"""
Hybrid search combining ChromaDB vector search and BM25 keyword search.
Results are fused via Reciprocal Rank Fusion (RRF).

Why hybrid:
- Vector search: good on semantics, bad on exact molecule names
- BM25: good on exact keywords, bad on paraphrase
- RRF: covers both cases without tuning extra parameters
"""

import pickle
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

DATA_DIR      = Path(__file__).parent.parent.parent / "data"
CHROMA_DIR    = DATA_DIR / "chromadb_store"
BM25_PATH     = DATA_DIR / "bm25_corpus.pkl"
COLLECTION    = "anmv_notices"
EMBED_MODEL   = "sentence-transformers/all-MiniLM-L6-v2"

RRF_K = 60


class HybridSearcher:
    """
    Loads ChromaDB collection and BM25 index once at init.
    Call .search(query, k) to retrieve top-k chunks.
    """

    def __init__(self):
        print("[HybridSearcher] Loading vector store...")
        self.model      = SentenceTransformer(EMBED_MODEL)
        self.client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(COLLECTION)

        print("[HybridSearcher] Loading BM25 corpus...")
        with open(BM25_PATH, "rb") as f:
            corpus = pickle.load(f)

        self.chunks    = corpus["chunks"]
        self.metadatas = corpus["metadatas"]
        tokenized      = [doc.lower().split() for doc in self.chunks]
        self.bm25      = BM25Okapi(tokenized)
        print(f"[HybridSearcher] Ready — {len(self.chunks)} chunks indexed")

    def _vector_search(self, query: str, k: int) -> list[tuple[int, str, dict]]:
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k * 2, self.collection.count()),
        )
        docs      = results["documents"][0]
        metadatas = results["metadatas"][0]
        return [(i, doc, meta) for i, (doc, meta) in enumerate(zip(docs, metadatas))]

    def _bm25_search(self, query: str, k: int) -> list[tuple[int, str, dict]]:
        scores  = self.bm25.get_scores(query.lower().split())
        top_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k * 2]
        return [(rank, self.chunks[idx], self.metadatas[idx])
                for rank, idx in enumerate(top_idx)]

    def _rrf_merge(
        self,
        vec_results: list[tuple[int, str, dict]],
        bm25_results: list[tuple[int, str, dict]],
        k: int,
    ) -> list[dict]:
        """
        Score = sum(1 / (RRF_K + rank)) across both result lists.
        Higher score = better combined rank.
        """
        scores: dict[str, float] = {}
        texts:  dict[str, str]   = {}
        metas:  dict[str, dict]  = {}

        for rank, text, meta in vec_results:
            key = text[:120]
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
            texts[key]  = text
            metas[key]  = meta

        for rank, text, meta in bm25_results:
            key = text[:120]
            scores[key] = scores.get(key, 0) + 1 / (RRF_K + rank + 1)
            texts[key]  = text
            metas[key]  = meta

        ranked = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:k]
        return [
            {"text": texts[key], "metadata": metas[key], "score": round(scores[key], 4)}
            for key in ranked
        ]

    def search(self, query: str, k: int = 5) -> list[dict]:
        """
        Main entry point. Returns top-k chunks with metadata and RRF score.
        Each result: {"text": str, "metadata": dict, "score": float}
        """
        vec_results  = self._vector_search(query, k)
        bm25_results = self._bm25_search(query, k)
        return self._rrf_merge(vec_results, bm25_results, k)
