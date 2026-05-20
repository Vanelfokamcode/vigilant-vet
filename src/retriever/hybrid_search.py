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

class HybridSearcher:
    def __init__(self):
        self.model      = SentenceTransformer(EMBED_MODEL)
        self.client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self.collection = self.client.get_collection(COLLECTION)
        with open(BM25_PATH, "rb") as f:
            corpus = pickle.load(f)
        self.chunks    = corpus["chunks"]
        self.metadatas = corpus["metadatas"]
        tokenized      = [doc.lower().split() for doc in self.chunks]
        self.bm25      = BM25Okapi(tokenized)

    def search(self, query: str, k: int = 5) -> list[dict]:
        query_upper = query.upper()
        # On extrait les mots importants (Noms de produits)
        keywords = [w for w in query_upper.replace("'", " ").split() if len(w) > 3]
        
        scores = {}
        
        # On parcourt TOUS les chunks pour trouver le produit EXACT (Brute force sur les noms)
        for i, meta in enumerate(self.metadatas):
            doc = self.chunks[i]
            p_name = meta.get("product_name", "").upper()
            
            # Si le nom du produit est dans la question
            for kw in keywords:
                if kw in p_name:
                    key = doc[:120]
                    # BOOST MASSIF si c'est le bon produit ET la section Posologie
                    bonus = 10.0 if kw in p_name else 0.0
                    if "3.9" in doc or "POSOLOGIE" in doc.upper() or "MG/KG" in doc.upper():
                        bonus += 5.0
                    
                    score = bonus + (1 / (60 + i + 1))
                    if key not in scores or score > scores[key]["score"]:
                        scores[key] = {"text": doc, "metadata": meta, "score": score}

        # Si on n'a rien trouvé avec le nom, on utilise le vectoriel en secours
        if not scores:
            v_res = self.collection.query(query_texts=[query], n_results=10)
            for i, (doc, meta) in enumerate(zip(v_res["documents"][0], v_res["metadatas"][0])):
                key = doc[:120]
                scores[key] = {"text": doc, "metadata": meta, "score": 1/(60+i)}

        sorted_res = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return sorted_res[:k]
