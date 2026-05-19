# Vigilant-Vet-Orchestrator

> A sovereign multi-agent RAG system for veterinary drug compliance.  
> Not a chatbot. A workflow that retrieves, calculates, and audits.

---

## The Problem

A naive RAG is dangerous in veterinary medicine.

**1. Numerical hallucination.** Ask an LLM "10mg/kg for an 80kg pig" and it may return 800mg instead of 80mg. In a clinical context, that kills the animal. This is not a prompt engineering problem — it is an architecture problem.

**2. Vector search imprecision on molecule names.** "Amoxicillin" and "Ampicillin" are semantically close in vector space. They are chemically distinct. Keyword search is required.

This project solves both with a 3-agent orchestration layer.

---

## Architecture

```text
User Question
      |
      v
[Agent 01 — Retriever]  →  [Agent 02 — Calculator]  →  [Agent 03 — Garde-Fou]
 ChromaDB + BM25 + RRF      Python math (not LLM)       Audit + confidence score
                                                                |
                                                   score < 0.85 → loop back
                                                   score ≥ 0.85 → Validated Output
```

**Agent 01 — Retriever**: Hybrid search over ANMV regulatory PDFs. Runs ChromaDB vector search and BM25 in parallel, fuses via Reciprocal Rank Fusion. Returns top-5 chunks.

**Agent 02 — Calculator**: LLM extracts structured parameters (species, weight, mg/kg), Python computes the result. The LLM does not touch the arithmetic.

**Agent 03 — Garde-Fou**: Audits response against source chunks. Checks species, units, contraindications. Confidence score [0,1]. Below 0.85 → sends Agent 01 back with corrected instructions.

---

## Why Local-First

All inference runs on Ollama. No external API calls. No data sent to OpenAI or Anthropic.

For pharmaceutical R&D data, sovereign infrastructure is not optional — it is the architecture. Air-gap deployment is possible.

---

## Data

Real regulatory PDFs from the ANMV (Agence Nationale du Médicament Vétérinaire), sourced from [ircp.anmv.anses.fr](https://ircp.anmv.anses.fr). 50+ notices covering porcine and avian species. 100% public, 100% regulated.

---

## Stack

| Layer | Technology |
|---|---|
| Agent orchestration | LangGraph — StateGraph + conditional edges |
| LLM | Ollama · Mistral 7B (local inference) |
| Vector store | ChromaDB + sentence-transformers/all-MiniLM-L6-v2 |
| Keyword search | rank-bm25 (BM25Okapi) |
| Fusion | Reciprocal Rank Fusion (RRF) |
| PDF extraction | PyMuPDF (fitz) |
| API | FastAPI — async /query endpoint |
| UI | Streamlit — dark mode + agent traces tab |

---

## Project Structure

```text
vigilant-vet/
├── src/
│   ├── agents/
│   │   ├── state.py            # VetState TypedDict
│   │   ├── retriever_node.py   # Agent 01
│   │   ├── calculator_node.py  # Agent 02
│   │   ├── guardian_node.py    # Agent 03
│   │   └── graph.py            # LangGraph wiring
│   ├── retriever/
│   │   ├── ingest.py           # PDF → chunks → ChromaDB
│   │   └── hybrid_search.py    # BM25 + vector + RRF
│   ├── api/
│   │   └── main.py             # FastAPI /query
│   └── ui/
│       └── app.py              # Streamlit interface
├── data/
│   ├── raw_pdfs/               # ANMV PDFs (gitignored)
│   └── chromadb_store/         # vector DB (gitignored)
├── docs/
├── scripts/
│   └── download_anmv.py
├── .env.example
├── requirements.txt
└── README.md
```

---

## Getting Started

```bash
git clone https://github.com/Vanelfokamcode/vigilant-vet.git
cd vigilant-vet
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Pull the LLM
ollama pull mistral:7b

# Ingest ANMV PDFs (place PDFs in data/raw_pdfs/ first)
python scripts/ingest.py

# Run the API
uvicorn src.api.main:app --reload

# Run the UI
streamlit run src/ui/app.py
```

---

## Key Concepts

**RAG** — The LLM is not trained on your documents. At each query, relevant passages are injected into its context. No fine-tuning required, documents update independently of the model.

**Hybrid Search** — Vector search captures semantic proximity. BM25 captures exact keyword matches. RRF fuses both rankings. State of the art for precision-critical RAG in 2025.

**Embeddings** — Text transformed into a 384-dimensional vector. Semantically similar texts produce mathematically close vectors (cosine similarity). Model: all-MiniLM-L6-v2, runs locally.

**Self-correction loop** — Conditional edge in LangGraph. If audit_score < 0.85 and iterations < 3, routes back to Agent 01. Score and iteration count always exposed in the API response.

---

*Vanel FOKAM — Vigilant Infrastructure — 2026*
