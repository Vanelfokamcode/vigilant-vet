# Vigilant-Vet-Orchestrator

> A sovereign multi-agent RAG system for veterinary drug compliance.  
> Not a chatbot. A workflow that retrieves, calculates, and audits.

---

## The Problem

A naive RAG is dangerous in veterinary medicine.

Two failure modes:

**1. Numerical hallucination.** Ask an LLM "10mg/kg for an 80kg pig" and it may return 800mg instead of 80mg. In a clinical context, that kills the animal. This is not a prompt engineering problem — it is an architecture problem.

**2. Vector search imprecision on molecule names.** "Amoxicillin" and "Ampicillin" are semantically close in vector space. They are chemically distinct. Keyword search is required.

This project solves both with a 3-agent orchestration layer.

---

## Architecture
User Question
│
▼
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Agent 01   │────▶│   Agent 02   │────▶│    Agent 03     │
│  Retriever  │     │  Calculator  │     │   Garde-Fou     │
│             │     │              │     │                 │
│ ChromaDB    │     │ Python math  │     │ Audit + score   │
│ + BM25      │     │ (not LLM)    │     │ self-correction │
│ RRF fusion  │     │              │     │ loop if < 0.85  │
└─────────────┘     └──────────────┘     └─────────────────┘
│
score ≥ 0.85│
▼
Validated Output
(sources cited)

**Agent 01 — Retriever**: Hybrid search over ANMV regulatory PDFs. Runs ChromaDB vector search and BM25 keyword search in parallel, fuses results via Reciprocal Rank Fusion. Returns top-5 chunks.

**Agent 02 — Calculator**: Extracts structured parameters (species, weight, dosage in mg/kg) from retrieved text via LLM, then computes with Python. The LLM does not touch the arithmetic.

**Agent 03 — Garde-Fou**: Audits the full response against source chunks. Checks species match, unit consistency, cross-contraindications. Produces a confidence score [0, 1]. Below 0.85, it sends Agent 01 back with corrected instructions. Above 0.85, output is released.

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
vigilant-vet/
├── src/
│   ├── agents/
│   │   ├── state.py          # VetState TypedDict
│   │   ├── retriever_node.py # Agent 01
│   │   ├── calculator_node.py# Agent 02
│   │   ├── guardian_node.py  # Agent 03
│   │   └── graph.py          # LangGraph wiring
│   ├── retriever/
│   │   ├── ingest.py         # PDF → chunks → ChromaDB
│   │   └── hybrid_search.py  # BM25 + vector + RRF
│   ├── api/
│   │   └── main.py           # FastAPI /query
│   └── ui/
│       └── app.py            # Streamlit interface
├── data/
│   ├── raw_pdfs/             # ANMV source PDFs (gitignored)
│   └── chromadb_store/       # vector DB (gitignored)
├── docs/
├── scripts/
│   └── download_anmv.py
├── .env.example
├── requirements.txt
└── README.md

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

**RAG (Retrieval Augmented Generation)** — The LLM is not trained on your documents. At each query, relevant passages are injected into its context. It synthesizes. No fine-tuning required, documents update independently of the model.

**Hybrid Search** — Vector search captures semantic proximity. BM25 captures exact keyword matches. RRF fuses both rankings. State of the art for precision-critical RAG in 2025.

**Embeddings** — Text transformed into a 384-dimensional vector. Semantically similar texts produce mathematically close vectors (cosine similarity). Model: all-MiniLM-L6-v2, runs locally, zero cost.

**Self-correction loop** — A conditional edge in the LangGraph state machine. If audit_score < 0.85 and iterations < 3, the graph routes back to Agent 01 with corrected instructions. Observability is native: score and iteration count are always exposed in the API response.

---

*Vanel FOKAM — Vigilant Infrastructure — 2026*
