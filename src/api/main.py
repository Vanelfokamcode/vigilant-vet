"""
FastAPI endpoint — exposes the LangGraph pipeline via HTTP.

POST /query
  body: {"question": "quelle dose amoxicilline pour un porc de 80kg"}
  returns: answer + sources + audit_score + iterations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.agents.graph import vet_graph

app = FastAPI(
    title="Vigilant-Vet API",
    description="Sovereign multi-agent RAG for veterinary drug compliance",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    question: str


class SourceItem(BaseModel):
    product_name: str
    species:      str
    score:        float
    excerpt:      str


class QueryResponse(BaseModel):
    answer:       str
    audit_score:  float
    iterations:   int
    calculation:  str
    sources:      list[SourceItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(body: QueryRequest):
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question cannot be empty")

    result = vet_graph.invoke({
        "query":       body.question,
        "chunks":      [],
        "answer":      "",
        "calc_result": {},
        "audit_score": 0.0,
        "iterations":  0,
    })

    sources = [
        SourceItem(
            product_name = c["metadata"].get("product_name", "")[:60],
            species      = c["metadata"].get("species", ""),
            score        = c["score"],
            excerpt      = c["text"][:200],
        )
        for c in result["chunks"][:3]
    ]

    return QueryResponse(
        answer      = result["answer"],
        audit_score = result["audit_score"],
        iterations  = result["iterations"],
        calculation = result["calc_result"].get("calculation", ""),
        sources     = sources,
    )
