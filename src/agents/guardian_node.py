"""
Agent 03 — Garde-Fou

Responsibility: audit the generated answer against source chunks.
Produces a confidence score [0, 1]. Below threshold → triggers self-correction loop.

Audit checks:
- Species match: does the answer concern the right animal?
- Source grounding: is the answer supported by the chunks?
- Calculation validity: did the calculator succeed?
- No hallucinated product names
"""

import json
import re

from langchain_ollama import ChatOllama
from src.agents.state import VetState

_llm       = ChatOllama(model="llama3.2:3b", temperature=0.0)
THRESHOLD  = 0.85
MAX_ITER   = 3

AUDIT_PROMPT = """You are a veterinary drug safety auditor.
Evaluate the following answer against the source chunks and return a JSON audit.

Question: {query}
Answer: {answer}
Calculation: {calculation}

Source chunks:
{chunks}

Return ONLY a JSON object:
{{
  "score": <float between 0 and 1>,
  "species_match": <true/false>,
  "grounded_in_sources": <true/false>,
  "calculation_used": <true/false>,
  "issues": ["list of issues found, empty if none"]
}}

Scoring guide:
- 1.0: perfect — answer matches sources, correct species, calculation used
- 0.85-0.99: minor issues
- 0.5-0.84: significant issues, missing info
- 0.0-0.49: major problems, wrong species or hallucinated data

Return only the JSON, no explanation."""


def _audit(state: VetState) -> dict:
    """Ask LLM to audit the answer. Returns parsed audit dict."""
    chunks_text = "\n---\n".join(c["text"][:250] for c in state["chunks"][:3])
    calculation = state["calc_result"].get("calculation", "none")

    prompt   = AUDIT_PROMPT.format(
        query=state["query"],
        answer=state["answer"],
        calculation=calculation,
        chunks=chunks_text,
    )
    response = _llm.invoke(prompt)
    raw      = response.content.strip()
    raw      = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # conservative fallback — force retry
        return {"score": 0.0, "issues": ["audit parse error"], "parse_error": True}


def guardian_node(state: VetState) -> VetState:
    """
    LangGraph node.
    Audits the answer and writes audit_score to state.
    The conditional edge in graph.py decides whether to loop or exit.
    """
    audit  = _audit(state)
    score  = float(audit.get("score", 0.0))
    issues = audit.get("issues", [])

    print(f"[Garde-Fou] score={score} | issues={issues}")
    print(f"[Garde-Fou] species_match={audit.get('species_match')} | grounded={audit.get('grounded_in_sources')}")

    return {
        **state,
        "audit_score": score,
        "iterations":  state["iterations"] + 1,
    }


def should_retry(state: VetState) -> str:
    """
    Conditional edge function for LangGraph.
    Returns 'retry' to loop back to retriever, 'end' to exit.
    """
    if state["audit_score"] < THRESHOLD and state["iterations"] < MAX_ITER:
        print(f"[Garde-Fou] Score {state['audit_score']} < {THRESHOLD} — retrying (iter {state['iterations']})")
        return "retry"
    print(f"[Garde-Fou] Score {state['audit_score']} — validated after {state['iterations']} iteration(s)")
    return "end"
