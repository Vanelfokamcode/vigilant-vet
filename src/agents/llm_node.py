"""
LLM synthesis node — generates a veterinary answer from retrieved chunks.
Sits between the Calculator and the Garde-Fou in the graph.
"""

import re
from langchain_ollama import ChatOllama
from src.agents.state import VetState

_llm = ChatOllama(model="llama3.2:3b", temperature=0.1)

SYNTHESIS_PROMPT = """You are a veterinary drug compliance assistant.
Using the drug notice excerpts and dosage calculation below, write a clear,
accurate answer to the veterinary question. Always cite the product name.
Never invent dosages not present in the excerpts. Answer in the same language
as the question.

Question: {query}

Dosage calculation: {calculation}

Drug notice excerpts:
{chunks}

Answer:"""


def llm_node(state: VetState) -> VetState:
    """Generate a synthesized answer from chunks and calc_result."""
    query       = state["query"]
    chunks      = state["chunks"]
    calc_result = state["calc_result"]

    chunks_text = "\n---\n".join(c["text"][:300] for c in chunks[:3])
    calculation = calc_result.get("calculation", "No calculation available")

    prompt   = SYNTHESIS_PROMPT.format(
        query=query,
        calculation=calculation,
        chunks=chunks_text,
    )
    response = _llm.invoke(prompt)
    answer   = response.content.strip()

    print(f"[LLM] Answer generated ({len(answer)} chars)")
    return {**state, "answer": answer}
