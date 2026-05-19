"""
Agent 02 — Calculator

Responsibility: extract dosage parameters from retrieved chunks via LLM,
then compute the result in pure Python. The LLM never does arithmetic.

Why split extract/compute:
- LLM is good at reading unstructured text and extracting structured data
- LLM is unreliable on arithmetic (10mg/kg * 80kg = 800mg hallucination)
- Python is 100% reliable on arithmetic
"""

import json
import re

from langchain_ollama import ChatOllama
from src.agents.state import VetState

_llm = ChatOllama(model="llama3.2:3b", temperature=0.0)

EXTRACTION_PROMPT = """You are a veterinary pharmacology assistant.
Given the following drug notice chunks and a user question, extract dosage parameters as JSON.

Return ONLY a JSON object with these fields (use null if not found):
{{
  "molecule": "name of the active substance",
  "species": "target animal species",
  "dose_mg_per_kg": <float or null>,
  "dose_mg_per_animal": <float or null>,
  "weight_kg": <float or null>,
  "frequency": "dosing frequency as string or null",
  "route": "administration route or null",
  "notes": "any critical warnings in one sentence or null"
}}

User question: {query}

Drug notice chunks:
{chunks}

Return only the JSON object, no explanation."""


def _extract_params(query: str, chunks: list) -> dict:
    """Ask LLM to extract structured dosage parameters from chunks."""
    chunks_text = "\n---\n".join(c["text"][:400] for c in chunks[:3])
    prompt      = EXTRACTION_PROMPT.format(query=query, chunks=chunks_text)

    response = _llm.invoke(prompt)
    raw      = response.content.strip()

    # strip markdown fences if present
    raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # fallback: return raw text in a dict so the pipeline doesn't break
        return {"raw_extraction": raw, "parse_error": True}


def _compute_dose(params: dict) -> dict:
    """
    Pure Python dosage calculation. No LLM involved.
    Returns enriched params dict with computed fields.
    """
    result = {**params}

    dose_per_kg = params.get("dose_mg_per_kg")
    weight      = params.get("weight_kg")

    if dose_per_kg and weight:
        total = round(dose_per_kg * weight, 2)
        result["total_dose_mg"]    = total
        result["calculation"]      = f"{dose_per_kg} mg/kg × {weight} kg = {total} mg"
        result["calculation_valid"] = True
    elif params.get("dose_mg_per_animal"):
        result["total_dose_mg"]    = params["dose_mg_per_animal"]
        result["calculation"]      = f"Fixed dose: {params['dose_mg_per_animal']} mg/animal"
        result["calculation_valid"] = True
    else:
        result["total_dose_mg"]    = None
        result["calculation"]      = "Insufficient data for calculation"
        result["calculation_valid"] = False

    return result


def calculator_node(state: VetState) -> VetState:
    """
    LangGraph node.
    1. LLM extracts structured params from chunks
    2. Python computes the dose
    3. Result stored in state["calc_result"]
    """
    query  = state["query"]
    chunks = state["chunks"]

    if not chunks:
        print("[Calculator] No chunks available — skipping")
        return {**state, "calc_result": {"calculation_valid": False}}

    print("[Calculator] Extracting dosage parameters via LLM...")
    params = _extract_params(query, chunks)

    if params.get("parse_error"):
        print(f"[Calculator] JSON parse failed — raw: {params.get('raw_extraction', '')[:100]}")
    else:
        print(f"[Calculator] Extracted: {params}")

    result = _compute_dose(params)
    print(f"[Calculator] {result.get('calculation', 'no calculation')}")

    return {**state, "calc_result": result}
