import json, re
from src.agents.state import VetState
from src.agents.gemini_client import GeminiClient

_llm = None
def get_llm():
    global _llm
    if _llm is None: _llm = GeminiClient()
    return _llm

EXTRACTION_PROMPT = """Tu es un expert en pharmacologie. Analyse l'extrait pour extraire les données EXACTES du produit demandé.

DANGER : Ne confonds pas les molécules. Si la question porte sur AMATIB, cherche uniquement les données liées à l'AMATIB ou à l'Amoxicilline.

Extraits : {chunks}
Question : {query}

Réponds en JSON :
{{
  "molecule": "nom exact de la substance active (ex: Amoxicilline)",
  "dose_mg_per_kg": <float>,
  "weight_kg": <float>,
  "animal_count": <int>,
  "is_found": <bool>
}}"""

def calculator_node(state: VetState) -> VetState:
    chunks_text = "\n---\n".join(c["text"] for c in state["chunks"])
    prompt = EXTRACTION_PROMPT.format(chunks=chunks_text, query=state["query"])
    try:
        raw = get_llm().invoke(prompt).content.strip()
        raw = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
        data = json.loads(raw)
        
        if data.get("is_found") and data.get("dose_mg_per_kg"):
            d = float(data["dose_mg_per_kg"])
            w = float(data.get("weight_kg", 1.0))
            n = int(data.get("animal_count", 1))
            total = round(d * w * n, 2)
            data["total_dose_mg"] = total
            data["calculation"] = f"{d} mg/kg x {w} kg x {n} animaux = {total} mg total"
            data["calculation_valid"] = True
        else:
            data["calculation_valid"] = False
            data["calculation"] = "Données non trouvées."
        return {**state, "calc_result": data}
    except:
        return {**state, "calc_result": {"calculation_valid": False}}
