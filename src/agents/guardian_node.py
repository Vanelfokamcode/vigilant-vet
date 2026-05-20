import json, re
from src.agents.state import VetState
from src.agents.gemini_client import GeminiClient

_llm      = None
THRESHOLD = 0.40
MAX_ITER  = 3

def get_llm():
    global _llm
    if _llm is None:
        _llm = GeminiClient()
    return _llm

AUDIT_PROMPT = """Tu es un auditeur de sécurité médicamenteuse vétérinaire.
Évalue la réponse ci-dessous par rapport aux extraits sources et retourne un JSON d'audit.

Question: {query}
Réponse: {answer}
Calcul: {calculation}

Extraits sources:
{chunks}

Retourne UNIQUEMENT ce JSON:
{{
  "score": <float entre 0 et 1>,
  "species_match": <true/false>,
  "grounded_in_sources": <true/false>,
  "calculation_used": <true/false>,
  "issues": ["liste des problèmes, vide si aucun"]
}}

Guide: 1.0=parfait, 0.85-0.99=mineur, 0.5-0.84=significatif, 0-0.49=majeur"""


def _audit(state):
    chunks_text = "\n---\n".join(c["text"][:250] for c in state["chunks"][:3])
    prompt      = AUDIT_PROMPT.format(
        query       = state["query"],
        answer      = state["answer"],
        calculation = state["calc_result"].get("calculation", "none"),
        chunks      = chunks_text,
    )
    response = get_llm().invoke(prompt)
    raw      = response.content.strip()
    raw      = re.sub(r"^```json\s*|^```\s*|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return {"score": 0.0, "issues": ["audit parse error"], "parse_error": True}


def guardian_node(state: VetState) -> VetState:
    audit  = _audit(state)
    score  = float(audit.get("score", 0.0))
    issues = audit.get("issues", [])
    print(f"[Garde-Fou] score={score} | issues={issues}")
    print(f"[Garde-Fou] species_match={audit.get('species_match')} | grounded={audit.get('grounded_in_sources')}")
    return {**state, "audit_score": score, "iterations": state["iterations"] + 1}


def should_retry(state: VetState) -> str:
    if state["audit_score"] < THRESHOLD and state["iterations"] < MAX_ITER:
        print(f"[Garde-Fou] Score {state['audit_score']} < {THRESHOLD} — retrying (iter {state['iterations']})")
        return "retry"
    print(f"[Garde-Fou] Score {state['audit_score']} — validated after {state['iterations']} iteration(s)")
    return "end"
