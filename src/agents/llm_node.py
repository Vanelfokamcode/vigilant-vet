from src.agents.state import VetState
from src.agents.gemini_client import GeminiClient

_llm = None

def get_llm():
    global _llm
    if _llm is None:
        _llm = GeminiClient()
    return _llm

SYNTHESIS_PROMPT = """Tu es un assistant réglementaire vétérinaire.
Réponds à la question en utilisant UNIQUEMENT les informations ci-dessous.
N'effectue AUCUN calcul. Recopie exactement le résultat de calcul fourni.

Question: {query}

Résultat de calcul officiel (NE PAS MODIFIER): {calculation}
Molécule: {molecule}
Espèce: {species}
Voie d'administration: {route}
Notes: {notes}

Extrait de notice source ({product}):
{chunk}

Rédige une réponse courte et factuelle en français. Cite le nom du produit.
Ne recalcule pas. Ne modifie pas le résultat de calcul."""


def llm_node(state: VetState) -> VetState:
    calc    = state["calc_result"]
    chunks  = state["chunks"]
    product = chunks[0]["metadata"]["product_name"] if chunks else "inconnu"
    chunk   = chunks[0]["text"][:400] if chunks else ""
    prompt  = SYNTHESIS_PROMPT.format(
        query       = state["query"],
        calculation = calc.get("calculation", "non disponible"),
        molecule    = calc.get("molecule", "inconnue"),
        species     = calc.get("species", "inconnue"),
        route       = calc.get("route", "non précisée"),
        notes       = calc.get("notes", "aucune"),
        product     = product,
        chunk       = chunk,
    )
    response = get_llm().invoke(prompt)
    answer   = response.content.strip()
    print(f"[LLM] Answer generated ({len(answer)} chars)")
    return {**state, "answer": answer}
