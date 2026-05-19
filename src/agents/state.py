"""
Shared state passing through the LangGraph state machine.
Every node reads from and writes to this TypedDict.
"""

from typing import TypedDict


class VetState(TypedDict):
    query:        str         # original user question
    chunks:       list        # retrieved chunks from hybrid search
    answer:       str         # generated answer from LLM
    calc_result:  dict        # structured dosage calculation output
    audit_score:  float       # confidence score from Garde-Fou [0, 1]
    iterations:   int         # number of self-correction loops so far
