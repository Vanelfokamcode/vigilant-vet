"""
LangGraph state machine — wires the 4 nodes into a complete pipeline.

Flow:
  retriever → calculator → llm → guardian
                                     │
                        score < 0.70 │ iterations < 3
                                     ▼
                                  retry → retriever
                                     │
                        score ≥ 0.70 │ iterations >= 3
                                     ▼
                                    END
"""

from langgraph.graph import StateGraph, END

from src.agents.state import VetState
from src.agents.retriever_node import retriever_node
from src.agents.calculator_node import calculator_node
from src.agents.llm_node import llm_node
from src.agents.guardian_node import guardian_node, should_retry


def build_graph():
    graph = StateGraph(VetState)

    graph.add_node("retriever",  retriever_node)
    graph.add_node("calculator", calculator_node)
    graph.add_node("llm",        llm_node)
    graph.add_node("guardian",   guardian_node)

    graph.set_entry_point("retriever")
    graph.add_edge("retriever",  "calculator")
    graph.add_edge("calculator", "llm")
    graph.add_edge("llm",        "guardian")

    graph.add_conditional_edges(
        "guardian",
        should_retry,
        {
            "retry": "retriever",
            "end":   END,
        }
    )

    return graph.compile()


# compiled graph — imported by FastAPI and Streamlit
vet_graph = build_graph()
