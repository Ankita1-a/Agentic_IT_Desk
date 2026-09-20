"""
Wires the nodes (nodes.py) and routing logic (edges.py) into a compiled
LangGraph StateGraph.

Graph shape:

              ┌─────────────┐
   START ───► │  classify   │
              └──────┬──────┘
                      │  route_after_classification()
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
   ┌─────────┐   ┌─────────┐   ┌───────────┐
   │ retrieve│   │  spam   │   │ escalate  │
   └────┬────┘   └────┬────┘   └─────┬─────┘
        ▼             ▼               ▼
   ┌─────────┐        │               │
   │ resolve │        │               │
   └────┬────┘        │               │
        ▼             ▼               ▼
                    END
"""

from langgraph.graph import END, StateGraph

from app.graph.edges import ESCALATE, RETRIEVE, SPAM, route_after_classification
from app.graph.nodes import (
    classify_ticket,
    draft_resolution,
    handle_escalation,
    handle_spam,
    retrieve_context,
)
from app.graph.state import TriageState


def build_graph():
    graph = StateGraph(TriageState)

    graph.add_node("classify", classify_ticket)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("resolve", draft_resolution)
    graph.add_node("spam", handle_spam)
    graph.add_node("escalate", handle_escalation)

    graph.set_entry_point("classify")

    graph.add_conditional_edges(
        "classify",
        route_after_classification,
        {
            RETRIEVE: "retrieve",
            SPAM: "spam",
            ESCALATE: "escalate",
        },
    )

    graph.add_edge("retrieve", "resolve")
    graph.add_edge("resolve", END)
    graph.add_edge("spam", END)
    graph.add_edge("escalate", END)

    return graph.compile()


# Built once at import time and reused — the FastAPI wrapper in Step 6
# will import this same compiled_graph rather than rebuilding per request.
compiled_graph = build_graph()
