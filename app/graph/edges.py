"""
Conditional edge logic: decides which node runs next based on the
Classifier Agent's output. Used with graph.add_conditional_edges(...)
in Step 5.

Returns a routing key, not a node name directly — Step 5 maps these keys
to actual node names when wiring the graph, which keeps this function
decoupled from the final graph's node naming.
"""

from app.graph.state import TriageState

RETRIEVE = "retrieve"
SPAM = "spam"
ESCALATE = "escalate"


def route_after_classification(state: TriageState) -> str:
    category = state.get("category")

    if category == "actionable":
        return RETRIEVE
    if category == "spam":
        return SPAM
    if category == "escalate":
        return ESCALATE

    # Defensive fallback: an unexpected/missing category should never
    # silently fall through to auto-resolution — route it to a human.
    return ESCALATE
