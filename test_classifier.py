"""
Live test for Step 3: the Classifier Agent + conditional routing.
Needs a real MISTRAL_API_KEY in .env (unlike smoke_test.py's parts 1-2,
every case here calls the actual LLM). Run from the project root:

    python test_classifier.py
"""

from app.graph.edges import ESCALATE, SPAM, route_after_classification
from app.graph.nodes import classify_ticket, handle_escalation, handle_spam

# One ticket per category we expect the classifier to land on. Real-world
# tickets won't always be this clear-cut — this is a first sanity check,
# not a full eval set.
TEST_TICKETS = [
    {
        "ticket_id": "T-1",
        "subject": "VPN not connecting",
        "description": "I've been unable to connect to the VPN since this morning. Getting a timeout error.",
    },
    {
        "ticket_id": "T-2",
        "subject": "CHEAP WATCHES 70% OFF",
        "description": "Click here to buy luxury watches at unbeatable prices!!! Limited time offer.",
    },
    {
        "ticket_id": "T-3",
        "subject": "Possible customer data breach",
        "description": "We just noticed unusual access patterns on the customer database from an unrecognized IP. This looks like it could be a security incident.",
    },
]


def run_ticket(ticket: dict) -> None:
    print(f"\n{'=' * 60}\nTicket {ticket['ticket_id']}: {ticket['subject']}")

    state = {
        "ticket": ticket,
        "category": None,
        "classification_reason": None,
        "retrieved_context": None,
        "final_response": None,
    }

    classification_update = classify_ticket(state)
    state.update(classification_update)
    print(f"  Category: {state['category']}")
    print(f"  Reason:   {state['classification_reason']}")

    route = route_after_classification(state)
    print(f"  Routes to: {route!r}")

    if route == SPAM:
        state.update(handle_spam(state))
        print(f"  Final response: {state['final_response']}")
    elif route == ESCALATE:
        state.update(handle_escalation(state))
        print(f"  Final response: {state['final_response']}")
    else:
        print("  (Would proceed to retrieval + resolution — built in Step 4)")


if __name__ == "__main__":
    for t in TEST_TICKETS:
        run_ticket(t)
