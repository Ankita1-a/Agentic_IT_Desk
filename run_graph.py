"""
Step 5 local test: runs real tickets through the fully compiled graph
(all real nodes — needs a working MISTRAL_API_KEY and an already-built
chroma_db from Step 1). Run from the project root:

    python run_graph.py
"""

from app.graph.graph import compiled_graph

DUMMY_TICKETS = [
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
        "description": "We just noticed unusual access patterns on the customer database from an unrecognized IP.",
    },
]


def run(ticket: dict) -> None:
    initial_state = {
        "ticket": ticket,
        "category": None,
        "classification_reason": None,
        "retrieved_context": None,
        "final_response": None,
    }

    final_state = compiled_graph.invoke(initial_state)

    print(f"\n{'=' * 60}")
    print(f"Ticket {ticket['ticket_id']}: {ticket['subject']}")
    print(f"  Category:  {final_state['category']}")
    print(f"  Reason:    {final_state['classification_reason']}")
    print(f"  Context:   {final_state['retrieved_context']}")
    print(f"  Response:  {final_state['final_response']}")


if __name__ == "__main__":
    for ticket in DUMMY_TICKETS:
        run(ticket)
