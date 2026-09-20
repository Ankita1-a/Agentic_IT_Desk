"""
FastAPI wrapper for the Autonomous IT Incident Triage Engine.

Run:
    uvicorn app.main:app --reload

Then POST a ticket:
    curl -X POST http://127.0.0.1:8000/triage \\
      -H "Content-Type: application/json" \\
      -d '{"ticket_id": "T-1", "subject": "VPN not connecting", "description": "Cannot connect since this morning"}'
"""

from fastapi import FastAPI, HTTPException

from app.graph.graph import compiled_graph
from app.graph.state import TicketInput

app = FastAPI(title="Autonomous IT Incident Triage Engine")


@app.post("/triage")
def triage_ticket(ticket: TicketInput) -> dict:
    """
    Accepts a JSON IT support ticket, runs it through the LangGraph
    pipeline (classify -> retrieve -> resolve, or classify -> spam/escalate),
    and returns the final graph state as JSON.
    """
    initial_state = {
        "ticket": ticket.model_dump(),
        "category": None,
        "classification_reason": None,
        "retrieved_context": None,
        "final_response": None,
    }

    try:
        final_state = compiled_graph.invoke(initial_state)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Triage pipeline failed: {e}"
        ) from e

    return final_state


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}
