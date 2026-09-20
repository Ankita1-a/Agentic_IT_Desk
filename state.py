"""
- TicketInput: the Pydantic model validating the FastAPI request body.
- TriageState: the TypedDict LangGraph passes between nodes. Every node
  reads from and returns a (partial) update to this shared state.
"""

from typing import List, Optional, TypedDict

from pydantic import BaseModel, Field


class TicketInput(BaseModel):
    """Shape of the incoming JSON IT support ticket."""

    ticket_id: str = Field(..., description="Unique identifier from the ticketing system")
    subject: str = Field(..., description="Short ticket title/subject line")
    description: str = Field(..., description="Full body text of the user's request")
    requester_email: Optional[str] = Field(
        default=None, description="Email of the person who filed the ticket"
    )
    priority: Optional[str] = Field(
        default=None, description="Priority as set by the requester, e.g. 'low'/'high'"
    )


class TriageState(TypedDict):
    """
    Shared state threaded through the graph. Every node takes this dict
    (or a subset it needs) and returns a partial dict of updates that
    LangGraph merges back in.
    """

    ticket: dict
    category: Optional[str]          # "actionable" | "spam" | "escalate"
    classification_reason: Optional[str]
    retrieved_context: Optional[List[str]]
    final_response: Optional[str]
