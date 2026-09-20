"""
PyTest suite for the Autonomous IT Incident Triage Engine.

Two groups of tests, matching what's actually risky to leave unverified:

  1. Endpoint/schema tests — the FastAPI <-> Pydantic contract. These
     mock out compiled_graph.invoke entirely, so they run instantly and
     need no API key, no ChromaDB, no network.

  2. Routing tests — mock out the Classifier Agent's LLM call and run
     the REAL compiled graph, to prove the conditional routing logic
     (classify -> spam / escalate / retrieve+resolve) actually works,
     again with no real API call.

Run:
    pytest -v
"""

from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from app.graph.graph import build_graph
from app.main import app

VALID_TICKET = {
    "ticket_id": "T-1",
    "subject": "VPN not connecting",
    "description": "Cannot connect since this morning",
}


@pytest.fixture
def client():
    return TestClient(app)


# ---------------------------------------------------------------------
# 1. Endpoint structure / Pydantic input-output contract
# ---------------------------------------------------------------------


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "bad_payload",
    [
        {"subject": "missing ticket_id and description"},
        {"ticket_id": "T-1", "description": "missing subject"},
        {"ticket_id": "T-1", "subject": "missing description"},
    ],
)
def test_triage_rejects_incomplete_ticket(client, bad_payload):
    """Pydantic must reject an incomplete ticket with 422 before the
    request ever reaches the graph."""
    response = client.post("/triage", json=bad_payload)
    assert response.status_code == 422


def test_triage_returns_full_state_shape_for_valid_ticket(client):
    """
    With the graph mocked out, this test is purely about the endpoint's
    contract: does a valid TicketInput get accepted, and does the
    response contain exactly the TriageState fields it's supposed to?
    """
    fake_final_state = {
        "ticket": {**VALID_TICKET, "requester_email": None, "priority": None},
        "category": "actionable",
        "classification_reason": "stub reason",
        "retrieved_context": ["stub policy text"],
        "final_response": "stub resolution",
    }

    with patch("app.main.compiled_graph.invoke", return_value=fake_final_state):
        response = client.post("/triage", json=VALID_TICKET)

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "ticket",
        "category",
        "classification_reason",
        "retrieved_context",
        "final_response",
    }
    assert body["ticket"]["ticket_id"] == "T-1"
    assert body["category"] == "actionable"
    assert body["final_response"] == "stub resolution"


def test_triage_pipeline_failure_returns_clean_500(client):
    """A raised exception inside the graph must surface as a handled
    500 with a detail message, never an unhandled crash."""
    with patch("app.main.compiled_graph.invoke", side_effect=RuntimeError("boom")):
        response = client.post("/triage", json=VALID_TICKET)

    assert response.status_code == 500
    assert "boom" in response.json()["detail"]


# ---------------------------------------------------------------------
# 2. Routing logic — mock the Classifier's LLM call, run the real graph
# ---------------------------------------------------------------------


def _initial_state(ticket: dict) -> dict:
    return {
        "ticket": ticket,
        "category": None,
        "classification_reason": None,
        "retrieved_context": None,
        "final_response": None,
    }


@pytest.mark.parametrize(
    "forced_category,expected_snippet",
    [
        ("spam", "identified as spam"),
        ("escalate", "escalated to a human"),
    ],
)
def test_graph_routes_terminal_categories_without_real_llm_call(
    forced_category, expected_snippet
):
    """
    Patches classify_ticket itself (not just get_llm) so this test is
    fully decoupled from LangChain/Mistral internals — it forces each
    category and proves the REAL compiled graph's conditional routing
    sends it to the right terminal node, with the right final_response,
    and with no live API or ChromaDB dependency at all (spam/escalate
    never reach retrieval).
    """

    def fake_classify_ticket(state: dict) -> dict:
        return {
            "category": forced_category,
            "classification_reason": f"forced for test: {forced_category}",
        }

    with patch("app.graph.graph.classify_ticket", side_effect=fake_classify_ticket):
        test_graph = build_graph()
        result = test_graph.invoke(_initial_state(VALID_TICKET))

    assert result["category"] == forced_category
    assert expected_snippet in result["final_response"].lower()


def test_graph_routes_actionable_to_retrieve_and_resolve():
    """
    Same technique, but for the 'actionable' branch: also mocks
    retrieve_context and draft_resolution (the only two nodes on this
    path that would otherwise need ChromaDB / a real LLM call) to prove
    the graph actually chains classify -> retrieve -> resolve -> END in
    that order.
    """

    def fake_classify_ticket(state: dict) -> dict:
        return {"category": "actionable", "classification_reason": "forced actionable"}

    def fake_retrieve_context(state: dict) -> dict:
        return {"retrieved_context": ["fake policy chunk"]}

    def fake_draft_resolution(state: dict) -> dict:
        # Prove this node actually received what retrieve_context produced.
        assert state["retrieved_context"] == ["fake policy chunk"]
        return {"final_response": "fake drafted resolution"}

    with (
        patch("app.graph.graph.classify_ticket", side_effect=fake_classify_ticket),
        patch("app.graph.graph.retrieve_context", side_effect=fake_retrieve_context),
        patch("app.graph.graph.draft_resolution", side_effect=fake_draft_resolution),
    ):
        test_graph = build_graph()
        result = test_graph.invoke(_initial_state(VALID_TICKET))

    assert result["category"] == "actionable"
    assert result["retrieved_context"] == ["fake policy chunk"]
    assert result["final_response"] == "fake drafted resolution"
