"""
Classifier Agent: the first node every ticket passes through.

Uses `with_structured_output` (tool-calling under the hood) rather than
asking the LLM to output a category as free text + parsing it, so we get
a validated category every time instead of brittle string matching.
"""

from typing import Literal

import chromadb
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from app.core.config import CHROMA_PERSIST_DIR, IT_POLICY_COLLECTION
from app.core.llm import get_llm
from app.graph.state import TriageState


class ClassificationResult(BaseModel):
    category: Literal["actionable", "spam", "escalate"] = Field(
        description=(
            "'actionable' if this is a legitimate IT support request that can "
            "likely be resolved by following an internal policy/runbook. "
            "'spam' if it is not a genuine IT request at all (promotional "
            "content, test/garbage text, clearly unrelated to IT). "
            "'escalate' if it is a genuine IT issue but too sensitive, urgent, "
            "or high-impact for an automated response — e.g. security "
            "incidents, data breaches, outages affecting many users, legal/HR "
            "matters, or anything the user explicitly demands a human for."
        )
    )
    reason: str = Field(description="One sentence explaining why this category was chosen")


CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an IT support triage classifier. Read the ticket and "
            "assign exactly one category: actionable, spam, or escalate.",
        ),
        ("human", "Subject: {subject}\n\nDescription: {description}"),
    ]
)


def classify_ticket(state: TriageState) -> dict:
    """Classifier Agent node. Reads state['ticket'], returns state updates."""
    ticket = state["ticket"]

    llm = get_llm()
    structured_llm = llm.with_structured_output(ClassificationResult)
    chain = CLASSIFIER_PROMPT | structured_llm

    result: ClassificationResult = chain.invoke(
        {"subject": ticket["subject"], "description": ticket["description"]}
    )

    return {
        "category": result.category,
        "classification_reason": result.reason,
    }


def handle_spam(state: TriageState) -> dict:
    """Terminal node for the 'spam' branch — no LLM call needed."""
    return {
        "final_response": (
            "This ticket was identified as spam / not a genuine IT support "
            "request and has been closed automatically. No action taken."
        )
    }


def handle_escalation(state: TriageState) -> dict:
    """Terminal node for the 'escalate' branch — hands off to a human."""
    reason = state.get("classification_reason", "requires human review")
    return {
        "final_response": (
            f"This ticket has been escalated to a human IT support agent. "
            f"Reason: {reason}"
        )
    }


def retrieve_context(state: TriageState) -> dict:
    """
    Retrieval Agent node. Only reached for 'actionable' tickets. Queries the
    persisted ChromaDB collection for the internal policy text most relevant
    to this ticket.
    """
    ticket = state["ticket"]
    query_text = f"{ticket['subject']} {ticket['description']}"

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    collection = client.get_collection(name=IT_POLICY_COLLECTION)

    results = collection.query(query_texts=[query_text], n_results=2)
    retrieved_docs = results["documents"][0]  # top matches for the single query

    return {"retrieved_context": retrieved_docs}


RESOLUTION_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an IT support assistant drafting a resolution for a "
            "support ticket. Base your answer strictly on the provided "
            "internal policy context. If the context doesn't fully cover the "
            "issue, say so honestly instead of inventing steps that aren't "
            "in the policy.",
        ),
        (
            "human",
            "Ticket subject: {subject}\n"
            "Ticket description: {description}\n\n"
            "Relevant internal policy context:\n{context}\n\n"
            "Write a clear, actionable resolution response to send to the user.",
        ),
    ]
)


def draft_resolution(state: TriageState) -> dict:
    """
    Resolution Agent node. Drafts the final response using the ticket plus
    whatever context the Retrieval Agent found.
    """
    ticket = state["ticket"]
    context = state.get("retrieved_context") or []
    context_text = (
        "\n\n---\n\n".join(context) if context else "No matching internal policy found."
    )

    llm = get_llm()
    chain = RESOLUTION_PROMPT | llm

    response = chain.invoke(
        {
            "subject": ticket["subject"],
            "description": ticket["description"],
            "context": context_text,
        }
    )

    return {"final_response": response.content}