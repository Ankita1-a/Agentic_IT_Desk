"""
Sanity check for everything built in Step 1 + Step 2, before Step 3 wires
these pieces into an actual LangGraph graph. Run from the project root:

    python smoke_test.py

Exercises three things independently:
  1. TicketInput validation (Pydantic)
  2. ChromaDB retrieval quality against the mock IT policies
  3. A live round-trip call to the Mistral LLM (skipped if no API key set)
"""

from app.core.config import CHROMA_PERSIST_DIR, IT_POLICY_COLLECTION
from app.graph.state import TicketInput


def check_ticket_validation() -> None:
    print("== 1. TicketInput validation ==")
    ticket = TicketInput(
        ticket_id="T-1001",
        subject="VPN not connecting",
        description="I've been unable to connect to the VPN since this morning.",
    )
    print("OK:", ticket.model_dump())


def check_retrieval() -> None:
    print("\n== 2. ChromaDB retrieval ==")
    import chromadb

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    try:
        collection = client.get_collection(name=IT_POLICY_COLLECTION)
    except Exception as e:
        print(f"FAILED — did you run 'python -m app.db.init_chroma' first? ({e})")
        return

    sample_queries = [
        "I can't connect to the office VPN, it keeps timing out",
        "My laptop screen is flickering and won't turn on properly",
        "I need Adobe Photoshop installed for the design team",
    ]

    for q in sample_queries:
        results = collection.query(query_texts=[q], n_results=2)
        print(f"\nQuery: {q!r}")
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            print(f"  [dist={dist:.4f}] {meta['title']}")


def check_llm() -> None:
    print("\n== 3. LLM round-trip ==")
    from app.core.llm import get_llm

    try:
        llm = get_llm()
    except RuntimeError as e:
        print(f"SKIPPED — {e}")
        return

    response = llm.invoke("Reply with exactly the word: OK")
    print("LLM responded:", response.content)


if __name__ == "__main__":
    check_ticket_validation()
    check_retrieval()
    check_llm()
