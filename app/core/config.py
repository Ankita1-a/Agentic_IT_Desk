"""
Shared configuration constants for the IT Incident Triage Engine.
Kept in one place so the ingestion script (init_chroma.py) and the
runtime retrieval node (built in a later step) always point at the
same ChromaDB collection.
"""

from pathlib import Path

# Persisted on disk so the graph can query it without re-embedding
# every time the FastAPI process restarts.
CHROMA_PERSIST_DIR = str(Path(__file__).resolve().parent.parent.parent / "chroma_db")

IT_POLICY_COLLECTION = "it_policies"
