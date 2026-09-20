"""
One-off / idempotent script to (re)build the local ChromaDB collection
that backs the Retrieval Agent.

Run it directly:
    python -m app.db.init_chroma

It uses Chroma's bundled default embedding function (a local
sentence-transformers model, all-MiniLM-L6-v2) so this step needs no
API key and works fully offline. The runtime retrieval node built in
a later step will query this same persisted collection.
"""

import chromadb

from app.core.config import CHROMA_PERSIST_DIR, IT_POLICY_COLLECTION

# --------------------------------------------------------------------------
# Mock IT policy knowledge base.
# In a real system these would be chunked from actual internal wiki pages /
# runbooks. Keeping them short and self-contained here so retrieval quality
# is easy to eyeball during development.
# --------------------------------------------------------------------------
MOCK_POLICIES = [
    {
        "id": "policy-vpn-reset",
        "title": "VPN Connectivity & Credential Reset",
        "text": (
            "If a user cannot connect to the corporate VPN, first confirm their "
            "account is active in the IAM console. Most VPN failures are caused "
            "by an expired VPN client certificate or an expired password. To fix: "
            "(1) ask the user to run the 'VPN Repair Tool' from the company "
            "portal, which re-issues the client certificate automatically; "
            "(2) if that fails, reset the user's directory password via the "
            "Admin Console and instruct them to reconnect using the new "
            "credentials; (3) VPN certificates auto-expire after 90 days of "
            "inactivity, so if the user has been on leave, a fresh certificate "
            "issuance is the most common fix."
        ),
    },
    {
        "id": "policy-password-reset",
        "title": "Password Reset Policy",
        "text": (
            "Employees who are locked out of their account after 5 failed login "
            "attempts must be verified via their registered secondary email or "
            "manager confirmation before a password reset is issued. Passwords "
            "must be at least 12 characters, include one number and one special "
            "character, and cannot reuse any of the last 5 passwords. Self-service "
            "password resets are available through the 'Forgot Password' portal "
            "for accounts with MFA enabled; accounts without MFA require a "
            "help-desk-assisted reset for security reasons."
        ),
    },
    {
        "id": "policy-printer-network",
        "title": "Network Printer Connectivity Troubleshooting",
        "text": (
            "For 'printer not found' or 'printer offline' tickets, first confirm "
            "the printer's IP has not changed (DHCP leases are renewed weekly on "
            "the office network). Steps: (1) have the user remove and re-add the "
            "printer using the current IP from the Printer Directory page; "
            "(2) confirm the print spooler service is running on the user's "
            "machine; (3) if multiple users on the same floor report the same "
            "printer as offline, escalate to Network Operations as this "
            "indicates a switch or printer hardware fault rather than a "
            "per-user issue."
        ),
    },
    {
        "id": "policy-software-license",
        "title": "Software License Request Procedure",
        "text": (
            "Requests for paid software (e.g. Adobe Creative Cloud, JetBrains, "
            "specialized analytics tools) require manager approval attached to "
            "the ticket before IT can provision a license. Standard developer "
            "tools (IDEs on the pre-approved list, Git clients) can be "
            "self-installed via the Software Center without approval. License "
            "requests missing manager sign-off should be placed in a "
            "'pending approval' state and the requester notified, not silently "
            "closed."
        ),
    },
    {
        "id": "policy-hardware-replacement",
        "title": "Laptop Hardware Fault & Replacement Policy",
        "text": (
            "For hardware faults (battery not charging, screen flicker, "
            "keyboard failure) on company laptops still within the 3-year "
            "warranty window, the device must be sent to the IT hardware desk "
            "for diagnosis before a replacement is approved; loaner laptops are "
            "available for stays longer than 1 business day. Devices out of "
            "warranty with a repair cost estimate exceeding 40% of replacement "
            "cost should be escalated to the Asset Management team for a "
            "replace-vs-repair decision rather than repaired directly."
        ),
    },
]


def build_collection(reset: bool = True) -> None:
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    if reset:
        try:
            client.delete_collection(IT_POLICY_COLLECTION)
        except Exception:
            pass  # collection didn't exist yet — fine

    # No embedding_function passed explicitly -> Chroma uses its bundled
    # default (all-MiniLM-L6-v2 via sentence-transformers), applied
    # consistently at both write time and query time.
    collection = client.get_or_create_collection(name=IT_POLICY_COLLECTION)

    collection.add(
        ids=[p["id"] for p in MOCK_POLICIES],
        documents=[p["text"] for p in MOCK_POLICIES],
        metadatas=[{"title": p["title"]} for p in MOCK_POLICIES],
    )

    print(
        f"Loaded {collection.count()} policy documents into "
        f"collection '{IT_POLICY_COLLECTION}' at {CHROMA_PERSIST_DIR}"
    )


if __name__ == "__main__":
    build_collection()
