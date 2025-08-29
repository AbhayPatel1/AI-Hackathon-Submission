from typing import List
from tools.vector_tool import semantic_search

def build_joined_context(query: str, candidate_emails: List[str], top_k: int = 5) -> str:
    context_blocks = []

    for email in candidate_emails:
        chunks = semantic_search(query=query, scope_emails=[email], top_k=top_k)
        if not chunks:
            continue

        block = f"=== Candidate: {email} ===\n"
        for c in chunks:
            block += f"- {c['document']}\n"
        context_blocks.append(block)

    if not context_blocks:
        return "No relevant candidate context found."

    return "\n\n".join(context_blocks)