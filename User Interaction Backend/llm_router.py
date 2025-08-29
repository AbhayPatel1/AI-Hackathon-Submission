import os
from typing import List, Dict, Optional, Any
from openai import OpenAI

GPT_MODEL = os.getenv("GPT_MODEL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

def format_sql_rows(rows: List[Dict]) -> str:
    print(f"[DEBUG] Formatting SQL rows. Total rows: {len(rows) if rows else 0}")
    if not rows:
        return ""
    lines = []
    for idx, row in enumerate(rows, start=1):
        filtered = {k: v for k, v in row.items() if k not in ("candidate_id",)}
        print(f"[DEBUG] SQL Row {idx}: {filtered}")
        line = " • " + ", ".join(f"{k}: {v}" for k, v in filtered.items())
        lines.append(line)
    return "\n".join(lines)

def format_chunks(chunks: List[Dict[str, Any]]) -> str:
    print(f"[DEBUG] Formatting semantic chunks. Total chunks: {len(chunks) if chunks else 0}")
    if not chunks:
        return ""
    lines = []
    for idx, c in enumerate(chunks, start=1):
        meta = c.get("metadata", {})
        prefix = f"({meta.get('candidate_name', 'Unknown')})"
        print(f"[DEBUG] Chunk {idx}: candidate={meta.get('candidate_name')} doc_snippet={c['document'][:80]}...")
        lines.append(f" • {prefix} {c['document']}")
    return "\n".join(lines)

def build_prompt(
    query: str,
    sql_rows: Optional[List[dict]] = None,
    semantic_chunks: Optional[List[Dict]] = None,
    candidate_scope: Optional[List[str]] = None,
) -> str:
    print(f"[DEBUG] Building prompt for query: {query}")
    print(f"[DEBUG] candidate_scope: {candidate_scope}")

    sql_context = f"Here is what we found from the SQL database:\n{format_sql_rows(sql_rows)}" if sql_rows else ""
    vector_context = f"Here are resume chunks retrieved via semantic search:\n{format_chunks(semantic_chunks)}" if semantic_chunks else ""

    system = [
        "You are an intelligent assistant for HR recruiters.",
        "Always give direct and helpful answers based on candidate data.",
        "Never ask clarifying questions. If information is insufficient, state so plainly.",
        "Prefer concise, direct answers grounded ONLY in the provided SQL results and resume chunks.",
    ]

    prompt = "\n".join([
        "\n".join(system),
        f"User question: {query}",
        f"\nSQL results:\n{sql_context}" if sql_context else "",
        f"\nResume data:\n{vector_context}" if vector_context else ""
    ])

    print(f"[DEBUG] Final prompt length: {len(prompt)} characters")
    return prompt

def call_llm(prompt: str) -> str:
    print("[DEBUG] Calling LLM with non-stream mode")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert recruiter assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
    )
    answer = response.choices[0].message.content.strip()
    print(f"[DEBUG] LLM response received (length={len(answer)} chars)")
    return answer

def stream_llm_response(prompt: str):
    print("[DEBUG] Calling LLM with stream mode")
    response = client.chat.completions.create(
        model=GPT_MODEL,
        messages=[
            {"role": "system", "content": "You are an expert recruiter assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.4,
        stream=True,
    )

    full_answer = ""
    for idx, chunk in enumerate(response, start=1):
        delta = chunk.choices[0].delta
        if delta and delta.content:
            content = delta.content
            full_answer += content
            yield content
    print("[DEBUG] Full streamed answer length:", len(full_answer))

def generate_answer(
    query: str,
    sql_rows: Optional[List[dict]] = None,
    semantic_chunks: Optional[List[Dict]] = None,
    candidate_scope: Optional[List[str]] = None,
) -> str:
    print(f"[DEBUG] Generating answer for query: {query}")
    if not sql_rows and not semantic_chunks:
        print("[DEBUG] No SQL or semantic chunks provided -> insufficient data")
        return "Sorry, I couldn’t find enough information to answer that."

    prompt = build_prompt(query, sql_rows, semantic_chunks, candidate_scope)
    return call_llm(prompt)