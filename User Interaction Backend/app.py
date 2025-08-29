from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from intent_classifier import classify_intent
from tools.sql_tool import execute_sql_for_intent
from tools.context_joiner import semantic_search
from tools.candidate_scope import resolve_candidate_scope
from llm_router import generate_answer, stream_llm_response, build_prompt


import uvicorn

app = FastAPI()

# ========================
# CORS Configuration
# ========================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ========================
# Request Schema
# ========================
class QueryRequest(BaseModel):
    query: str
    selected_emails: Optional[List[str]] = None

# ========================
# Chat Endpoint (Non-Streaming)
# ========================
@app.post("/chat")
def chat(req: QueryRequest):
    query = req.query
    selected_emails = req.selected_emails

    # print(f"[DEBUG] Incoming /chat request: query='{query}', selected_emails={selected_emails}", flush=True)


    # print(f"[DEBUG] Loaded memory history: {chat_history}", flush=True)

    candidate_emails = resolve_candidate_scope(query, selected_emails)
    print(f"[DEBUG] Resolved candidate scope: {candidate_emails}", flush=True)

    intent = classify_intent(query)
    print(f"[DEBUG] Classified intent: {intent}", flush=True)

    sql_result = None
    semantic_chunks = []

    if intent in ("sql", "hybrid"):
        try:
            sql_result = execute_sql_for_intent(query, candidate_emails)
            print(f"[DEBUG] SQL Tool result: {sql_result}", flush=True)
        except Exception as e:
            print(f"[SQL TOOL ERROR] {e}", flush=True)

    if intent in ("semantic", "hybrid"):
        try:
            semantic_chunks = semantic_search(query, candidate_emails=candidate_emails)
            print(f"[DEBUG] Semantic Search result: {semantic_chunks}", flush=True)
        except Exception as e:
            print(f"[SEMANTIC TOOL ERROR] {e}", flush=True)

    answer = generate_answer(query, sql_result, semantic_chunks, candidate_emails)
    print(f"[DEBUG] LLM final answer: {answer}", flush=True)


    print("[DEBUG] Answer saved to memory ✅", flush=True)

    return {"answer": answer}

# ========================
# Chat Endpoint (Streaming)
# ========================
@app.post("/chat-stream")
def chat_stream(req: QueryRequest):
    query = req.query
    selected_emails = req.selected_emails

    print(f"[DEBUG] Incoming /chat-stream request: query='{query}', selected_emails={selected_emails}", flush=True)



    candidate_emails = resolve_candidate_scope(query, selected_emails)
    print(f"[DEBUG] Resolved candidate scope: {candidate_emails}", flush=True)

    intent = classify_intent(query)
    print(f"[DEBUG] Classified intent: {intent}", flush=True)

    sql_result = None
    semantic_chunks = []

    if intent in ("sql", "hybrid"):
        try:
            sql_result = execute_sql_for_intent(query, candidate_emails)
            print(f"[DEBUG] SQL Tool result: {sql_result}", flush=True)
        except Exception as e:
            print(f"[SQL TOOL ERROR] {e}", flush=True)

    if intent in ("semantic", "hybrid"):
        try:
            semantic_chunks = semantic_search(query, candidate_emails=candidate_emails)
            print(f"[DEBUG] Semantic Search result: {semantic_chunks}", flush=True)
        except Exception as e:
            print(f"[SEMANTIC TOOL ERROR] {e}", flush=True)

    prompt = build_prompt(query, sql_result, semantic_chunks, candidate_emails)
    print(f"[DEBUG] Built LLM prompt (length={len(prompt)} chars)", flush=True)

    # ✅ Wrapper to capture and log the streamed response
    def response_stream():
        full_response = ""
        print("[DEBUG] Starting LLM streaming response...", flush=True)
        for chunk in stream_llm_response(prompt):
            full_response += chunk
            yield chunk

    return StreamingResponse(response_stream(), media_type="text/plain")

# ========================
# Health Check
# ========================
@app.get("/healthz")
def health():
    print("[DEBUG] Health check ping received", flush=True)
    return {"status": "ok"}

# ========================
# Run Server
# ========================
if __name__ == "__main__":
    print("[DEBUG] Starting FastAPI server on port 8000 🚀", flush=True)
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
