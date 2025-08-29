# vector_tool.py  — Fusion-RAG (advanced) with GPT cross-rerank + MMR diversity
import os
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import math
import re

from openai import OpenAI
import chromadb

# -------------------------
# Env & Clients
# -------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = OpenAI(api_key=OPENAI_API_KEY)

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_index")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "resumes")
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

# Embedding models
EMBED_QDOC = os.getenv("EMBED_MODEL", "text-embedding-3-small")      # retrieval
EMBED_RERANK = os.getenv("RERANK_EMBED_MODEL", "text-embedding-3-large")  # stronger embeddings (if using embed-rerank)

# LLM models
FUSION_LLM = os.getenv("FUSION_LLM", "gpt-4o-mini")  # for subqueries, HyDE
GPT_RERANK_MODEL = os.getenv("GPT_RERANK_MODEL", "gpt-4o-mini")  # for GPT cross-reranking

# RAG Fusion knobs
FUSION_NUM_SUBQUERIES = int(os.getenv("FUSION_NUM_SUBQUERIES", "6"))   # number *excluding* original and HyDE is 1 slot
RETRIEVAL_TOPK_PER_QUERY = int(os.getenv("RETRIEVAL_TOPK_PER_QUERY", "30"))
RRF_K = int(os.getenv("RRF_K", "60"))
FUSED_CANDIDATE_POOL = int(os.getenv("FUSED_CANDIDATE_POOL", "50"))
ORIGINAL_QUERY_WEIGHT = float(os.getenv("ORIGINAL_QUERY_WEIGHT", "1.7"))  # >1 gives a slight boost to original
RAG_DEBUG = os.getenv("RAG_FUSION_DEBUG", "1") == "1"

# Rerank strategy: "gpt_cross" | "embed" (fast) 
RERANK_STRATEGY = os.getenv("RERANK_STRATEGY", "gpt_cross").lower()

# Distance threshold from first-pass retrieval (set high; final filtering is by rerank score)
THRESHOLD_DISTANCE = float(os.getenv("THRESHOLD_DISTANCE", "0.95"))

# MMR diversity (applied after rerank, optional)
USE_MMR = os.getenv("USE_MMR", "1") == "1"
MMR_LAMBDA = float(os.getenv("MMR_LAMBDA", "0.65"))  # 0 = only diversity, 1 = only relevance
MMR_MAX = int(os.getenv("MMR_MAX", "200"))  # cap items considered for mmr

# -------------------------
# Utilities
# -------------------------
def _log(msg: str):
    if RAG_DEBUG:
        print(msg, flush=True)

def embed(text: str, model: Optional[str] = None) -> List[float]:
    mdl = model or EMBED_QDOC
    resp = openai_client.embeddings.create(model=mdl, input=[text])
    return resp.data[0].embedding

def cosine_sim(a: List[float], b: List[float]) -> float:
    dot = 0.0; na = 0.0; nb = 0.0
    for x, y in zip(a, b):
        dot += x*y; na += x*x; nb += y*y
    if na == 0 or nb == 0:
        return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

def _build_scope_filter(candidate_emails: Optional[List[str]]) -> Optional[Dict]:
    if not candidate_emails:
        return None
    clean = [e for e in candidate_emails if isinstance(e, str) and e]
    return {"primary_email": {"$in": clean}} if clean else None

# -------------------------
# Advanced Subquery Generation (Fusion Step 1)
# Buckets: 2 keyword, 2 facet, 1 step-back, 1 HyDE, + original
# -------------------------
SUBQUERY_PROMPT = """You are generating retrieval subqueries in buckets for a vector database.

User query: "{q}"

Produce exactly 5 LINES (no numbering). Each line should be one of:
1) KEYWORD variant (short, keyword-heavy, no stopwords)
2) KEYWORD variant (short, keyword-heavy, no stopwords)
3) FACET EXPANSION (add likely attributes/entities/time ranges)
4) FACET EXPANSION (different angle than #3)
5) STEP-BACK GENERALIZATION (zoom out one level to the parent concept)

Return only the 5 lines. No extra text.
"""

HYDE_PROMPT = """Write a concise 2–3 sentence hypothetical answer for retrieval (HyDE) that would ideally satisfy:

User query: "{q}"

Be specific, factual in tone, and include likely keywords/phrases that might appear in a relevant resume/project description. Return only the 2–3 sentences.
"""

def generate_subqueries_advanced(user_query: str) -> Tuple[List[str], Optional[str]]:
    """
    Returns:
      - subqueries (list of 5 strings: 2 keyword, 2 facet, 1 step-back)
      - hyde (single short paragraph string) or None on failure
    """
    try:
        comp = openai_client.chat.completions.create(
            model=FUSION_LLM,
            messages=[
                {"role": "system", "content": "Return only the requested lines. Be concise and diverse."},
                {"role": "user", "content": SUBQUERY_PROMPT.format(q=user_query)},
            ],
            temperature=0.5,
        )
        text = comp.choices[0].message.content.strip()
        lines = [ln.strip(" -•\t") for ln in text.splitlines() if ln.strip()]
        # Keep exactly 5 (allow model drift)
        subqs = lines[:5]
    except Exception as e:
        _log(f"[FUSION][WARN] subquery buckets failed: {e}. Falling back to single query.")
        subqs = []

    # HyDE line
    hyde_txt = None
    try:
        comp_h = openai_client.chat.completions.create(
            model=FUSION_LLM,
            messages=[
                {"role": "system", "content": "Return only a concise paragraph (2–3 sentences)."},
                {"role": "user", "content": HYDE_PROMPT.format(q=user_query)},
            ],
            temperature=0.7,
        )
        hyde_txt = comp_h.choices[0].message.content.strip()
    except Exception as e:
        _log(f"[FUSION][WARN] HyDE generation failed: {e}")

    # Simple dedup by lowercase string
    seen = set()
    uniq = []
    for s in subqs:
        k = s.lower()
        if k not in seen:
            uniq.append(s)
            seen.add(k)

    _log("[FUSION] Subquery buckets:")
    for i, s in enumerate(uniq, 1):
        _log(f"  {i:02d}. {s}")
    if hyde_txt:
        _log(f"[FUSION] HyDE len={len(hyde_txt)} chars")

    return uniq, hyde_txt

# -------------------------
# Retrieval per subquery
# -------------------------
def retrieve_for_query(q: str, where_filter: Optional[Dict], n_results: int) -> List[Tuple[str, str, Dict, float]]:
    q_emb = embed(q, model=EMBED_QDOC)
    res = collection.query(query_embeddings=[q_emb], n_results=n_results, where=where_filter)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]  # cosine distance (0 best; lower is better)
    ranked = [(ids[i], docs[i], metas[i], dists[i]) for i in range(len(ids))]
    _log(f"[RETRIEVE] '{q[:70]}...' -> {len(ranked)} hits")
    return ranked

def retrieve_for_hyde(hyde: str, where_filter: Optional[Dict], n_results: int) -> List[Tuple[str, str, Dict, float]]:
    # HyDE is embedded as-is (paragraph)
    q_emb = embed(hyde, model=EMBED_QDOC)
    res = collection.query(query_embeddings=[q_emb], n_results=n_results, where=where_filter)
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ranked = [(ids[i], docs[i], metas[i], dists[i]) for i in range(len(ids))]
    _log(f"[RETRIEVE][HyDE] -> {len(ranked)} hits")
    return ranked

# -------------------------
# Weighted Reciprocal Rank Fusion (RRF)
# -------------------------
def rrf_fuse_weighted(lists: List[List[Tuple[str, str, Dict, float]]],
                      weights: Optional[List[float]] = None,
                      k: int = RRF_K) -> Dict[str, Dict]:
    fused: Dict[str, Dict[str, Any]] = {}
    if not lists:
        return {}

    if weights is None:
        weights = [1.0] * len(lists)

    for li, ranked in enumerate(lists):
        w = weights[li] if li < len(weights) else 1.0
        for r_idx, (cid, doc, meta, dist) in enumerate(ranked, start=1):
            inc = w * (1.0 / (k + r_idx))
            node = fused.get(cid)
            if node is None:
                fused[cid] = {"score": inc, "best": (cid, doc, meta, dist), "first_rank": r_idx}
            else:
                node["score"] += inc
                if dist < node["best"][3]:
                    node["best"] = (cid, doc, meta, dist)
                if r_idx < node["first_rank"]:
                    node["first_rank"] = r_idx
    _log(f"[RRF] Fused {len(fused)} unique chunks")
    return fused

# -------------------------
# Reranking
#   A) GPT cross-encoder style (reads query + doc together) — higher quality
#   B) Embedding cosine with EMBED_RERANK — faster
# -------------------------
def _gpt_score_pair(query: str, doc: str) -> float:
    """
    Returns a float 0..1 relevance score using GPT (cross-encoder-like).
    We keep the prompt short to minimize cost/latency.
    """
    system = "You are a strict relevance grader. Output only a number between 0 and 1."
    user = f"""Query: {query}

Document:
\"\"\"{doc[:4000]}\"\"\"

Score how relevant the document is to the query (0=irrelevant, 1=perfect). Output only the number."""
    try:
        resp = openai_client.chat.completions.create(
            model=GPT_RERANK_MODEL,
            messages=[{"role":"system","content":system}, {"role":"user","content":user}],
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip()
        m = re.search(r"([01](?:\.\d+)?)", raw)
        if m:
            return float(m.group(1))
    except Exception as e:
        _log(f"[RERANK][WARN] GPT score failed: {e}")
    return 0.0

def rerank_gpt_cross(query: str, pool: List[Tuple[str,str,Dict,float]], max_items: int = 120) -> List[Tuple[str,str,Dict,float,float]]:
    take = pool[:max_items]
    scored = []
    for (cid, doc, meta, dist) in take:
        s = _gpt_score_pair(query, doc)
        scored.append((cid, doc, meta, dist, s))
    scored.sort(key=lambda x: x[4], reverse=True)
    _log(f"[RERANK] GPT cross ranked {len(scored)} items. Top score={scored[0][4]:.3f}" if scored else "[RERANK] Empty")
    return scored

def rerank_embed_cos(query: str, pool: List[Tuple[str,str,Dict,float]]) -> List[Tuple[str,str,Dict,float,float]]:
    if not pool:
        return []
    _log(f"[RERANK] Embedding rerank with {EMBED_RERANK}")
    qv = embed(query, model=EMBED_RERANK)
    scored = []
    for (cid, doc, meta, dist) in pool:
        try:
            dv = embed(doc[:8000], model=EMBED_RERANK)
            s = cosine_sim(qv, dv)
        except Exception as e:
            _log(f"[RERANK][WARN] embed rerank failed for {cid}: {e}")
            s = 0.0
        scored.append((cid, doc, meta, dist, s))
    scored.sort(key=lambda x: x[4], reverse=True)
    return scored

# -------------------------
# MMR diversity (on reranked list)
# -------------------------
def mmr_select(query: str, items: List[Tuple[str,str,Dict,float,float]], top_k: int) -> List[Tuple[str,str,Dict,float,float]]:
    """
    items: (cid, doc, meta, orig_dist, rerank_score)
    We compute embeddings for selected items for diversity using EMBED_QDOC for speed.
    """
    if not items:
        return []
    k = min(top_k, len(items))
    base = items[:min(MMR_MAX, len(items))]
    qv = embed(query, model=EMBED_QDOC)

    # Pre-embed docs
    dv_cache: List[List[float]] = []
    for (_, doc, _, _, _) in base:
        dv_cache.append(embed(doc[:2000], model=EMBED_QDOC))

    selected = []
    selected_idx = set()

    # Start with the best by rerank score
    best_idx = 0
    best_score = base[0][4]
    selected.append(base[best_idx])
    selected_idx.add(best_idx)

    while len(selected) < k:
        best = None
        best_val = -1e9
        for i in range(len(base)):
            if i in selected_idx:
                continue
            # relevance = normalized rerank score
            rel = base[i][4]
            # diversity: max similarity to already selected
            max_sim = 0.0
            for j in selected_idx:
                sim = cosine_sim(dv_cache[i], dv_cache[j])
                if sim > max_sim:
                    max_sim = sim
            val = MMR_LAMBDA * rel - (1 - MMR_LAMBDA) * max_sim
            if val > best_val:
                best_val = val
                best = i
        if best is None:
            break
        selected.append(base[best])
        selected_idx.add(best)

    return selected

# -------------------------
# Public API
# -------------------------
def semantic_search(
    query: str,
    top_k: int = 5,
    candidate_emails: Optional[List[str]] = None
) -> List[Dict]:
    _log("\n====== SEMANTIC SEARCH (Fusion-RAG Advanced) ======")
    _log(f"Query: {query!r}")
    _log(f"Scope candidate_emails: {candidate_emails}")

    where_filter = _build_scope_filter(candidate_emails)

    # 1) Subqueries: buckets + HyDE + original
    subqs, hyde_txt = generate_subqueries_advanced(query)
    lists = []
    weights = []

    # Retrieve per bucketed subquery
    for s in subqs:
        lists.append(retrieve_for_query(s, where_filter, RETRIEVAL_TOPK_PER_QUERY))
        weights.append(1.0)

    # HyDE retrieval (optional)
    if hyde_txt:
        lists.append(retrieve_for_hyde(hyde_txt, where_filter, RETRIEVAL_TOPK_PER_QUERY))
        weights.append(1.15)  # small boost to HyDE

    # Add original query retrieval, with extra weight
    lists.append(retrieve_for_query(query, where_filter, RETRIEVAL_TOPK_PER_QUERY))
    weights.append(ORIGINAL_QUERY_WEIGHT)

    # 2) RRF fusion (weighted)
    fused = rrf_fuse_weighted(lists, weights=weights, k=RRF_K)

    # 3) Build candidate pool (sorted by fused score desc), keep best-distance doc for each id
    pool_sorted = sorted(fused.values(), key=lambda x: (x["score"], -x["first_rank"]), reverse=True)
    pool = [v["best"] for v in pool_sorted[:FUSED_CANDIDATE_POOL]]
    _log(f"[POOL] Size before rerank: {len(pool)}")

    if not pool:
        _log("[POOL] Empty pool; returning empty result.")
        return []

    # 4) Rerank
    if RERANK_STRATEGY == "gpt_cross":
        reranked = rerank_gpt_cross(query, pool, max_items=min(150, len(pool)))
        # Filter out extremely poor matches
        reranked = [t for t in reranked if t[4] > 0.15]
    else:
        reranked = rerank_embed_cos(query, pool)
        reranked = [t for t in reranked if t[3] <= THRESHOLD_DISTANCE]  # keep decent first-pass distance

    _log(f"[RERANK] Kept {len(reranked)} after filtering")

    if not reranked:
        # Fallback to fused by distance if rerank eliminated all
        _log("[RERANK] Empty after filter; falling back to fused by lowest distance")
        backup = sorted(pool, key=lambda x: x[3])[:top_k]
        return [{"id": cid, "document": doc, "metadata": meta, "distance": dist} for (cid, doc, meta, dist) in backup]

    # 5) MMR diversity (optional)
    final_items = reranked
    if USE_MMR and len(reranked) > top_k:
        final_items = mmr_select(query, reranked, top_k)

    # If not using MMR, just slice
    final_items = final_items[:top_k]

    # 6) Format output (preserve original schema; use original first-pass distance)
    results = []
    for (cid, doc, meta, orig_dist, rr_score) in final_items:
        results.append({
            "id": cid,
            "document": doc,
            "metadata": meta,
            "distance": orig_dist  # keep compatibility with your callers
        })

    _log(f"[FINAL] Returned {len(results)} items")
    for i, it in enumerate(results, 1):
        title = it["metadata"].get("title") if isinstance(it["metadata"], dict) else ""
        _log(f"  {i:02d}. id={it['id']} dist={it['distance']:.4f} title={title}")

    return results