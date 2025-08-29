import os
from typing import List, Dict, Optional, Tuple, Any
from collections import defaultdict
import math, re, json

from openai import OpenAI
import chromadb
from dotenv import load_dotenv

# Fuzzy matching (you already have fuzzywuzzy + python-Levenshtein in your reqs)
from fuzzywuzzy import process as fw_process, fuzz as fw_fuzz

load_dotenv()

# =========================
# Config & Clients
# =========================
OPENAI_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")   # must match index dims
RERANK_EMBED_MODEL = os.getenv("RERANK_EMBED_MODEL", "text-embedding-3-large")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma_index")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "resumes")

# RAG Fusion knobs
FUSION_NUM_SUBQUERIES = int(os.getenv("FUSION_NUM_SUBQUERIES", "6"))
RETRIEVAL_TOPK_PER_QUERY = int(os.getenv("RETRIEVAL_TOPK_PER_QUERY", "30"))
RRF_K = int(os.getenv("RRF_K", "60"))
FUSED_CANDIDATE_POOL = int(os.getenv("FUSED_CANDIDATE_POOL", "30"))
THRESHOLD_DISTANCE = float(os.getenv("THRESHOLD_DISTANCE", "0.9"))

# Entity extraction knobs
FUZZY_LIMIT = int(os.getenv("ENTITY_FUZZY_LIMIT", "8"))          # max candidates to consider
FUZZY_SKILL_THRESHOLD = int(os.getenv("SKILL_FUZZY_THRESHOLD", "87"))
FUZZY_ROLE_THRESHOLD = int(os.getenv("ROLE_FUZZY_THRESHOLD", "90"))
FUZZY_DEGREE_THRESHOLD = int(os.getenv("DEGREE_FUZZY_THRESHOLD", "92"))

# Debug toggle
DEBUG = os.getenv("RAG_FUSION_DEBUG", "1") == "1"

openai_client = OpenAI(api_key=OPENAI_API_KEY)
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = chroma_client.get_or_create_collection(name=CHROMA_COLLECTION)

# =========================
# Helper: embeddings
# =========================
def embed_text(text: str, model: Optional[str] = None) -> List[float]:
    mdl = model or OPENAI_MODEL
    resp = openai_client.embeddings.create(model=mdl, input=[text])
    return resp.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0; na = 0.0; nb = 0.0
    for x, y in zip(a, b):
        dot += x * y; na += x * x; nb += y * y
    if na == 0 or nb == 0: return 0.0
    return dot / (math.sqrt(na) * math.sqrt(nb))

# =========================
# Rule/Fuzzy vocab (extendable)
# =========================
# Base seed; extend via ENV (comma-separated) if you like
BASE_SKILLS = {
    "python","django","flask","fastapi","react","node","javascript","typescript",
    "aws","gcp","azure","docker","kubernetes","k8s","grpc","redis","postgres","mysql",
    "spark","hadoop","snowflake","dbt",
    "pytorch","tensorflow","keras","scikit-learn","opencv","cuda","tensorrt","triton",
    "yolov7","yolov5","zed","ros","ros2","opencv","tensorRT","triton inference server"
}
BASE_ROLES = {
    "backend engineer","backend developer","frontend engineer","frontend developer",
    "full stack","full stack developer",
    "data engineer","ml engineer","machine learning engineer","data scientist",
    "ios developer","android developer","devops","sre","site reliability engineer"
}
BASE_DEGREES = {
    "btech","be","b.e.","mtech","m.tech","mca","bca","bsc","msc","mba","phd","ph.d."
}
SECTION_HINTS = {
    "experience": ("experience","worked at","role","roles","position","positions"),
    "project": ("project","projects"),
    "education": ("education","degree","college","university"),
    "skills": ("skills","tech stack","stack"),
    "achievement": ("achievement","awards","accomplishment"),
}

def _env_set(name: str) -> set:
    raw = os.getenv(name, "")
    if not raw.strip(): return set()
    return {x.strip().lower() for x in raw.split(",") if x.strip()}

SKILL_VOCAB = {s.lower() for s in BASE_SKILLS} | _env_set("EXTRA_SKILLS")
ROLE_VOCAB  = {r.lower() for r in BASE_ROLES}  | _env_set("EXTRA_ROLES")
DEG_VOCAB   = {d.lower() for d in BASE_DEGREES}| _env_set("EXTRA_DEGREES")

YEARS_RE = re.compile(r'(\d+)\s*(?:\+|plus)?\s*(?:years|yrs|yr)\b', re.I)

def _normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.lower()).strip()

# =========================
# Rule + Fuzzy entity extractor (NO LLM)
# =========================
def extract_query_entities(q: str) -> Dict[str, Any]:
    qn = _normalize(q)

    # Direct substring hits first (cheap)
    skills_direct = {s for s in SKILL_VOCAB if s in qn}
    roles_direct  = {r for r in ROLE_VOCAB if r in qn}
    degrees_direct= {d for d in DEG_VOCAB if d in qn}

    # Fuzzy supplement (only if few direct hits)
    skills = set(skills_direct)
    roles  = set(roles_direct)
    degrees= set(degrees_direct)

    # Prepare a tokenizable "needle" (limit length)
    q_needle = qn[:200]

    if len(skills) < 3 and SKILL_VOCAB:
        cand = fw_process.extract(q_needle, list(SKILL_VOCAB), scorer=fw_fuzz.WRatio, limit=FUZZY_LIMIT)
        for val, score in cand:
            if score >= FUZZY_SKILL_THRESHOLD:
                skills.add(val)
    if len(roles) < 2 and ROLE_VOCAB:
        cand = fw_process.extract(q_needle, list(ROLE_VOCAB), scorer=fw_fuzz.WRatio, limit=FUZZY_LIMIT)
        for val, score in cand:
            if score >= FUZZY_ROLE_THRESHOLD:
                roles.add(val)
    if len(degrees) < 1 and DEG_VOCAB:
        cand = fw_process.extract(q_needle, list(DEG_VOCAB), scorer=fw_fuzz.WRatio, limit=FUZZY_LIMIT)
        for val, score in cand:
            if score >= FUZZY_DEGREE_THRESHOLD:
                degrees.add(val)

    # Sections
    sections = set()
    for sec, keys in SECTION_HINTS.items():
        if any(k in qn for k in keys):
            sections.add(sec)

    # Years
    min_years = None
    m = YEARS_RE.search(qn)
    if m:
        try:
            min_years = int(m.group(1))
        except:  # noqa
            min_years = None

    entities = {
        "skills": sorted(skills),
        "roles": sorted(roles),
        "degrees": sorted(degrees),
        "sections": sorted(sections),
        "min_years": min_years,
    }
    if DEBUG:
        print(f"[META] Extracted entities (rule+fuzzy): {entities}")
    return entities

# =========================
# Build Chroma where-filter + post-filter
# =========================
def build_metadata_filter(candidate_emails: Optional[List[Dict]], entities: Dict[str, Any]) -> Optional[Dict]:
    """
    Coarse filter using only fields/operators Chroma handles well.
    """
    where = {}

    # Scope by candidate emails (unchanged)
    if candidate_emails:
        clean = [e["email"] for e in candidate_emails if isinstance(e, dict) and e.get("email")]
        if clean:
            where["primary_email"] = {"$in": clean}

    # Sections: equality is supported
    if entities.get("sections"):
        where["section"] = {"$in": entities["sections"]} if len(entities["sections"]) > 1 else {"$eq": entities["sections"][0]}

    # Roles (title) — coarse, because titles are free text; still helpful as a hint
    if entities.get("roles"):
        where["title"] = {"$in": entities["roles"]}

    # Degree level
    if entities.get("degrees"):
        where["degree_level"] = {"$in": entities["degrees"]}

    if where:
        if DEBUG:
            print(f"[META] Chroma where-filter (coarse): {where}")
        return where
    return None

def _parse_maybe_json(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return v
    return v

def passes_post_filters(meta: Dict[str, Any], entities: Dict[str, Any]) -> bool:
    """
    Tight checks impossible/awkward in Chroma 'where' (lists stored as JSON strings).
    - Skills: check in skills_mentioned / tech_stack
    - (Optionally) Years of experience: if you later add total_years to metadata.
    """
    if not entities:
        return True

    want_skills = set(_normalize(s) for s in (entities.get("skills") or []))
    if want_skills:
        skills_mentioned = _parse_maybe_json(meta.get("skills_mentioned"))
        tech_stack = _parse_maybe_json(meta.get("tech_stack"))
        found = set()

        def _collect(v):
            if isinstance(v, list):
                for x in v:
                    if isinstance(x, str):
                        found.add(_normalize(x))
            elif isinstance(v, str):
                for x in re.split(r'[,/|;]', v):
                    if x.strip():
                        found.add(_normalize(x.strip()))

        _collect(skills_mentioned)
        _collect(tech_stack)

        if want_skills and not any(ws in found for ws in want_skills):
            return False

    # Years: if you maintain meta["total_years"], enforce it here:
    # if entities.get("min_years") is not None:
    #     if float(meta.get("total_years", 0)) < entities["min_years"]:
    #         return False

    return True

# =========================
# Subquery generation (RAG Fusion step 1)
# =========================
def generate_subqueries(user_query: str, k: int = FUSION_NUM_SUBQUERIES) -> List[str]:
    # LLM-free variant (deterministic): synthesize simple paraphrases
    q = user_query.strip()
    seeds = [
        q,
        q.replace("developer", "engineer"),
        q.replace("engineer", "developer"),
        q + " projects",
        q + " experience",
        q + " resume",
        q + " skills",
        q.replace(" and ", " & "),
        q.replace(" with ", " using "),
    ]
    # dedupe and cap at k unique (plus original already included)
    uniq, seen = [], set()
    for s in seeds:
        s = s.strip()
        if not s: continue
        key = s.lower()
        if key not in seen:
            uniq.append(s); seen.add(key)
        if len(uniq) >= k: break

    if DEBUG:
        print("[RAG-FUSION] Generated subqueries (deterministic):")
        for i, qu in enumerate(uniq, 1):
            print(f"  {i:02d}. {qu}")
    # include original once at the end to mimic the diagram behavior
    if uniq[-1].lower() != q.lower():
        uniq.append(q)
    return uniq

# =========================
# Retrieval per subquery
# =========================
def _build_scope_filter(candidate_emails: Optional[List[Dict]]) -> Optional[Dict]:
    if not candidate_emails:
        return None
    clean_emails = [e["email"] for e in candidate_emails if isinstance(e, dict) and e.get("email")]
    if clean_emails:
        return {"primary_email": {"$in": clean_emails}}
    return None

def retrieve_for_query(q: str, where_filter: Optional[Dict], n_results: int) -> List[Tuple[str, str, Dict, float]]:
    q_emb = embed_text(q, model=OPENAI_MODEL)
    res = collection.query(
        query_embeddings=[q_emb],
        n_results=n_results,
        where=where_filter
    )
    ids = res.get("ids", [[]])[0]
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    ranked = [(ids[i], docs[i], metas[i], dists[i]) for i in range(len(ids))]
    if DEBUG:
        print(f"[RAG-FUSION] Retrieved {len(ranked)} hits for subquery: {q!r}")
    return ranked

# =========================
# RRF fusion
# =========================
def rrf_fuse(rank_lists: List[List[Tuple[str, str, Dict, float]]], k: int = RRF_K) -> Dict[str, Dict]:
    fused = {}
    for ranked in rank_lists:
        for rank_idx, (cid, doc, meta, dist) in enumerate(ranked, start=1):
            inc = 1.0 / (k + rank_idx)
            node = fused.get(cid)
            if node is None:
                fused[cid] = {"score": inc, "best": (cid, doc, meta, dist), "first_rank": rank_idx}
            else:
                node["score"] += inc
                if dist < node["best"][3]:
                    node["best"] = (cid, doc, meta, dist)
                if rank_idx < node["first_rank"]:
                    node["first_rank"] = rank_idx
    if DEBUG:
        print(f"[RAG-FUSION] Fused unique candidates: {len(fused)}")
    return fused

# =========================
# Re-ranking (semantic precision bump)
# =========================
def rerank_with_large_embeddings(query: str, pool: List[Tuple[str, str, Dict, float]]) -> List[Tuple[str, str, Dict, float, float]]:
    if not pool:
        return []
    if DEBUG:
        print(f"[RAG-FUSION] Re-ranking {len(pool)} candidates with {RERANK_EMBED_MODEL} ...")
    q_vec = embed_text(query, model=RERANK_EMBED_MODEL)
    scored = []
    for cid, doc, meta, dist in pool:
        try:
            d_vec = embed_text(doc[:8000], model=RERANK_EMBED_MODEL)
            score = cosine_similarity(q_vec, d_vec)
        except Exception as e:
            if DEBUG:
                print(f"[RAG-FUSION][WARN] Re-rank embedding failed for {cid}: {e}. Using fallback score 0.")
            score = 0.0
        scored.append((cid, doc, meta, dist, score))
    scored.sort(key=lambda x: x[4], reverse=True)
    if DEBUG and scored:
        print("[RAG-FUSION] Top 5 after re-rank (cid, score, orig_dist):")
        for s in scored[:5]:
            print(f"  {s[0]}  score={s[4]:.4f}  dist={s[3]:.4f}")
    return scored

# =========================
# Public API (unchanged): semantic_search
# =========================
def semantic_search(
    query: str,
    top_k: int = 5,
    candidate_emails: Optional[List[str]] = None
) -> List[Dict]:
    """
    RAG Fusion + Re-ranking + metadata-aware filtering (rule/fuzzy entities, no LLM).
    """
    if DEBUG:
        print("\n====== SEMANTIC SEARCH (RAG FUSION + META RULE/FUZZY) ======")
        print(f"Query: {query!r}")
        print(f"Scope candidate_emails: {candidate_emails}")

    # 0) Entities from rule/fuzzy
    entities = extract_query_entities(query)

    # 1) Build scope + coarse metadata where-filter
    scope_filter = _build_scope_filter(candidate_emails)
    meta_filter = build_metadata_filter(candidate_emails=None, entities=entities)  # emails already in scope_filter

    where_filter = {}
    if scope_filter:
        where_filter.update(scope_filter)
    if meta_filter:
        where_filter.update(meta_filter)
    if not where_filter:
        where_filter = None

    if DEBUG:
        print(f"[META] Final where_filter sent to Chroma: {where_filter}")

    # 2) Generate diverse subqueries (deterministic; no LLM)
    subqueries = generate_subqueries(query, k=FUSION_NUM_SUBQUERIES)

    # 3) Retrieval per subquery
    ranked_lists: List[List[Tuple[str, str, Dict, float]]] = []
    for sq in subqueries:
        ranked_lists.append(retrieve_for_query(sq, where_filter=where_filter, n_results=RETRIEVAL_TOPK_PER_QUERY))

    # 4) RRF fusion
    fused = rrf_fuse(ranked_lists, k=RRF_K)
    pool_sorted = sorted(fused.values(), key=lambda x: (x["score"], -x["first_rank"]), reverse=True)
    pool = [v["best"] for v in pool_sorted[:FUSED_CANDIDATE_POOL]]

    if DEBUG:
        print(f"[RAG-FUSION] Candidate pool before post-filter: {len(pool)}")

    # 5) Post-filter using rich metadata (skills_mentioned, tech_stack)
    if entities and any(entities.values()):
        filtered_pool = []
        for cid, doc, meta, dist in pool:
            try:
                if passes_post_filters(meta or {}, entities):
                    filtered_pool.append((cid, doc, meta, dist))
            except Exception as e:
                if DEBUG:
                    print(f"[META][WARN] Post-filter failed for {cid}: {e}. Keeping item.")
                filtered_pool.append((cid, doc, meta, dist))
        if DEBUG:
            print(f"[META] Pool after post-filter: {len(filtered_pool)} (was {len(pool)})")
        pool = filtered_pool

    if DEBUG:
        print(f"[RAG-FUSION] Candidate pool size before re-rank: {len(pool)}")

    # 6) Re-rank with stronger embeddings
    reranked = rerank_with_large_embeddings(query, pool)

    # 7) Select Top-K final (preserve original schema)
    final = []
    for cid, doc, meta, orig_dist, rr_score in reranked[:top_k]:
        if orig_dist <= THRESHOLD_DISTANCE:
            final.append({"id": cid, "document": doc, "metadata": meta, "distance": orig_dist})

    # Fallback if threshold filters everything out
    if not final:
        if DEBUG:
            print("[RAG-FUSION] No items passed threshold after re-rank. Falling back to fused by distance.")
        backup = sorted(pool, key=lambda x: x[3])[:top_k]
        for cid, doc, meta, dist in backup:
            final.append({"id": cid, "document": doc, "metadata": meta, "distance": dist})

    if DEBUG:
        print(f"[RAG-FUSION] Final returned items: {len(final)}")
        for i, it in enumerate(final, 1):
            name = ""
            if isinstance(it.get("metadata"), dict):
                name = it["metadata"].get("candidate_name") or it["metadata"].get("title") or ""
            print(f"  {i:02d}. id={it['id']} dist={it['distance']:.4f} name/title={name}")

    return final