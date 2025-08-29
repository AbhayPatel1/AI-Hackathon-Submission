from typing import Union, Dict
from collections import defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model globally to avoid reloading each time
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def compute_job_score(sb, candidate_id: str, job_desc: str) -> float:
    """
    Fetches raw section texts from DB using candidate_id, computes weighted section-wise
    BERT-based similarity score against job description. Returns a percentage score (0-100).
    """
    if not candidate_id or not job_desc:
        return 0.0

    sections = (
        sb.table("raw_section_text")
        .select("section_type", "text")
        .eq("candidate_id", candidate_id)
        .execute()
        .data
    )

    section_texts = defaultdict(str)
    for sec in sections:
        section_type = sec.get("section_type") or "other"
        section_texts[section_type.lower()] += " " + (sec.get("text") or "")

    section_weights = {
        "experience": 0.5,
        "skills": 0.3,
        "education": 0.2
    }

    total_score = 0.0
    total_weight = 0.0
    for section_type, text in section_texts.items():
        if not text.strip():
            continue
        weight = section_weights.get(section_type, 0.1)
        resume_emb = model.encode([text], convert_to_tensor=False)
        jd_emb = model.encode([job_desc], convert_to_tensor=False)
        sec_score = cosine_similarity(resume_emb, jd_emb)[0][0]
        total_score += weight * sec_score
        total_weight += weight

    if total_weight == 0:
        return 0.0

    return round((total_score / total_weight) * 100, 2)