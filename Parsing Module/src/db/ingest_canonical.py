from __future__ import annotations

import argparse, json, os, re
from typing import Any, Dict, List, Optional, Tuple, Set
from datetime import datetime
from tqdm import tqdm
from supabase import Client
from .supa import get_client
from src.utils.bert_score import compute_job_score

SPACE_RE = re.compile(r"\s+")

def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()

def _norm(s: Optional[str]) -> str:
    return SPACE_RE.sub(" ", (s or "").strip())

def _first(xs: Optional[List[Any]]) -> Optional[Any]:
    for x in (xs or []):
        if x:
            return x
    return None

def _name_norm(s: Optional[str]) -> Optional[str]:
    if not s: return None
    return _norm(s).lower()

def _skill_tokens(skills_block: Any) -> List[str]:
    out = []
    for s in (skills_block or []):
        if isinstance(s, dict):
            name = s.get("name")
        else:
            name = s
        name = _name_norm(name)
        if name:
            out.append(name)
    return sorted(list(set(out)))

def upsert_resume_source(sb: Client, candidate: Dict[str, Any]) -> None:
    src_id = candidate.get("source_file_id")
    if not src_id:
        return
    resume_info = candidate.get("resume_source", {})
    row = {
        "source_file_id": src_id,
        "filename": resume_info.get("filename"),
        "bytes": resume_info.get("bytes"),
        "mime_type": resume_info.get("mime_type"),
    }
    sb.table("resume_source").upsert(row, on_conflict="source_file_id").execute()

def upsert_candidate(sb: Client, cand: Dict[str, Any], job_id: Optional[str] = None) -> str:
    emails = cand.get("emails") or []
    primary_email = _first(emails)
    alt_emails = emails[1:] if len(emails) > 1 else []

    location = cand.get("location") or {}
    links = cand.get("links") or {}

    row = {
        "full_name": cand.get("full_name"),
        "primary_email": primary_email,
        "alt_emails": alt_emails or None,
        "primary_phone": _first(cand.get("phones") or []),
        "location_city": location.get("city"),
        "location_state": location.get("state"),
        "location_country": location.get("country"),
        "linkedin_url": links.get("linkedin"),
        "github_url": links.get("github"),
        "portfolio_url": None,
        "resume_text_hash": cand.get("resume_text_hash"),
        "source_file_id": cand.get("source_file_id"),
        "parsed_at": cand.get("parsed_at"),
    }

    # Attach job_id only if provided, else set it to None
    row["job_id"] = str(job_id) if job_id else None

    print(f"[debug] Upserting candidate with job_id={row['job_id']}")
    res = sb.table("candidate").upsert(row, on_conflict="resume_text_hash").execute()
    if res.data and len(res.data) > 0 and res.data[0].get("candidate_id"):
        return res.data[0]["candidate_id"]

    sel = sb.table("candidate").select("candidate_id").eq("resume_text_hash", row["resume_text_hash"]).single().execute()
    return sel.data["candidate_id"]

def insert_experience(sb: Client, candidate_id: str, experiences: List[Dict[str, Any]]) -> None:
    for e in (experiences or []):
        sb.table("experience").insert({
            "candidate_id": candidate_id,
            "company_raw": e.get("company_raw"),
            "company_norm": _name_norm(e.get("company_norm")),
            "title_raw": e.get("title_raw"),
            "title_norm": _name_norm(e.get("title_norm")),
            "seniority": e.get("seniority"),
            "employment_type": e.get("employment_type"),
            "location": e.get("location"),
            "start_date": e.get("start_date"),
            "end_date": e.get("end_date"),
            "is_current": e.get("is_current"),
        }).execute()
    print(f"[debug] Inserted {len(experiences)} experience entries")

def insert_education(sb: Client, candidate_id: str, educations: List[Dict[str, Any]]) -> None:
    for ed in educations:
        sb.table("education").insert({
            "candidate_id": candidate_id,
            "degree_raw": ed.get("degree_raw"),
            "degree_level": ed.get("degree_level"),
            "field_raw": ed.get("field_raw"),
            "field_norm": None,
            "institution_raw": ed.get("institution_raw"),
            "institution_norm": None,
            "start_date": ed.get("start_date"),
            "end_date": ed.get("end_date"),
            "gpa": ed.get("gpa"),
        }).execute()
    print(f"[debug] Inserted {len(educations)} education entries")

def insert_projects(sb: Client, candidate_id: str, projects: List[Dict[str, Any]]) -> None:
    for p in projects:
        sb.table("project").insert({
            "candidate_id": candidate_id,
            "name": p.get("name"),
            "role": p.get("role"),
            "start_date": p.get("start_date"),
            "end_date": p.get("end_date"),
            "repo_url": p.get("repo_url"),
            "tech_stack": p.get("tech_stack"),
            "project_summary": p.get("summary"),
        }).execute()
    print(f"[debug] Inserted {len(projects)} project entries")

def insert_misc(sb: Client, candidate_id: str, doc: Dict[str, Any]) -> None:
    for c in doc.get("certifications", []):
        sb.table("certification").insert({
            "candidate_id": candidate_id,
            "name": c.get("name"),
            "detailed_text": c.get("summary"),
            "issue_date": c.get("issue_date"),
            "credential_id": c.get("credential_id"),
            "url": c.get("url"),
        }).execute()
    print(f"[debug] Inserted {len(doc.get('certifications', []))} certifications")
    for p in doc.get("publications", []):
        sb.table("publication").insert({
            "candidate_id": candidate_id,
            "title": p.get("title"),
            "year": p.get("year"),
            "url": p.get("url"),
        }).execute()
    print(f"[debug] Inserted {len(doc.get('publications', []))} publications")
    achs = doc.get("achievements", [])
    description = " ".join([a if isinstance(a, str) else a.get("description") for a in achs if a])
    if description:
        sb.table("achievement").insert({
            "candidate_id": candidate_id,
            "description": description,
        }).execute()
    print(f"[debug] Inserted achievements (if description was non-empty)")
    langs = doc.get("languages", [])
    sb.table("candidate_language").upsert({
        "candidate_id": candidate_id,
        "data": langs
    }, on_conflict="candidate_id").execute()
    print(f"[debug] Upserted {len(langs)} languages")

def upsert_skills(sb: Client, candidate_id: str, skills: List[str]) -> None:
    if skills:
        sb.table("skill").upsert({
            "candidate_id": candidate_id,
            "name_norm": skills
        }, on_conflict="candidate_id").execute()
        print(f"[debug] Upserted {len(skills)} skills")

def ingest_one(sb: Client, in_json: str, job_id: Optional[str] = None) -> Dict[str, Any]:
    with open(in_json, "r", encoding="utf-8") as f:
        doc = json.load(f)

    cand = doc["candidate"]
    if job_id:
        cand["job_id"] = str(job_id)
        
    candidate_id = upsert_candidate(sb, cand, job_id)
    upsert_resume_source(sb, cand)

    # --- Insert raw section text if present ---
    for section in doc.get("raw_sections", []):
        sb.table("raw_section_text").insert({
            "candidate_id": candidate_id,
            "section_type": section.get("section_type"),
            "text": section.get("text"),
            "source_file_id": cand.get("source_file_id")
        }).execute()
    print(f"[debug] Inserted {len(doc.get('raw_sections', []))} raw section texts")

    insert_experience(sb, candidate_id, doc.get("experience") or [])
    insert_education(sb, candidate_id, doc.get("education") or [])
    insert_projects(sb, candidate_id, doc.get("projects") or [])
    insert_misc(sb, candidate_id, doc)

    skill_names = _skill_tokens(doc.get("skills") or [])
    upsert_skills(sb, candidate_id, skill_names)

    # --- Fetch job description from Supabase ---
    job_data = sb.table("job_postings").select("job_description").eq("id", job_id).execute()
    if not job_data.data:
        print(f"[warn] No job description found for job_id={job_id}")
        job_score = None
    else:
        job_desc = job_data.data[0]["job_description"]

        # Prefer raw resume text if available, fallback to skills+experience text
        resume_text = doc.get("raw_text") or " ".join(skill_names)
        job_score = compute_job_score(resume_text, candidate_id, job_desc)

        # --- Update candidate record with score ---
        sb.table("candidate").update({"job_score": job_score}).eq("candidate_id", candidate_id).execute()
        print(f"[debug] Candidate {candidate_id} scored {job_score}% for job {job_id}")

    return {"candidate_id": candidate_id, "job_score": job_score}
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_json", required=True)
    args = ap.parse_args()
    sb = get_client()
    ingest_one(sb, args.in_json)

if __name__ == "__main__":
    main()