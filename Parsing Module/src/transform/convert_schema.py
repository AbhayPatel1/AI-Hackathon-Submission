from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# ---------- helpers ----------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})")

def _norm(s: Optional[str]) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _digits_only(s: str) -> str:
    return re.sub(r"[^0-9]", "", s)

def _valid_phone(tok: str) -> bool:
    d = _digits_only(tok)
    return 8 <= len(d) <= 15

def _normalize_phone(tok: str, country_hint: Optional[str] = "IN") -> str:
    d = _digits_only(tok)
    if country_hint == "IN" and not tok.strip().startswith("+") and len(d) == 10:
        return "+91 " + d
    return ("+" + d) if tok.strip().startswith("+") else d

def _sha256_of_sections(sections: List[Dict[str, Any]]) -> str:
    blob = "\n\n".join([sec.get("text", "") or "" for sec in sections])
    return "sha256:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()

def _basename(path: str) -> str:
    return os.path.basename(path) if path else None

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def _infer_employment_type(title_raw: Optional[str], title_norm: Optional[str]) -> Optional[str]:
    tr = (title_raw or "").lower()
    tn = (title_norm or "").lower()
    if "intern" in tr or "intern" in tn:
        return "internship"
    return None

# ---------- core transform ----------

def convert_to_canonical(
    ner_doc: Dict[str, Any],
    source_file_id: Optional[str] = None,
    parser_version: str = "v0.1.0"
) -> Dict[str, Any]:

    meta = ner_doc.get("meta", {}) or {}
    candidate_in = ner_doc.get("candidate", {}) or {}
    sections = ner_doc.get("sectioned_text", []) or []

    # candidate block
    emails = sorted(set(candidate_in.get("emails", [])))
    phones_in = candidate_in.get("phones", []) or []
    phones = sorted({ _normalize_phone(p) for p in phones_in if _valid_phone(p) })

    links_in = candidate_in.get("links", {}) or {}
    links = {
        "linkedin": links_in.get("linkedin"),
        "github": links_in.get("github")
    }

    canonical_candidate = {
        "full_name": candidate_in.get("full_name"),
        "emails": emails,
        "phones": phones,
        "location": {"city": None, "state": None, "country": None},  # placeholder; can be enriched later
        "links": links,
        "resume_text_hash": _sha256_of_sections(sections),
        "source_file_id": source_file_id or _basename(meta.get("source_path", "")) or None,
        "parsed_at": _now_iso(),
        "parser_version": parser_version,
    }

    # experience
    experiences_out = []
    for e in ner_doc.get("experience", []) or []:
        title_raw = e.get("title_raw")
        title_norm = e.get("title_norm")
        employment_type = e.get("employment_type") or _infer_employment_type(title_raw, title_norm)

        bullets_out = []
        for b in e.get("bullets", []) or []:
            bullets_out.append({
                "bullet_text": b.get("bullet_text"),
                "metrics": b.get("metrics", {}) or {},
                "skills_mentioned": b.get("skills_mentioned", []) or [],
                "source_page": b.get("source_page"),
            })

        experiences_out.append({
            "company_raw": e.get("company_raw"),
            "company_norm": e.get("company_norm"),
            "title_raw": title_raw,
            "title_norm": title_norm,
            "seniority": e.get("seniority"),
            "employment_type": employment_type,
            "location": e.get("location"),
            "start_date": e.get("start_date"),
            "end_date": e.get("end_date"),
            "is_current": bool(e.get("is_current")),
            "extraction_confidence": e.get("extraction_confidence", 0.0),
            "source_page": e.get("source_page"),
            "bullets": bullets_out,
        })

    # education
    education_out = []
    for ed in ner_doc.get("education", []) or []:
        education_out.append({
            "degree_raw": ed.get("degree_raw"),
            "degree_level": ed.get("degree_level"),
            "field_raw": ed.get("field_raw"),
            "institution_raw": ed.get("institution_raw"),
            "start_date": ed.get("start_date"),
            "end_date": ed.get("end_date"),
            "gpa": ed.get("gpa"),
            "extraction_confidence": ed.get("extraction_confidence", 0.0),
            "source_page": ed.get("source_page"),
        })

    # skills
    skills_out = []
    for s in ner_doc.get("skills", []) or []:
        if isinstance(s, dict):
            skills_out.append({"name": s.get("name"), "category": s.get("category")})
        elif isinstance(s, str):
            skills_out.append({"name": s, "category": None})

    # projects
    projects_out = []
    for p in ner_doc.get("projects", []) or []:
        projects_out.append({
            "name": p.get("name"),
            "summary": p.get("summary"),
            "role": p.get("role"),
            "tech_stack": p.get("tech_stack", []) or [],
            "start_date": p.get("start_date"),
            "end_date": p.get("end_date"),
            "repo_url": p.get("repo_url"),
            "impact_metrics": p.get("impact_metrics", {}) or {},
            "source_page": p.get("source_page"),
        })

    # certifications / publications / achievements / languages
    certs_out = ner_doc.get("certifications", []) or []
    pubs_out = ner_doc.get("publications", []) or []
    achievements_out = ner_doc.get("achievements", []) or []

    languages_out = []
    for l in ner_doc.get("languages", []) or []:
        if isinstance(l, dict):
            languages_out.append({"name": l.get("name"), "level": l.get("level")})
        elif isinstance(l, str):
            languages_out.append({"name": l, "level": None})

    # final doc
    out = {
        "candidate": canonical_candidate,
        "experience": experiences_out,
        "education": education_out,
        "skills": skills_out,
        "projects": projects_out,
        "certifications": certs_out,
        "publications": pubs_out,
        "achievements": achievements_out,
        "languages": languages_out,
        "sectioned_text": sections,  # keep for traceability
    }
    return out

def _derive_out_path(in_json: str, out_json: Optional[str]) -> str:
    if out_json:
        return out_json
    base = os.path.splitext(os.path.basename(in_json))[0]
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", f"{base.replace('_ner', '').replace('_hf','').replace('_ens','')}_canonical.json")

def main():
    ap = argparse.ArgumentParser(description="Convert NER JSON to canonical resume schema")
    ap.add_argument("--in_json", required=True, help="Input: *_ner.json / *_ner_hf.json / *_ner_ens.json")
    ap.add_argument("--out_json", help="Output path (default: outputs/<stem>_canonical.json)")
    ap.add_argument("--source_file_id", help="Override source_file_id (default: basename of meta.source_path)")
    ap.add_argument("--parser_version", default="v0.1.0", help="Parser version string")
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        ner_doc = json.load(f)

    out_doc = convert_to_canonical(
        ner_doc=ner_doc,
        source_file_id=args.source_file_id,
        parser_version=args.parser_version,
    )

    out_path = _derive_out_path(args.in_json, args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_doc, f, ensure_ascii=False, indent=2)
    print(f"[ok] canonical JSON → {out_path}")

if __name__ == "__main__":
    main()