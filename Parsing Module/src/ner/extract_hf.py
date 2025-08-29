from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import dateparser
from transformers import pipeline
from rapidfuzz import fuzz, process as rfprocess

# --------------------------- Patterns & resources ---------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})")
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
DATE_RANGE_RE = re.compile(
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*,?\s*\d{4}|\d{4})"
    r"(?:\s*[-–—]\s*| to )"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*,?\s*\d{4}|\d{4}|present|current)\b",
    re.I,
)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

TITLE_CANDIDATES = [
    "software engineer","sde","data scientist","data analyst","ml engineer","product manager",
    "technical product manager","intern","research intern","frontend engineer",
    "backend engineer","full stack engineer","devops engineer"
]

SKILL_CANON = [
    "python","c","c++","java","sql","pandas","numpy","scikit-learn","tensorflow","pytorch",
    "langchain","streamlit","django","flask","javascript","html","css","tableau","excel",
    "git","github","linux","docker","kubernetes","figma","google analytics","mixpanel","phonepe",
    "aws","gcp","azure","postgresql","mysql","solidity","plaxis","matlab","autocad","revit","staad pro"
]

HF_DEFAULT = "dslim/bert-base-NER"  # solid CoNLL NER baseline

# ------------------------------ Small helpers -------------------------------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _digits_only(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

def _valid_phone(tok: str) -> bool:
    if not tok:
        return False
    d = _digits_only(tok)
    if len(d) < 8 or len(d) > 15:
        return False
    if len(d) in (8, 9) and not tok.strip().startswith("+"):
        return False
    return True

def _normalize_phone(tok: str, country_hint: Optional[str] = "IN") -> str:
    d = _digits_only(tok)
    if country_hint == "IN" and not tok.strip().startswith("+") and len(d) == 10:
        return "+91 " + d
    return ("+" + d) if tok.strip().startswith("+") else d

def _parse_date(s: str) -> Optional[str]:
    dt = dateparser.parse(s, settings={"PREFER_DAY_OF_MONTH": "first"})
    return dt.date().isoformat() if dt else None

def _extract_date_range(text: str) -> Tuple[Optional[str], Optional[str], bool]:
    m = DATE_RANGE_RE.search(text)
    if not m:
        return None, None, False
    start_raw, end_raw = m.group(1), m.group(2)
    is_current = str(end_raw).lower() in {"present","current"}
    start_iso = _parse_date(start_raw)
    end_iso = None if is_current else _parse_date(end_raw)
    return start_iso, end_iso, is_current

def _best_fuzzy(item: str, candidates: List[str], cutoff: int = 85) -> Optional[str]:
    item = (item or "").lower()
    match = rfprocess.extractOne(item, candidates, scorer=fuzz.token_set_ratio)
    if match and match[1] >= cutoff:
        return match[0]
    return None

# ------------------------------- Data classes --------------------------------

@dataclass
class Bullet:
    bullet_text: str
    metrics: Dict[str, Any]
    skills_mentioned: List[str]
    source_page: Optional[int] = None

@dataclass
class Experience:
    company_raw: Optional[str]
    company_norm: Optional[str]
    title_raw: Optional[str]
    title_norm: Optional[str]
    seniority: Optional[str]
    employment_type: Optional[str]
    location: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    is_current: bool
    extraction_confidence: float
    source_page: Optional[int]
    bullets: List[Bullet]

@dataclass
class Education:
    degree_raw: Optional[str]
    degree_level: Optional[str]
    field_raw: Optional[str]
    institution_raw: Optional[str]
    start_date: Optional[str]
    end_date: Optional[str]
    gpa: Optional[str]
    extraction_confidence: float
    source_page: Optional[int]

@dataclass
class Project:
    name: Optional[str]
    summary: Optional[str]
    role: Optional[str]
    tech_stack: List[str]
    start_date: Optional[str]
    end_date: Optional[str]
    repo_url: Optional[str]
    impact_metrics: Dict[str, Any]
    source_page: Optional[int]

@dataclass
class Skill:
    name: str
    category: Optional[str] = None

@dataclass
class Language:
    name: str
    level: Optional[str] = None

# ------------------------------- HF pipeline ---------------------------------

def build_hf_ner(model_name: str = HF_DEFAULT, device: Optional[int] = None):
    """Hugging Face token-classification pipeline with simple aggregation."""
    return pipeline(
        task="token-classification",
        model=model_name,
        aggregation_strategy="simple",
        device=-1 if device is None else device,
    )

def ents_by_label(hf_pipeline, text: str) -> Dict[str, List[str]]:
    """Return a dict like {'ORG': [...], 'PER': [...], 'LOC': [...], 'MISC': [...], 'DATE': [...]}."""
    if not text.strip():
        return {}
    out = hf_pipeline(text)
    # Normalize labels like B-ORG/I-ORG → ORG (pipeline aggregation handles most)
    by = {}
    for ent in out:
        lbl = ent.get("entity_group", ent.get("entity", "")).upper()
        val = ent.get("word", "").strip()
        if not val:
            continue
        by.setdefault(lbl, []).append(val)
    return by

# ------------------------------ Extraction bits ------------------------------

def extract_skills(text: str) -> List[str]:
    t = text.lower()
    return sorted({s for s in SKILL_CANON if s in t})

def split_bullets(text: str) -> List[str]:
    lines = [ln.strip() for ln in text.splitlines()]
    out, buf = [], []
    for ln in lines:
        if not ln:
            continue
        if ln.startswith(("•","-","–","—","*")):
            if buf:
                out.append(" ".join(buf).strip())
                buf = []
            out.append(ln.lstrip("•-–—* ").strip())
        else:
            buf.append(ln)
    if buf:
        out.append(" ".join(buf).strip())
    return [o for o in out if o]

def extract_company_and_title_hf(header: str, ents: Dict[str, List[str]]) -> Tuple[Optional[str], Optional[str], float]:
    """Heuristic: pick first ORG as company; fuzzy title from lexicon within header."""
    text = _norm(header)
    company = ents.get("ORG", [None])[0]
    # title via fuzzy on header parts
    title_norm = None
    for chunk in re.split(r"[|,•\-–—@/]", text):
        hit = _best_fuzzy(chunk.strip().lower(), TITLE_CANDIDATES, cutoff=88)
        if hit:
            title_norm = hit
            break
    conf = 0.65 if (company or title_norm) else 0.25
    return company, title_norm, conf

def extract_experiences(section_text: str, hf_ner) -> List[Experience]:
    blocks = re.split(r"\n{2,}", section_text.strip()) or [section_text]
    exps: List[Experience] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        header = lines[0]
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""

        ents_header = ents_by_label(hf_ner, header)
        company_raw, title_norm, conf_ct = extract_company_and_title_hf(header, ents_header)

        s, e, cur = _extract_date_range(header)
        if not s and body:
            s2, e2, c2 = _extract_date_range(body)
            s, e, cur = s or s2, e or e2, cur or c2

        bullets_raw = split_bullets(body)
        bullets: List[Bullet] = []
        for b in bullets_raw:
            skills_b = extract_skills(b)
            metrics = {}
            pct = re.search(r"(\d{1,3})\s*%", b)
            if pct:
                metrics["percent"] = int(pct.group(1))
            qps = re.search(r"(\d{2,5})\s*qps", b, re.I)
            if qps:
                metrics["throughput_qps"] = int(qps.group(1))
            p99 = re.search(r"p99\s*[:=]?\s*(\d{1,5})\s*ms", b, re.I)
            if p99:
                metrics["latency_p99_ms"] = int(p99.group(1))
            bullets.append(Bullet(bullet_text=b, metrics=metrics, skills_mentioned=skills_b))

        exps.append(Experience(
            company_raw=company_raw,
            company_norm=company_raw,
            title_raw=header,
            title_norm=title_norm,
            seniority=None,
            employment_type="internship" if (title_norm and "intern" in title_norm) else None,
            location=None,
            start_date=s, end_date=e, is_current=bool(cur),
            extraction_confidence=round(min(1.0, conf_ct + (0.1 if s else 0.0) + (0.15 if bullets else 0.0)), 2),
            source_page=None,
            bullets=bullets
        ))
    return exps

def extract_education(section_text: str, hf_ner) -> List[Education]:
    blocks = re.split(r"\n{2,}", section_text.strip()) or [section_text]
    out: List[Education] = []
    for blk in blocks:
        txt = _norm(blk)
        ents = ents_by_label(hf_ner, txt)
        inst = (ents.get("ORG") or [None])[0]
        s, e, _ = _extract_date_range(txt)
        gpa = None
        g = re.search(r"\b(\d+(?:\.\d+)?)/(10|4)\b", txt)
        if g:
            gpa = g.group(0)

        # very light degree guess
        degree_raw, degree_level = None, None
        low = txt.lower()
        if "b.tech" in low or "btech" in low or "b.e." in low or "b.e " in low:
            degree_raw, degree_level = "B.Tech/B.E.", "bachelors"
        if "m.tech" in low or "mtech" in low or "m.e." in low or "m.e " in low:
            degree_raw, degree_level = "M.Tech/M.E.", "masters"
        if "dual degree" in low:
            degree_raw = "Dual Degree"

        out.append(Education(
            degree_raw=degree_raw, degree_level=degree_level,
            field_raw=None, institution_raw=inst,
            start_date=s, end_date=e, gpa=gpa,
            extraction_confidence=0.6 if inst or degree_raw else 0.3,
            source_page=None
        ))
    return out

def extract_projects(section_text: str, hf_ner) -> List[Project]:
    blocks = re.split(r"\n(?=[A-Z][A-Za-z0-9 ()/&+\-]{2,}$)", section_text.strip(), flags=re.M)
    out: List[Project] = []
    for blk in blocks:
        lines = [ln for ln in blk.splitlines() if ln.strip()]
        if not lines:
            continue
        name = lines[0].strip()
        summary = " ".join(lines[1:]) if len(lines) > 1 else None
        tech = extract_skills(blk)
        repo = None
        m = URL_RE.search(blk)
        if m and "github" in m.group(1).lower():
            repo = m.group(1)
        s, e, _ = _extract_date_range(blk)
        impact = {}
        acc = re.search(r"(\d{2,3})\s*%[^a-zA-Z]?(\s*accuracy|\bacc\b)", blk, re.I)
        if acc:
            impact["accuracy_pct"] = int(acc.group(1))
        qps = re.search(r"(\d{2,5})\s*qps", blk, re.I)
        if qps:
            impact["throughput_qps"] = int(qps.group(1))

        out.append(Project(
            name=name, summary=summary, role=None, tech_stack=tech,
            start_date=s, end_date=e, repo_url=repo, impact_metrics=impact, source_page=None
        ))
    return out

def extract_skills_section(text: str) -> List[Skill]:
    return [Skill(name=s) for s in extract_skills(text)]

def extract_achievements(text: str) -> List[str]:
    return [ln.lstrip("•-–—* ").strip() for ln in text.splitlines() if ln.strip()]

def extract_languages(text: str) -> List[Language]:
    langs = []
    t = text.lower()
    for lang in ["english","hindi","french","german","spanish","malayalam","tamil","telugu"]:
        m = re.search(rf"\b{lang}\b(?:\s*\(([^)]+)\))?", t)
        if m:
            langs.append(Language(name=lang.title(), level=(m.group(1) or None)))
    return langs

# -------------------------------- Orchestrator --------------------------------

def process(in_json: str, model_name: str = HF_DEFAULT, device: Optional[int] = None) -> Dict[str, Any]:
    with open(in_json, "r", encoding="utf-8") as f:
        doc = json.load(f)

    hf_ner = build_hf_ner(model_name=model_name, device=device)

    candidate = doc.get("candidate", {})
    sections = doc.get("sectioned_text", [])

    experiences: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []
    projects: List[Dict[str, Any]] = []
    skills_flat: List[Dict[str, Any]] = []
    achievements: List[str] = []
    languages: List[Dict[str, Any]] = []

    for sec in sections:
        stype = sec.get("section_type")
        text = sec.get("text", "")

        if stype in {"work_experience", "internships"}:
            exps = extract_experiences(text, hf_ner)
            if stype == "internships":
                for e in exps:
                    e.employment_type = e.employment_type or "internship"
            experiences.extend([asdict(e) for e in exps])

        elif stype == "education":
            education.extend([asdict(e) for e in extract_education(text, hf_ner)])

        elif stype == "projects":
            projects.extend([asdict(p) for p in extract_projects(text, hf_ner)])

        elif stype == "skills":
            skills_flat.extend([asdict(s) for s in extract_skills_section(text)])

        elif stype in {"achievements", "responsibility"}:
            achievements.extend(extract_achievements(text))

        elif stype == "languages":
            languages.extend([asdict(l) for l in extract_languages(text)])

        # others left as raw section text in sectioned_text for traceability

    # Candidate cleanup
    emails = sorted(set(candidate.get("emails", [])))
    phones_raw = candidate.get("phones", [])
    phones = []
    for p in phones_raw:
        if _valid_phone(p):
            phones.append(_normalize_phone(p))
    phones = sorted(set(phones))

    out = {
        "meta": doc.get("meta", {}),
        "candidate": {
            "full_name": candidate.get("full_name"),
            "emails": emails,
            "phones": phones,
            "links": candidate.get("links", {}),
        },
        "experience": experiences,
        "education": education,
        "skills": skills_flat,
        "projects": projects,
        "certifications": [],
        "publications": [],
        "achievements": achievements,
        "languages": languages,
        "sectioned_text": sections,
    }
    return out

def _derive_out(in_json: str, out_json: Optional[str]) -> str:
    if out_json:
        return out_json
    base = os.path.splitext(os.path.basename(in_json))[0]
    os.makedirs("outputs", exist_ok=True)
    # produces e.g., abhay_lines_sections_nli_ner_hf.json
    return os.path.join("outputs", f"{base}_ner_hf.json")

def main():
    ap = argparse.ArgumentParser(description="HF NER over NLI-sectioned resume")
    ap.add_argument("--in_json", required=True, help="Input *_lines_sections_nli.json")
    ap.add_argument("--out_json", help="Output JSON (default: outputs/<in>_ner_hf.json)")
    ap.add_argument("--model_name", default=HF_DEFAULT, help="HF token-classification model (default dslim/bert-base-NER)")
    ap.add_argument("--device", type=int, default=None, help="GPU id; omit for CPU")
    args = ap.parse_args()

    result = process(args.in_json, model_name=args.model_name, device=args.device)
    out_path = _derive_out(args.in_json, args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ok] HF NER output → {out_path}")

if __name__ == "__main__":
    main()