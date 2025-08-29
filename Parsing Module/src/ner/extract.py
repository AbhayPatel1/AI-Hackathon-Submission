from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import dateparser
import spacy
from rapidfuzz import fuzz, process as rfprocess

# --------------------------- Setup & small resources ---------------------------

# High-precision patterns
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})")
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
DATE_RANGE_RE = re.compile(r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*,?\s*\d{4}|\d{4})(?:\s*[-–—]\s*| to )((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s*,?\s*\d{4}|\d{4}|present|current)\b", re.I)
YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")

# Light title lexicon (expand as you go)
TITLE_CANDIDATES = [
    "software engineer","sde","data scientist","data analyst","ml engineer","product manager",
    "technical product manager","intern","research intern","research consultant","frontend engineer",
    "backend engineer","full stack engineer","devops engineer"
]

# Seed skills lexicon (expand/replace with CSV later)
SKILL_CANON = [
    "python","c","c++","java","sql","pandas","numpy","scikit-learn","tensorflow","pytorch",
    "langchain","streamlit","django","flask","javascript","html","css","tableau","excel",
    "git","github","linux","docker","kubernetes","figma","google analytics","mixpanel","phonepe",
    "aws","gcp","azure","postgresql","mysql"
]

# ------------------------------ Small utilities -------------------------------

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
    """Return (start_date_iso, end_date_iso, is_current)."""
    m = DATE_RANGE_RE.search(text) or YEAR_RE.search(text)  # backup: single year
    if not m:
        return None, None, False
    if isinstance(m, re.Match) and m.re is DATE_RANGE_RE:
        start_raw, end_raw = m.group(1), m.group(2)
    else:
        # Single year fallback for education lines like "2020 - 2025" captured separately
        years = YEAR_RE.findall(text)
        if years:
            # crude heuristic: first -> start, last -> end
            ys = re.findall(r"(19|20)\d{2}", text)
        start_raw = None
        end_raw = None
        # bail to safer default
        return None, None, False

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

# ------------------------------- NER pipeline --------------------------------

def load_nlp(model: str = "en_core_web_trf"):
    return spacy.load(model, exclude=["lemmatizer"])  # small & fast

def extract_skills(text: str) -> List[str]:
    t = text.lower()
    found = set()
    for s in SKILL_CANON:
        if s in t:
            found.add(s)
    return sorted(found)

def extract_company_and_title(line: str, doc) -> Tuple[Optional[str], Optional[str], float]:
    """Heuristic split: try to separate 'Title | Company' or 'Title at Company'."""
    text = _norm(line)
    # Title via lexicon fuzzy
    title_norm = None
    for chunk in re.split(r"[|,•\-–—@]", text):
        hit = _best_fuzzy(chunk.strip().lower(), TITLE_CANDIDATES, cutoff=88)
        if hit:
            title_norm = hit
            break
    # Company via NER (ORG) as first guess
    orgs = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
    company_raw = orgs[0] if orgs else None

    # If pattern like "X | Y", keep the non-title part as company
    if "|" in text:
        parts = [p.strip() for p in text.split("|") if p.strip()]
        if title_norm and len(parts) >= 2:
            # choose the part not matching title words
            parts_scored = [(p, fuzz.token_set_ratio(p.lower(), title_norm)) for p in parts]
            parts_scored.sort(key=lambda x: x[1])
            company_raw = parts_scored[0][0]
    # confidence: weak but useful
    conf = 0.6 if (title_norm or company_raw) else 0.2
    return company_raw, title_norm, conf

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
    # drop duplicates/empties
    out = [o for o in out if o]
    return out

def extract_experiences(section_text: str, nlp) -> List[Experience]:
    chunks = re.split(r"\n{2,}", section_text.strip()) or [section_text]
    exps: List[Experience] = []
    page_hint = None

    for block in chunks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        header = lines[0]
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""
        doc = nlp(header)

        # company/title
        company_raw, title_norm, conf_ct = extract_company_and_title(header, doc)

        # dates
        start_date, end_date, is_current = _extract_date_range(header)  # header first
        if not start_date and body:
            s2, e2, c2 = _extract_date_range(body)
            start_date = start_date or s2
            end_date = end_date or e2
            is_current = is_current or c2

        # bullets + skills
        bullets_raw = split_bullets(body)
        skills_all = set(extract_skills(block))
        bullets: List[Bullet] = []
        for b in bullets_raw:
            skills_b = extract_skills(b)
            # toy metric extractor (extend later)
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
            company_norm=company_raw,   # place-holder for future normalization
            title_raw=header,
            title_norm=title_norm,
            seniority=None,
            employment_type="internship" if "intern" in (title_norm or "") else None,
            location=None,
            start_date=start_date,
            end_date=end_date,
            is_current=is_current,
            extraction_confidence=round(min(1.0, conf_ct + (0.1 if start_date else 0.0) + (0.15 if bullets else 0.0)), 2),
            source_page=page_hint,
            bullets=bullets
        ))

    return exps

def extract_education(section_text: str, nlp) -> List[Education]:
    blocks = re.split(r"\n{2,}", section_text.strip()) or [section_text]
    out: List[Education] = []
    for block in blocks:
        lines = [ln for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue
        txt = " ".join(lines)
        doc = nlp(txt)
        inst = None
        for ent in doc.ents:
            if ent.label_ == "ORG":
                inst = ent.text
                break
        # degrees/fields (lexicon-lite)
        degree_level = None
        degree_raw = None
        field_raw = None
        low = txt.lower()
        if "b.tech" in low or "btech" in low:
            degree_level, degree_raw = "bachelors", "B.Tech"
        if "m.tech" in low or "mtech" in low:
            degree_level, degree_raw = "masters", (degree_raw or "M.Tech")
        if "dual degree" in low:
            degree_raw = "Dual Degree"

        # dates
        s, e, _ = _extract_date_range(txt)

        # gpa
        gpa = None
        g = re.search(r"\b(\d+(?:\.\d+)?)/(10|4)\b", txt)
        if g:
            gpa = g.group(0)

        out.append(Education(
            degree_raw=degree_raw,
            degree_level=degree_level,
            field_raw=field_raw,
            institution_raw=inst,
            start_date=s, end_date=e,
            gpa=gpa,
            extraction_confidence=0.6 if inst or degree_raw else 0.3,
            source_page=None
        ))
    return out

def extract_projects(section_text: str, nlp) -> List[Project]:
    # split by title-like lines (Title Case or standalone lines)
    blocks = re.split(r"\n(?=[A-Z][A-Za-z0-9 ()/&+\-]{2,}$)", section_text.strip(), flags=re.M)
    out: List[Project] = []
    for blk in blocks:
        lines = [ln for ln in blk.splitlines() if ln.strip()]
        if not lines:
            continue
        name = lines[0].strip() if len(lines) else None
        summary = " ".join(lines[1:]) if len(lines) > 1 else None
        tech = extract_skills(blk)
        repo = None
        m = URL_RE.search(blk)
        if m and "github" in m.group(1).lower():
            repo = m.group(1)
        s, e, _ = _extract_date_range(blk)
        # toy metric: qps/accuracy
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
    hits = extract_skills(text)
    return [Skill(name=h) for h in hits]

def extract_achievements(text: str) -> List[str]:
    # keep as plain strings; we can NER later if needed
    bullets = [ln.lstrip("•-–—* ").strip() for ln in text.splitlines() if ln.strip()]
    return [b for b in bullets if b]

def extract_languages(text: str) -> List[Language]:
    # very light heuristic; replace with a small lexicon later
    langs = []
    t = text.lower()
    for lang in ["english","hindi","french","german","spanish"]:
        m = re.search(rf"\b{lang}\b(?:\s*\(([^)]+)\))?", t)
        if m:
            langs.append(Language(name=lang.title(), level=(m.group(1) or None)))
    return langs

# ------------------------------ Orchestration --------------------------------

def process(in_json: str, spacy_model: str = "en_core_web_trf") -> Dict[str, Any]:
    with open(in_json, "r", encoding="utf-8") as f:
        doc = json.load(f)

    nlp = load_nlp(spacy_model)

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
            exps = extract_experiences(text, nlp)
            # mark internships explicitly
            if stype == "internships":
                for e in exps:
                    e.employment_type = e.employment_type or "internship"
            experiences.extend([asdict(e) for e in exps])

        elif stype == "education":
            education.extend([asdict(e) for e in extract_education(text, nlp)])

        elif stype == "projects":
            projects.extend([asdict(p) for p in extract_projects(text, nlp)])

        elif stype == "skills":
            skills_flat.extend([asdict(s) for s in extract_skills_section(text)])

        elif stype in {"achievements", "responsibility"}:
            achievements.extend(extract_achievements(text))

        elif stype == "languages":
            languages.extend([asdict(l) for l in extract_languages(text)])

        # other sections are left as raw text for now; we still keep them in sectioned_text

    # Candidate cleanup: normalize phones, dedupe emails
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
        "certifications": [],     # can be filled later
        "publications": [],
        "achievements": achievements,
        "languages": languages,
        # keep the sectioned text verbatim for traceability
        "sectioned_text": sections,
    }
    return out

def _derive_out(in_json: str, out_json: Optional[str]) -> str:
    if out_json:
        return out_json
    base = os.path.splitext(os.path.basename(in_json))[0]
    os.makedirs("outputs", exist_ok=True)
    return os.path.join("outputs", f"{base.replace('_sections_nli','')}_ner.json")

def main():
    ap = argparse.ArgumentParser(description="Step 3: NER over NLI-sectioned resume")
    ap.add_argument("--in_json", required=True, help="Input *_lines_sections_nli.json")
    ap.add_argument("--out_json", help="Output JSON (default: outputs/<in>_ner.json)")
    ap.add_argument("--spacy_model", default="en_core_web_sm", help="spaCy model to use")
    args = ap.parse_args()

    result = process(args.in_json, spacy_model=args.spacy_model)
    out_path = _derive_out(args.in_json, args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ok] NER output → {out_path}")

if __name__ == "__main__":
    main()
