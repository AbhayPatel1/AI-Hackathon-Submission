from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import dateparser
import spacy
from transformers import pipeline
from rapidfuzz import fuzz, process as rfprocess

# --------------------------- Config / resources ---------------------------

HF_DEFAULT = "dslim/bert-base-NER"     # solid CoNLL NER baseline
SPACY_DEFAULT = "en_core_web_trf"       # upgrade to en_core_web_trf when ready

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})")
URL_RE   = re.compile(r"(https?://\S+|www\.\S+)")
DATE_RANGE_RE = re.compile(
    r"\b((?:Jan|Feb|Mar|Apr|May|Jun|July?|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s*['’]?\d{2,4}|\d{4})"
    r"(?:\s*[-–—]\s*| to )"
    r"((?:Jan|Feb|Mar|Apr|May|Jun|July?|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s*['’]?\d{2,4}|\d{4}|present|current)\b",
    re.I,
)

TITLE_CANDIDATES = [
    "software engineer","sde","sde i","sde ii","senior software engineer","data scientist","data analyst",
    "ml engineer","machine learning engineer","product manager","technical product manager","intern",
    "research intern","frontend engineer","backend engineer","full stack engineer","devops engineer"
]

# Keep skills canonical, you can externalize as CSV later
SKILL_CANON = [
    "python","c","c++","java","sql","pandas","numpy","scikit-learn","tensorflow","pytorch",
    "langchain","streamlit","django","flask","javascript","html","css","tableau","excel",
    "git","github","linux","docker","kubernetes","figma","google analytics","mixpanel","phonepe",
    "aws","gcp","azure","postgresql","mysql","power bi","matlab","autocad","revit","staad pro"
]

# ------------------------------ Small helpers ------------------------------

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


SEASON_MAP = {
    "spring": "03",
    "summer": "06",
    "autumn": "09",
    "fall": "09",
    "winter": "12",
}

def _normalize_date_str(s: str) -> str:
    s = (s or "").lower().replace("’", "'").strip()
    # normalize months like "sept" to "sep"
    s = re.sub(r"\bsept\b", "sep", s)
    # handle seasons like "spring'20"
    for season, month in SEASON_MAP.items():
        if season in s:
            yr = re.search(r"['’]?(\d{2,4})", s)
            if yr:
                y = yr.group(1)
                if len(y) == 2:
                    y = "20" + y
                return f"{y}-{month}"
    return s

def _parse_date(s: str) -> Optional[str]:
    if not s:
        return None
    s_norm = _normalize_date_str(s)
    dt = dateparser.parse(s_norm, settings={"PREFER_DAY_OF_MONTH": "first"})
    if not dt:
        return None
    return dt.strftime("%Y-%m")

def _extract_date_range(text: str) -> Tuple[Optional[str], Optional[str], bool]:
    if not text:
        return None, None, False
    t = text.replace("’", "'")
    # Handle open-ended ranges like "Aug 2023 - Present" or "March 2022 – Current"
    if re.search(r"\b(present|current)\b", t, re.I):
        # try to capture start date before "present/current"
        parts = re.split(r"\b(?:present|current)\b", t, flags=re.I)
        start_iso = _parse_date(parts[0]) if parts else None
        return start_iso, None, True
    m = DATE_RANGE_RE.search(t)
    if m:
        start_raw, end_raw = m.group(1), m.group(2)
        is_current = str(end_raw).lower() in {"present", "current"}
        start_iso = _parse_date(start_raw)
        end_iso = None if is_current else _parse_date(end_raw)
        return start_iso, end_iso, is_current
    # fallback: try single date (e.g. "Aug, 2023", "March 2024", "2024")
    single = re.search(
        r"(?:Jan|Feb|Mar|Apr|May|Jun|July?|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?,?\s*['’]?\d{2,4}|\d{4}",
        t,
        re.I,
    )
    if single:
        s = _parse_date(single.group(0))
        return s, s, False
    return None, None, False

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

# ---------------------------- Model builders ---------------------------------

def load_spacy(model: str = SPACY_DEFAULT):
    return spacy.load(model, exclude=["lemmatizer"])

def build_hf_ner(model_name: str = HF_DEFAULT, device: Optional[int] = None):
    return pipeline(
        task="token-classification",
        model=model_name,
        aggregation_strategy="simple",
        device=-1 if device is None else device,
    )

# ---------------------------- Entity extractors ------------------------------

def ents_by_label_hf(hf_pipeline, text: str) -> Dict[str, List[str]]:
    if not text.strip():
        return {}
    by: Dict[str, List[str]] = {}
    for ent in hf_pipeline(text):
        lbl = ent.get("entity_group", ent.get("entity", "")).upper()
        val = ent.get("word", "").strip()
        if not val:
            continue
        by.setdefault(lbl, []).append(val)
    return by

def ents_by_label_spacy(nlp, text: str) -> Dict[str, List[str]]:
    if not text.strip():
        return {}
    doc = nlp(text)
    by: Dict[str, List[str]] = {}
    for ent in doc.ents:
        by.setdefault(ent.label_.upper(), []).append(ent.text)
    return by

def extract_skills(text: str) -> List[str]:
    """Dictionary hit + light cleanup."""
    t = text.lower()
    hits = {s for s in SKILL_CANON if s in t}
    # Drop single-letter skills except 'r'
    hits = {h for h in hits if len(h) > 1 or h == "r"}
    return sorted(hits)

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

# ---------------------------- Ensemble heuristics ----------------------------

def pick_company(header: str, ents_hf: Dict[str, List[str]], ents_sp: Dict[str, List[str]]) -> Optional[str]:
    # Prefer HF ORG, then spaCy ORG; fallback: capitalized token after 'at' or after a pipe
    company = (ents_hf.get("ORG") or [None])[0]
    if not company:
        company = (ents_sp.get("ORG") or [None])[0]
    if company:
        return company
    m = re.search(r"(?:at|@)\s+([A-Z][\w&.\- ]{2,})", header)
    if m:
        return m.group(1).strip()
    parts = [p.strip() for p in re.split(r"[|@]", header) if p.strip()]
    if len(parts) >= 2:
        return parts[-1]
    return None

def pick_title(header: str) -> Optional[str]:
    # Fuzzy match any header chunk to title lexicon
    for chunk in re.split(r"[|,•\-–—@/()]", header):
        hit = _best_fuzzy(chunk.strip().lower(), TITLE_CANDIDATES, cutoff=88)
        if hit:
            return hit
    return None

def merge_skills(*lists: List[str]) -> List[str]:
    s = set()
    for L in lists:
        for x in L or []:
            if not x:
                continue
            xx = x.strip().lower()
            if len(xx) == 1 and xx != "r":
                continue
            s.add(xx)
    return sorted(s)

# --------------------------- Section extractors ------------------------------

def extract_experiences(section_text: str, nlp, hf_ner) -> List[Experience]:
    blocks = re.split(r"\n{2,}", section_text.strip()) or [section_text]
    out: List[Experience] = []
    for blk in blocks:
        lines = [ln for ln in blk.splitlines() if ln.strip()]
        if not lines:
            continue
        header = _norm(lines[0])
        body = "\n".join(lines[1:]) if len(lines) > 1 else ""

        ents_hf_h = ents_by_label_hf(hf_ner, header)
        ents_sp_h = ents_by_label_spacy(nlp, header)

        company_raw = pick_company(header, ents_hf_h, ents_sp_h)
        title_norm = pick_title(header)

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
            if pct: metrics["percent"] = int(pct.group(1))
            qps = re.search(r"(\d{2,5})\s*qps", b, re.I)
            if qps: metrics["throughput_qps"] = int(qps.group(1))
            p99 = re.search(r"p99\s*[:=]?\s*(\d{1,5})\s*ms", b, re.I)
            if p99: metrics["latency_p99_ms"] = int(p99.group(1))
            bullets.append(Bullet(bullet_text=b, metrics=metrics, skills_mentioned=skills_b))

        conf = 0.5
        if company_raw: conf += 0.2
        if title_norm: conf += 0.2
        if s or e: conf += 0.1
        if bullets: conf += 0.1
        conf = min(1.0, conf)

        out.append(Experience(
            company_raw=company_raw,
            company_norm=company_raw,
            title_raw=header,
            title_norm=title_norm,
            seniority=None,
            employment_type="internship" if (title_norm and "intern" in title_norm) else None,
            location=None,
            start_date=s, end_date=e, is_current=bool(cur),
            extraction_confidence=round(conf, 2),
            source_page=None,
            bullets=bullets
        ))
    return out

def extract_education(section_text: str, nlp, hf_ner) -> List[Education]:
    blocks = re.split(r"\n{2,}", section_text.strip()) or [section_text]
    out: List[Education] = []
    for blk in blocks:
        txt = _norm(blk)
        ents_hf = ents_by_label_hf(hf_ner, txt)
        ents_sp = ents_by_label_spacy(nlp, txt)
        inst = (ents_hf.get("ORG") or ents_sp.get("ORG") or [None])[0]

        s, e, _ = _extract_date_range(txt)

        gpa = None
        g = re.search(r"\b(\d+(?:\.\d+)?)/(10|4)\b", txt)
        if g: gpa = g.group(0)

        degree_raw, degree_level = None, None
        low = txt.lower()
        if any(k in low for k in ["b.tech","btech","b.e.","b.e "]):
            degree_raw, degree_level = "B.Tech/B.E.", "bachelors"
        if any(k in low for k in ["m.tech","mtech","m.e.","m.e "]):
            degree_raw, degree_level = "M.Tech/M.E.", "masters"
        if "dual degree" in low:
            degree_raw = "Dual Degree"

        conf = 0.4 + (0.2 if inst else 0) + (0.1 if degree_raw else 0) + (0.1 if s or e else 0)
        out.append(Education(
            degree_raw=degree_raw, degree_level=degree_level,
            field_raw=None, institution_raw=inst,
            start_date=s, end_date=e, gpa=gpa,
            extraction_confidence=round(min(1.0, conf), 2),
            source_page=None
        ))
    return out

def extract_projects(section_text: str, nlp, hf_ner) -> List[Project]:
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
        if acc: impact["accuracy_pct"] = int(acc.group(1))
        qps = re.search(r"(\d{2,5})\s*qps", blk, re.I)
        if qps: impact["throughput_qps"] = int(qps.group(1))

        out.append(Project(
            name=name, summary=summary, role=None, tech_stack=tech,
            start_date=s, end_date=e, repo_url=repo, impact_metrics=impact, source_page=None
        ))
    return out

def extract_skills_section(text: str, nlp, hf_ner) -> List[Skill]:
    # Ensemble: dict hits + entities classified as MISC/LANGUAGE from HF + ORG/GPE fallbacks are ignored
    dict_hits = set(extract_skills(text))
    # In case a model tags named techs as MISC/LANGUAGE
    hf = ents_by_label_hf(hf_ner, text)
    misc_lang = set(map(str.lower, hf.get("MISC", []) + hf.get("LANGUAGE", [])))
    # Filter to canonical vocabulary to avoid noise like single 'c'
    merged = {s for s in dict_hits.union(misc_lang) if s in SKILL_CANON}
    return [Skill(name=s) for s in sorted(merged)]

def extract_achievements(text: str) -> List[str]:
    return [ln.lstrip("•-–—* ").strip() for ln in text.splitlines() if ln.strip()]

def extract_languages(text: str, nlp, hf_ner) -> List[Language]:
    langs = []
    t = text.lower()
    for lang in ["english","hindi","french","german","spanish","malayalam","tamil","telugu"]:
        m = re.search(rf"\b{lang}\b(?:\s*\(([^)]+)\))?", t)
        if m:
            langs.append(Language(name=lang.title(), level=(m.group(1) or None)))
    return langs

# -------------------------------- Orchestrator --------------------------------

def process(in_json: str, hf_model: str = HF_DEFAULT, spacy_model: str = SPACY_DEFAULT, device: Optional[int] = None) -> Dict[str, Any]:
    with open(in_json, "r", encoding="utf-8") as f:
        doc = json.load(f)

    nlp = load_spacy(spacy_model)
    hf_ner = build_hf_ner(model_name=hf_model, device=device)

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
            exps = extract_experiences(text, nlp, hf_ner)
            if stype == "internships":
                for e in exps:
                    e.employment_type = e.employment_type or "internship"
            experiences.extend([asdict(e) for e in exps])

        elif stype == "education":
            education.extend([asdict(e) for e in extract_education(text, nlp, hf_ner)])

        elif stype == "projects":
            projects.extend([asdict(p) for p in extract_projects(text, nlp, hf_ner)])

        elif stype == "skills":
            skills_flat.extend([asdict(s) for s in extract_skills_section(text, nlp, hf_ner)])

        elif stype in {"achievements", "responsibility"}:
            achievements.extend(extract_achievements(text))

        elif stype == "languages":
            languages.extend([asdict(l) for l in extract_languages(text, nlp, hf_ner)])

    # Candidate merge/cleanup (regex dominates)
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
    return os.path.join("outputs", f"{base}_ner_ens.json")

def main():
    ap = argparse.ArgumentParser(description="Ensemble NER (spaCy + HF) over NLI-sectioned resume")
    ap.add_argument("--in_json", required=True, help="Input *_lines_sections_nli.json")
    ap.add_argument("--out_json", help="Output JSON (default: outputs/<in>_ner_ens.json)")
    ap.add_argument("--hf_model", default=HF_DEFAULT, help="HF token-classification model (default dslim/bert-base-NER)")
    ap.add_argument("--spacy_model", default=SPACY_DEFAULT, help="spaCy model (default en_core_web_sm)")
    ap.add_argument("--device", type=int, default=None, help="GPU id for HF; omit for CPU")
    args = ap.parse_args()

    result = process(args.in_json, hf_model=args.hf_model, spacy_model=args.spacy_model, device=args.device)
    out_path = _derive_out(args.in_json, args.out_json)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ok] Ensemble NER output → {out_path}")

if __name__ == "__main__":
    main()