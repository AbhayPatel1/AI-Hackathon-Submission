from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple, Set

# ---------------------------
# small utils
# ---------------------------

SPACE_RE = re.compile(r"\s+")

def _norm(s: Optional[str]) -> str:
    if not s:
        return ""
    return SPACE_RE.sub(" ", s).strip().lower()

def _norm_set(xs: Optional[List[str]]) -> Set[str]:
    return { _norm(x) for x in (xs or []) if _norm(x) }

def _read(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _safe(doc: Optional[Dict[str, Any]], key: str, default):
    if not doc:
        return default
    return doc.get(key, default) or default

def _pairwise_keys(payloads: Dict[str, Dict[str, Any]]) -> List[Tuple[str,str]]:
    keys = list(payloads.keys())
    pairs = []
    for i in range(len(keys)):
        for j in range(i+1, len(keys)):
            pairs.append((keys[i], keys[j]))
    return pairs

def _jaccard(a: Set[str], b: Set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0

# ---------------------------
# extractors (normalized views)
# ---------------------------

def _candidate_view(doc: Dict[str, Any]) -> Dict[str, Set[str]]:
    cand = _safe(doc, "candidate", {})
    emails = _norm_set(cand.get("emails"))
    phones = set()
    for p in (cand.get("phones") or []):
        # keep digits + leading + optional country; normalize spacing
        phones.add(_norm(p))
    links = set()
    l = cand.get("links") or {}
    for k in ("linkedin", "github"):
        if l.get(k):
            links.add(_norm(l.get(k)))
    return {"emails": emails, "phones": phones, "links": links}

def _experience_view(doc: Dict[str, Any]) -> Dict[str, Set[str]]:
    exps = _safe(doc, "experience", [])
    companies, titles, locs, starts, ends = set(), set(), set(), set(), set()
    for e in exps:
        companies.add(_norm(e.get("company_raw") or e.get("company_norm")))
        titles.add(_norm(e.get("title_raw") or e.get("title_norm")))
        locs.add(_norm(e.get("location")))
        starts.add(_norm(e.get("start_date")))
        ends.add(_norm(e.get("end_date")))
    return {
        "companies": {x for x in companies if x},
        "titles": {x for x in titles if x},
        "locations": {x for x in locs if x},
        "start_dates": {x for x in starts if x},
        "end_dates": {x for x in ends if x},
    }

def _education_view(doc: Dict[str, Any]) -> Dict[str, Set[str]]:
    eds = _safe(doc, "education", [])
    degrees, fields, insts, grad_dates = set(), set(), set(), set()
    for ed in eds:
        degrees.add(_norm(ed.get("degree_raw") or ed.get("degree_level")))
        fields.add(_norm(ed.get("field_raw")))
        insts.add(_norm(ed.get("institution_raw")))
        grad_dates.add(_norm(ed.get("end_date") or ed.get("start_date")))
    return {
        "degrees": {x for x in degrees if x},
        "fields": {x for x in fields if x},
        "institutions": {x for x in insts if x},
        "dates": {x for x in grad_dates if x},
    }

def _projects_view(doc: Dict[str, Any]) -> Dict[str, Set[str]]:
    projs = _safe(doc, "projects", [])
    names = set()
    for p in projs:
        names.add(_norm(p.get("name")))
    return {"project_names": {x for x in names if x}}

def _skills_view(doc: Dict[str, Any]) -> Dict[str, Set[str]]:
    skills = _safe(doc, "skills", [])
    out = set()
    for s in skills:
        if isinstance(s, dict):
            out.add(_norm(s.get("name")))
        else:
            out.add(_norm(s))
    return {"skills": {x for x in out if x}}

# ---------------------------
# field-by-field compare
# ---------------------------

def _compare_field(views: Dict[str, Dict[str, Set[str]]], field: str) -> Dict[str, Any]:
    """
    Given per-system dicts of sets, compute sizes, jaccard overlaps and diffs.
    """
    systems = list(views.keys())
    sizes = {sys: len(views[sys].get(field, set())) for sys in systems}

    pairs = _pairwise_keys(views)
    overlaps = {}
    diffs = {}
    for a, b in pairs:
        A = views[a].get(field, set())
        B = views[b].get(field, set())
        overlaps[f"{a}__{b}"] = round(_jaccard(A, B), 4)
        diffs[f"{a}_minus_{b}"] = sorted(list(A - B))[:20]  # clip to 20 for readability
        diffs[f"{b}_minus_{a}"] = sorted(list(B - A))[:20]

    union_all: Set[str] = set()
    for sys in systems:
        union_all |= views[sys].get(field, set())
    return {"sizes": sizes, "jaccard": overlaps, "sample_diffs": diffs, "union_count": len(union_all)}

def _build_report(payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    # Build normalized views per system
    cand_v = {k: _candidate_view(v) for k, v in payloads.items()}
    exp_v  = {k: _experience_view(v) for k, v in payloads.items()}
    edu_v  = {k: _education_view(v) for k, v in payloads.items()}
    prj_v  = {k: _projects_view(v) for k, v in payloads.items()}
    skl_v  = {k: _skills_view(v) for k, v in payloads.items()}

    # Fieldwise comparisons
    report: Dict[str, Any] = {"systems": list(payloads.keys())}

    # Candidate
    report["candidate"] = {
        "emails": _compare_field(cand_v, "emails"),
        "phones": _compare_field(cand_v, "phones"),
        "links":  _compare_field(cand_v, "links"),
    }

    # Experience
    report["experience"] = {
        "companies":   _compare_field(exp_v, "companies"),
        "titles":      _compare_field(exp_v, "titles"),
        "locations":   _compare_field(exp_v, "locations"),
        "start_dates": _compare_field(exp_v, "start_dates"),
        "end_dates":   _compare_field(exp_v, "end_dates"),
    }

    # Education
    report["education"] = {
        "degrees":       _compare_field(edu_v, "degrees"),
        "fields":        _compare_field(edu_v, "fields"),
        "institutions":  _compare_field(edu_v, "institutions"),
        "dates":         _compare_field(edu_v, "dates"),
    }

    # Projects
    report["projects"] = {
        "names": _compare_field(prj_v, "project_names"),
    }

    # Skills
    report["skills"] = _compare_field(skl_v, "skills")

    # Quick winners (which system has the most for each field)
    winners: Dict[str, Any] = {}
    def _winner_from(field_block: Dict[str, Any], field_name: str):
        sizes = field_block[field_name]["sizes"]
        if sizes:
            winners[field_name] = max(sizes, key=lambda k: sizes[k])

    for f in ("emails", "phones", "links"):
        _winner_from(report["candidate"], f)
    for f in ("companies","titles","locations","start_dates","end_dates"):
        _winner_from(report["experience"], f)
    for f in ("degrees","fields","institutions","dates"):
        _winner_from(report["education"], f)
    _winner_from(report["projects"], "names")
    winners["skills"] = report["skills"]["sizes"] if isinstance(report["skills"], dict) else {}

    report["winners_most_items"] = winners
    return report

# ---------------------------
# CLI
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Compare NER outputs (spaCy vs HF vs Ensemble)")
    ap.add_argument("--spacy", help="Path to *_ner_spacy.json")
    ap.add_argument("--hf",    help="Path to *_ner_hf.json")
    ap.add_argument("--ens",   help="Path to *_ner_ens.json")
    ap.add_argument("--out_json", required=True, help="Where to write comparison JSON")
    args = ap.parse_args()

    payloads: Dict[str, Dict[str, Any]] = {}
    sp = _read(args.spacy)
    hf = _read(args.hf)
    en = _read(args.ens)

    if sp: payloads["spacy"] = sp
    if hf: payloads["hf"] = hf
    if en: payloads["ens"] = en

    if len(payloads) < 2:
        raise SystemExit("Need at least two inputs among --spacy, --hf, --ens")

    report = _build_report(payloads)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[ok] NER comparison → {args.out_json}")

if __name__ == "__main__":
    main()