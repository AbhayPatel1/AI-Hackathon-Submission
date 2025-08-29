# src/postprocess/sectioner_nli.py
from __future__ import annotations

import argparse
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

from transformers import pipeline
from tqdm import tqdm

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_MODEL = "facebook/bart-large-mnli"

SECTION_LABELS: List[str] = [
    "summary",
    "work_experience",
    "internships",
    "projects",
    "education",
    "skills",
    "achievements",
    "responsibility",
    "courses",
    "certifications",
    "publications",
    "patents",
    "workshops",
    "interests",
    "other",
]

HYPOTHESIS_TEMPLATE = "This line belongs to the '{}' section."


BULLET_ONLY = {"•", "●", "◦", "▪", "·", "-", "–", "—"}

# Canonical header name mapping → internal section labels
HEADER_CANON = {
    # Work Experience
    "experience": "work_experience",
    "work experience": "work_experience",
    "professional experience": "work_experience",
    "employment": "work_experience",
    "work history": "work_experience",
    "career history": "work_experience",
    "professional background": "work_experience",
    "career experience": "work_experience",
    "job experience": "work_experience",

    # Internships
    "internship": "internships",
    "internships": "internships",
    "internship experience": "internships",
    "training": "internships",

    # Projects
    "projects": "projects",
    "personal projects": "projects",
    "academic projects": "projects",
    "notable projects": "projects",

    # Education
    "education": "education",
    "academics": "education",
    "academic background": "education",
    "educational qualifications": "education",
    "qualifications": "education",

    # Skills
    "skills": "skills",
    "technical skills": "skills",
    "key skills": "skills",
    "professional skills": "skills",
    "core competencies": "skills",
    "competencies": "skills",
    "skill set": "skills",
    "tools & technologies": "skills",
    "technology stack": "skills",
    "tech skills": "skills",
    "technologies": "skills",

    # Achievements / Awards
    "honors": "achievements",
    "honours": "achievements",
    "awards": "achievements",
    "honors & awards": "achievements",
    "honours & awards": "achievements",
    "recognitions": "achievements",
    "achievements": "achievements",

    # Certifications
    "certifications": "certifications",
    "certificates": "certifications",
    "licenses": "certifications",
    "accreditations": "certifications",

    # Publications
    "publications": "publications",
    "research papers": "publications",
    "research publications": "publications",

    # Patents
    "patents": "patents",
    "intellectual property": "patents",

    # Workshops
    "workshops": "workshops",
    "seminars": "workshops",
    "training programs": "workshops",

    # Interests
    "interests": "interests",
    "hobbies": "interests",
    "personal interests": "interests",

    # Responsibilities
    "responsibilities": "responsibility",
    "positions of responsibility": "responsibility",
    "leadership roles": "responsibility",
    "leadership positions": "responsibility",
    "roles and responsibilities": "responsibility"
}

# -----------------------------------------------------------------------------
# Regex & small helpers
# -----------------------------------------------------------------------------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
LINKEDIN_RE = re.compile(r"(linkedin\.com/[A-Za-z0-9_/\-]+)", re.I)
GITHUB_RE = re.compile(r"(github\.com/[A-Za-z0-9_\-]+)", re.I)

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _digits_only(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

def _is_year_range(s: str) -> bool:
    return bool(re.search(r"\b(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}\b", s or ""))

def _valid_phone(tok: str) -> bool:
    if not tok:
        return False
    if _is_year_range(tok):
        return False
    d = _digits_only(tok)
    if len(d) < 8 or len(d) > 15:
        return False
    if len(d) in (8, 9) and not tok.strip().startswith("+"):
        return False
    return True

def _normalize_phone(tok: str, country_hint: Optional[str] = None) -> str:
    d = _digits_only(tok)
    if country_hint == "IN" and not tok.strip().startswith("+") and len(d) == 10:
        return "+91 " + d
    plus = "+" if tok.strip().startswith("+") else ""
    return (plus + d).strip()

def _is_allcapsish(s: str) -> bool:
    letters = [c for c in (s or "") if c.isalpha()]
    return bool(letters) and sum(c.isupper() for c in letters) / len(letters) >= 0.9

def _median_font_size(lines: List[dict]) -> float:
    xs = [float(l.get("font_size", 0) or 0) for l in lines if l.get("font_size") is not None]
    if not xs:
        return 0.0
    xs.sort()
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 == 1 else (xs[m - 1] + xs[m]) / 2.0

def _looks_headerish(line: dict, local_med: float) -> bool:
    text = line.get("text", "")
    bold = bool(line.get("is_bold"))
    big = float(line.get("font_size", 0) or 0.0) >= (local_med + 0.8)
    return bold or big or _is_allcapsish(text)

# -----------------------------------------------------------------------------
# Contacts (top anchors)
# -----------------------------------------------------------------------------

def extract_contacts(lines: List[dict], top_n: int = 12) -> Dict[str, Any]:
    """
    High-precision contact extraction from top-of-page lines with whole-doc fallback.
    """
    top = [l for l in lines if int(l.get("page", 0)) == 1][:top_n]

    emails: List[str] = []
    phones: List[str] = []
    links: Dict[str, Optional[str]] = {"linkedin": None, "github": None}

    # Top-first scan
    for l in top:
        t = l.get("text", "")
        for m in EMAIL_RE.findall(t):
            if m not in emails:
                emails.append(m)
        for m in PHONE_RE.findall(t):
            ms = _norm(m)
            if _valid_phone(ms):
                p = _normalize_phone(ms, country_hint="IN")
                if p not in phones:
                    phones.append(p)
        li = LINKEDIN_RE.search(t)
        if li:
            links["linkedin"] = li.group(1)
        gh = GITHUB_RE.search(t)
        if gh:
            links["github"] = gh.group(1)

    # Whole-doc fallback if anything is still missing
    if not emails or not phones or (not links.get("linkedin") and not links.get("github")):
        for l in lines:
            t = l.get("text", "")
            if not emails:
                for m in EMAIL_RE.findall(t):
                    if m not in emails:
                        emails.append(m)
            if not phones:
                for m in PHONE_RE.findall(t):
                    ms = _norm(m)
                    if _valid_phone(ms):
                        p = _normalize_phone(ms, country_hint="IN")
                        if p not in phones:
                            phones.append(p)
            if not links.get("linkedin"):
                li = LINKEDIN_RE.search(t)
                if li:
                    links["linkedin"] = li.group(1)
            if not links.get("github"):
                gh = GITHUB_RE.search(t)
                if gh:
                    links["github"] = gh.group(1)

    # Name = largest-font among the first few lines on page 1
    name_candidates = top[:5]
    if name_candidates:
        med = _median_font_size(name_candidates)
        name_line = max(name_candidates, key=lambda l: float(l.get("font_size", med) or med))
        raw = name_line.get("text", "")
        # scrub obvious contact tokens & pipes
        raw = EMAIL_RE.sub("", raw)
        raw = LINKEDIN_RE.sub("", raw)
        raw = GITHUB_RE.sub("", raw)
        raw = re.sub(r"\s+\|\s+", " ", raw)
        full_name = _norm(raw)
        # fix common spacing issues (e.g., A M A N → AMAN; S INGH → SINGH)
        full_name = re.sub(r"(?:\b[A-Z]\s+){2,}[A-Z]\b", lambda m: m.group(0).replace(" ", ""), full_name)
        full_name = re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", full_name)
        if full_name.isupper():
            full_name = " ".join(w.title() if len(w) > 2 else w for w in full_name.split())
    else:
        full_name = None

    return {"full_name": full_name, "emails": emails, "phones": phones, "links": links}

# -----------------------------------------------------------------------------
# NLI classification
# -----------------------------------------------------------------------------

def build_classifier(model_name: str = DEFAULT_MODEL, device: Optional[int] = None):
    """
    Zero-shot classifier pipeline. device: None -> CPU (-1); or pass GPU id int.
    """
    return pipeline(
        "zero-shot-classification",
        model=model_name,
        device=-1 if device is None else device
    )

def classify_lines_nli(
    clf,
    lines: List[dict],
    labels: List[str],
    hypothesis_template: str,
    min_conf: float = 0.45,
) -> List[Tuple[str, float]]:
    """
    Classify each line to a label; return (label, score) per line.
    """
    results: List[Tuple[str, float]] = []
    B = 16  # batch size
    for i in tqdm(range(0, len(lines), B), desc="NLI classify", leave=False):
        batch = lines[i:i+B]
        texts = [l.get("text", "") or "" for l in batch]
        out = clf(
            sequences=texts,
            candidate_labels=labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )
        if isinstance(out, dict):
            out = [out]
        for o in out:
            if not o or "labels" not in o or "scores" not in o:
                results.append(("other", 0.0))
                continue
            lbl = o["labels"][0]
            sc = float(o["scores"][0])
            if sc < min_conf:
                results.append(("other", sc))
            else:
                results.append((lbl, sc))
    return results

# -----------------------------------------------------------------------------
# Smoothing & grouping
# -----------------------------------------------------------------------------

def smooth_labels(
    lines: List[dict],
    preds: List[Tuple[str, float]],
    win: int = 2,
    header_boost: float = 0.15,
) -> List[str]:
    """
    Simple smoothing:
    - header-like lines (bold/big/ALLCAPS) get a confidence boost
    - low-confidence lines adopt the local window majority label
    - discourages jittering without hard-coded rules
    """
    n = len(lines)
    labels = [p[0] for p in preds]
    scores = [p[1] for p in preds]

    # header boost (local median font)
    for i in range(n):
        w = lines[max(0, i-5): i+6]
        med = _median_font_size(w)
        if _looks_headerish(lines[i], med):
            scores[i] = min(1.0, scores[i] + header_boost)

    smoothed: List[str] = []
    for i in range(n):
        l_i, s_i = labels[i], scores[i]
        if s_i < 0.45:
            left = labels[max(0, i - win):i]
            right = labels[i + 1:min(n, i + 1 + win)]
            pool = left + right + [l_i]
            if pool:
                maj = max(set(pool), key=pool.count)
                l_i = maj
        # discourage single-line flips
        if i >= 1 and i + 1 < n and l_i != smoothed[-1] and labels[i+1] == smoothed[-1] and s_i < 0.55:
            l_i = smoothed[-1]
        smoothed.append(l_i)
    return smoothed

def collect_sections(lines: List[dict], labels: List[str]) -> List[Dict[str, Any]]:
    """
    Group contiguous lines with the same section label into blocks,
    drop single-glyph bullet lines, compute page ranges, and merge adjacent same-type.
    """
    sections: List[Dict[str, Any]] = []
    cur_type: Optional[str] = None
    cur_lines: List[dict] = []

    def flush():
        nonlocal cur_type, cur_lines, sections
        if not cur_lines:
            return
        # build text excluding single-glyph bullets and empty lines
        text_lines = []
        for l in cur_lines:
            t = _norm(l.get("text", ""))
            if not t or t in BULLET_ONLY:
                continue
            text_lines.append(t)
        text = "\n".join(text_lines).strip()
        if not text:
            cur_type, cur_lines = None, []
            return
        pages = sorted(set(int(l.get("page", 0)) for l in cur_lines))
        pr = [pages[0], pages[-1]] if pages else []
        sections.append({"section_type": cur_type, "text": text, "page_range": pr})
        cur_type, cur_lines = None, []

    for l, lab in zip(lines, labels):
        if lab != cur_type:
            flush()
            cur_type = lab
            cur_lines = [l]
        else:
            cur_lines.append(l)
    flush()

    # merge adjacent same-type
    merged: List[Dict[str, Any]] = []
    for sec in sections:
        if merged and merged[-1]["section_type"] == sec["section_type"]:
            merged[-1]["text"] = (merged[-1]["text"] + "\n" + sec["text"]).strip()
            pr, cr = merged[-1]["page_range"], sec["page_range"]
            merged[-1]["page_range"] = [min(pr[0], cr[0]), max(pr[1], cr[1])]
        else:
            merged.append(sec)
    return merged

# -----------------------------------------------------------------------------
# Post-processing helpers (header merge, adjacent-other relabel, footer strip)
# -----------------------------------------------------------------------------

def _splitlines_nonempty(text: str) -> List[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln and ln.strip()]


DATE_RANGE_RE = re.compile(r"\b(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}\b")

def _infer_header_label(text: str) -> Optional[str]:
    """Infer a canonical section label from a likely header line."""
    t = (text or "").strip().lower()
    # remove common decoration characters
    t = re.sub(r"[:|•\-–—\s]+$", "", t)
    # normalize ampersands and extra spaces
    t = t.replace("&amp;", "&").replace("  ", " ").strip()
    # exact match
    if t in HEADER_CANON:
        return HEADER_CANON[t]
    # fuzzy contains e.g., "honors & awards", "work experience"
    for k, v in HEADER_CANON.items():
        if k in t:
            return v
    return None

def _is_date_only_block(text: str) -> Optional[bool]:
    """True if the block is just a date range (possibly with a leading bullet or 'z ')."""
    if not text:
        return False
    t = _norm(text)
    # allow optional leading marker like 'z' or '*' then a date range
    t_ = re.sub(r"^[\*\-–—•z]\s*", "", t, flags=re.I)
    return bool(DATE_RANGE_RE.search(t_)) and len(_splitlines_nonempty(text)) == 1

def _looks_awards_block(text: str) -> bool:
    """Heuristic: block with bullets and the word 'award' likely belongs to achievements."""
    if not text:
        return False
    tl = text.lower()
    if "award" in tl or "awards" in tl or "honors" in tl or "honours" in tl:
        # either multiple bullets or multiple award-like lines
        lines = _splitlines_nonempty(text)
        bul = sum(1 for ln in lines if ln.lstrip().startswith(("•", "-", "–", "—", "*")))
        return bul >= 1 or len(lines) >= 2
    return False

def _looks_headerish_text(line_text: str) -> bool:
    """A single-line block likely to be a header: short (<=4 words) and all-caps-ish or decorated."""
    t = (line_text or "").strip()
    if not t:
        return False
    words = t.split()
    if len(words) <= 4 and _is_allcapsish(t):
        return True
    # trailing underscores/dashes lines often come through from PDFs
    if re.search(r"[_\-]{4,}$", t):
        return True
    return False


def _looks_bullet_or_date_block(text: str) -> bool:
    lines = _splitlines_nonempty(text)
    if not lines:
        return False
    # bullety start or has clear date range
    if any(ln.lstrip().startswith(("•", "-", "–", "—", "*")) for ln in lines[:3]):
        return True
    if any(DATE_RANGE_RE.search(ln) for ln in lines[:5]):
        return True
    return False

# --------------------------------------------------------------------------
# Inline header splitting
# --------------------------------------------------------------------------
def _split_on_inline_header(sec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    If a multi-line block contains a header-like line (e.g., 'EXPERIENCE'),
    split into two blocks at that line. The tail adopts the inferred label.
    """
    text = sec.get("text", "") or ""
    lines = _splitlines_nonempty(text)
    if len(lines) <= 1:
        return [sec]

    # find first header-ish line that we can map to a canonical label
    for idx, ln in enumerate(lines):
        inferred = _infer_header_label(ln)
        if inferred:
            # if header is the only line, original logic will handle it; skip here
            if idx == 0 and len(lines) == 1:
                return [sec]
            head = "\n".join(lines[:idx]).strip()
            tail = "\n".join(lines[idx:]).strip()

            left = {**sec, "text": head} if head else None
            right = {**sec, "text": tail, "section_type": inferred}

            parts: List[Dict[str, Any]] = []
            if left and _splitlines_nonempty(left["text"]):
                parts.append(left)
            if right and _splitlines_nonempty(right["text"]):
                parts.append(right)
            return parts
    return [sec]

def _strip_footer_contacts(text: str, candidate: Dict[str, Any]) -> str:
    lines = [ln for ln in (text or "").splitlines()]
    kept = []
    for ln in lines:
        found = False
        for m in EMAIL_RE.findall(ln):
            if m not in candidate["emails"]:
                candidate["emails"].append(m)
                found = True
        for m in PHONE_RE.findall(ln):
            ms = _norm(m)
            if _valid_phone(ms):
                p = _normalize_phone(ms, country_hint="IN")
                if p not in candidate["phones"]:
                    candidate["phones"].append(p)
                    found = True
        li = LINKEDIN_RE.search(ln)
        if li and not candidate["links"].get("linkedin"):
            candidate["links"]["linkedin"] = li.group(1)
            found = True
        gh = GITHUB_RE.search(ln)
        if gh and not candidate["links"].get("github"):
            candidate["links"]["github"] = gh.group(1)
            found = True
        if URL_RE.search(ln):
            # keep generic URLs in links_other
            candidate.setdefault("links_other", [])
            for u in URL_RE.findall(ln):
                if u not in candidate["links_other"]:
                    candidate["links_other"].append(u)
                    found = True
        if not found:
            kept.append(ln)
    return "\n".join(kept).strip()

def postprocess_sections(sections: List[Dict[str, Any]], candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Apply header-merge, adjacent-other relabel, footer-strip, and small cleanups."""
    if not sections:
        return sections

    # 0) If a block looks like a single-line header, try to infer its label by text
    tmp: List[Dict[str, Any]] = []
    for sec in sections:
      lines = _splitlines_nonempty(sec.get("text", ""))
      if len(lines) == 1 and _looks_headerish_text(lines[0]):
          inferred = _infer_header_label(lines[0])
          if inferred:
              sec = {**sec, "section_type": inferred}
      tmp.append(sec)
    sections = tmp

    # 1) Relabel 'other' immediately after a strong section if it looks like bullets/dates
    strong = {"work_experience", "projects", "education", "internships"}
    out: List[Dict[str, Any]] = []
    for i, sec in enumerate(sections):
        prev = out[-1] if out else None
        if sec.get("section_type") == "other" and prev and prev.get("section_type") in strong:
            if _looks_bullet_or_date_block(sec.get("text", "")):
                sec = {**sec, "section_type": prev["section_type"]}
        out.append(sec)

    sections = out

    # 2) Collapse tiny header-only blocks forward into the next block
    collapsed: List[Dict[str, Any]] = []
    skip_next = False
    for i, sec in enumerate(sections):
        if skip_next:
            skip_next = False
            continue
        lines = _splitlines_nonempty(sec.get("text", ""))
        if len(lines) == 1 and _looks_headerish_text(lines[0]) and i + 1 < len(sections):
            nxt = sections[i + 1].copy()
            # if next is 'other' but header has a concrete type, adopt header's type
            if nxt.get("section_type") == "other" and sec.get("section_type") != "other":
                nxt["section_type"] = sec["section_type"]
            collapsed.append(nxt)
            skip_next = True
        else:
            collapsed.append(sec)

    sections = collapsed

    # 2aa) Split blocks that contain inline headers (e.g., '...\nEXPERIENCE\n...')
    split_blocks: List[Dict[str, Any]] = []
    for sec in sections:
        split_blocks.extend(_split_on_inline_header(sec))
    sections = split_blocks

    # 2a) Retype awards-like blocks into 'achievements' if misclassified
    fixed_types: List[Dict[str, Any]] = []
    for sec in sections:
        if sec.get("section_type") in {"responsibility", "other"} and _looks_awards_block(sec.get("text", "")):
            fixed_types.append({**sec, "section_type": "achievements"})
        else:
            fixed_types.append(sec)
    sections = fixed_types

    # 2c) If a single-line 'other' looks like only a date range, attach it to the previous labeled section
    attached: List[Dict[str, Any]] = []
    for i, sec in enumerate(sections):
        if (
            sec.get("section_type") == "other"
            and _is_date_only_block(sec.get("text", ""))
            and attached
            and attached[-1].get("section_type") in {"education", "work_experience", "internships", "projects"}
        ):
            # adopt previous label; final merge step will merge text & page ranges
            sec = {**sec, "section_type": attached[-1]["section_type"]}
        attached.append(sec)
    sections = attached

    # 2b) Absorb runs of 'other' between labeled sections into the previous labeled section
    absorbed: List[Dict[str, Any]] = []
    i = 0
    while i < len(sections):
        cur = sections[i]
        if cur.get("section_type") != "other":
            absorbed.append(cur)
            i += 1
            continue

        # cur is 'other' — gather the consecutive run
        j = i
        run: List[Dict[str, Any]] = []
        while j < len(sections) and sections[j].get("section_type") == "other":
            run.append(sections[j])
            j += 1

        left = absorbed[-1] if absorbed else None
        right = sections[j] if j < len(sections) else None

        # Only absorb if truly between two labeled sections
        if left and left.get("section_type") != "other" and right and right.get("section_type") != "other":
            # Body-like heuristic: bullets, dates, or multi-line text in the run
            body_like = any(
                _looks_bullet_or_date_block(r.get("text", "")) or len(_splitlines_nonempty(r.get("text", ""))) >= 2
                for r in run
            )
            if body_like:
                for r in run:
                    absorbed.append({**r, "section_type": left["section_type"]})
            else:
                absorbed.extend(run)
        else:
            absorbed.extend(run)

        i = j

    sections = absorbed

    # 3) Name-line guard: drop/convert misclassified name-only blocks (e.g., PATENTS)
    if candidate.get("full_name"):
        fn = candidate["full_name"].replace(" ", "").lower()
        fixed: List[Dict[str, Any]] = []
        for i, sec in enumerate(sections):
            lines = _splitlines_nonempty(sec.get("text", ""))
            if i == 0 and len(lines) == 1:
                t = lines[0].replace(" ", "").lower()
                if fn and (t == fn or t in fn or fn in t) and sec.get("section_type") in {"patents", "publications", "other"}:
                    # convert to a tiny summary line instead of a wrong section
                    fixed.append({"section_type": "summary", "text": sections[i].get("text", ""), "page_range": sections[i].get("page_range", [])})
                    continue
            fixed.append(sec)
        sections = fixed

    # 4) Strip footer-like contact lines out of sections and promote to candidate
    for sec in sections:
        sec["text"] = _strip_footer_contacts(sec.get("text", ""), candidate)

    # 5) Merge adjacent same-type after edits
    merged: List[Dict[str, Any]] = []
    for sec in sections:
        if merged and merged[-1]["section_type"] == sec["section_type"]:
            merged[-1]["text"] = (merged[-1]["text"] + "\n" + sec["text"]).strip()
            pr, cr = merged[-1]["page_range"], sec["page_range"]
            merged[-1]["page_range"] = [min(pr[0], cr[0]), max(pr[1], cr[1])]
        else:
            merged.append(sec)

    # 6) Drop empty sections after stripping
    merged = [s for s in merged if _splitlines_nonempty(s.get("text", ""))]

    return merged

# -----------------------------------------------------------------------------
# Main process
# -----------------------------------------------------------------------------

def process(
    in_json: str,
    model_name: str = DEFAULT_MODEL,
    device: Optional[int] = None,
    min_conf: float = 0.45,
) -> Dict[str, Any]:
    with open(in_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    lines: List[dict] = payload["lines"]

    # 1) Contacts
    candidate = extract_contacts(lines)

    # 2) Group by explicit headers first
    grouped_sections = []
    current_header = None
    buffer: List[str] = []
    page_buffer: List[int] = []

    # Refined header grouping: all lines after a header are grouped into that section until another header is found.
    for l in lines:
        txt = _norm(l.get("text", ""))
        if not txt:
            continue
        header_label = _infer_header_label(txt)
        if header_label:
            if buffer and current_header:
                grouped_sections.append({
                    "section_type": current_header,
                    "text": "\n".join(buffer).strip(),
                    "page_range": [min(page_buffer), max(page_buffer)] if page_buffer else []
                })
            buffer, page_buffer = [], []
            current_header = header_label
            continue
        # always append to current header until next header
        if current_header:
            buffer.append(txt)
            page_buffer.append(int(l.get("page", 0)))

    if buffer and current_header:
        grouped_sections.append({
            "section_type": current_header,
            "text": "\n".join(buffer).strip(),
            "page_range": [min(page_buffer), max(page_buffer)] if page_buffer else []
        })

    # If no headers detected at all, run NLI across all lines
    if not grouped_sections:
        clf = build_classifier(model_name=model_name, device=device)
        preds = classify_lines_nli(
            clf, lines, SECTION_LABELS, HYPOTHESIS_TEMPLATE, min_conf=min_conf
        )
        labels = [p[0] for p in preds]
        grouped_sections = collect_sections(lines, labels)

    # 4) Post-process sections
    sections = postprocess_sections(grouped_sections, candidate)

    return {
        "meta": {
            "source_path": payload["meta"]["source_path"],
            "pages": payload["meta"]["pages"],
            "model": model_name,
        },
        "candidate": candidate,
        "sectioned_text": sections,
        "stats": {
            "sections": len(sections),
            "contacts_found": {
                "name": bool(candidate.get("full_name")),
                "email": bool(candidate.get("emails")),
                "phone": bool(candidate.get("phones")),
            },
        },
    }

# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def _derive_default_out(in_json: str, out_json: Optional[str]) -> str:
    if out_json:
        return out_json
    base = os.path.splitext(os.path.basename(in_json))[0]
    out_dir = os.path.join("outputs")
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, f"{base}_sections_nli.json")

def main():
    ap = argparse.ArgumentParser(description="NLI-based sectioner (zero-shot)")
    ap.add_argument("--in_json", required=True, help="Input *_lines.json from Step 1")
    ap.add_argument("--out_json", help="Output JSON; default: outputs/<in>_sections_nli.json")
    ap.add_argument("--model_name", default=DEFAULT_MODEL, help="HF model id (default: facebook/bart-large-mnli)")
    ap.add_argument("--device", type=int, default=None, help="GPU id, or omit for CPU")
    ap.add_argument("--min_conf", type=float, default=0.45, help="Min confidence to accept label; else 'other'")
    ap.add_argument("--debug", action="store_true", help="Print a console summary of detected sections and contacts")
    args = ap.parse_args()

    result = process(
        in_json=args.in_json,
        model_name=args.model_name,
        device=args.device,
        min_conf=args.min_conf,
    )
    if args.debug:
        cand = result.get("candidate", {})
        secs = result.get("sectioned_text", []) or []
        # per-type counts
        from collections import Counter
        c = Counter(s.get("section_type", "other") for s in secs)
        print("\n[debug] Candidate:")
        print(f"  name   : {cand.get('full_name')}")
        print(f"  emails : {', '.join(cand.get('emails') or [])}")
        print(f"  phones : {', '.join(cand.get('phones') or [])}")
        lk = cand.get("links") or {}
        print(f"  linkedin: {lk.get('linkedin')}, github: {lk.get('github')}")
        print("\n[debug] Sections summary:")
        for k in SECTION_LABELS:
            if c.get(k):
                print(f"  - {k}: {c[k]} block(s)")
        # preview first line of each block
        print("\n[debug] Blocks preview:")
        for i, s in enumerate(secs, 1):
            t = s.get('text', '')
            first = (t.splitlines()[0] if t else '')[:120]
            pr = s.get('page_range', [])
            print(f"  [{i:02d}] {s.get('section_type')}  pages={pr}  text='{first}'")
        print("")
    out_path = _derive_default_out(args.in_json, args.out_json)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ok] NLI sectioned output → {out_path}")

if __name__ == "__main__":
    main()