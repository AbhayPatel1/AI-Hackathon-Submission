# src/postprocess/sectioner.py
from __future__ import annotations
import argparse, json, os, re
from typing import List, Dict, Any, Tuple, Optional

from rapidfuzz import fuzz

# ---------- Regex & helpers ----------

EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(
    r"(?:(?:\+?\d{1,3}[\s-]?)?(?:\(?\d{2,4}\)?[\s-]?)?\d{3,4}[\s-]?\d{3,4})"
)
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
LINKEDIN_RE = re.compile(r"(linkedin\.com/[A-Za-z0-9_/\-]+)", re.I)
GITHUB_RE = re.compile(r"(github\.com/[A-Za-z0-9_\-]+)", re.I)

# Long runs of underscores/dashes that decorate headers
HR_RUN_RE = re.compile(r"[_\-]{5,}")

def _clean_header_line(s: str) -> str:
    """Remove decorative runs like ______ or ----- and pipes before header matching."""
    t = re.sub(r"[|]", " ", s)
    t = HR_RUN_RE.sub(" ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _digits_only(s: str) -> str:
    return re.sub(r"[^0-9]", "", s or "")

def _is_year_range(s: str) -> bool:
    # e.g., 2018-2022 / 2018 – 2022 / 2018—2022
    return bool(re.search(r"\b(19|20)\d{2}\s*[-–—]\s*(19|20)\d{2}\b", s))

def _valid_phone(tok: str) -> bool:
    if not tok:
        return False
    if _is_year_range(tok):
        return False
    d = _digits_only(tok)
    if len(d) < 8 or len(d) > 15:
        return False
    # Reject likely IDs of 8–9 digits without separators or prefix
    if len(d) in (8, 9) and not tok.strip().startswith('+'):
        return False
    return True

def _collapse_spaced_caps(s: str) -> str:
    # Collapse sequences like 'A M A N P R E E T' -> 'AMANPREET'
    def repl(m):
        return m.group(0).replace(" ", "")
    return re.sub(r"(?:\b[A-Z]\s+){2,}[A-Z]\b", repl, s)

def _fix_inner_split_caps(s: str) -> str:
    # Fix splits like 'S INGH' -> 'SINGH'
    return re.sub(r"\b([A-Z])\s+([A-Z]{2,})\b", r"\1\2", s)

BULLET_PREFIX_RE = re.compile(r"^\s*(?:[•●◦▪·■‣⁃–—\-*]|(?:\d+|[A-Za-z])[\.)])\s+")

HEADER_SYNONYMS: Dict[str, List[str]] = {
    "work_experience": [
        "work experience", "experience", "professional experience",
        "employment history", "career history", "work history"
    ],
    "internships": [
        "internships", "internship experience", "internship", "industrial training"
    ],
    "projects": ["projects", "personal projects", "academic projects", "project work"],
    "education": ["education", "educational qualification", "academics", "academic profile"],
    "skills": ["skills", "technical skills", "skill set", "core competencies", "software proficiency", "tools"],
    "courses": ["courses", "relevant courses", "coursework", "relevant coursework"],
    "certifications": ["certifications", "certification", "licenses", "licenses & certifications", "certificates"],
    "achievements": ["achievements", "awards", "accomplishments", "honors"],
    "responsibility": [
        "positions of responsibility", "leadership", "responsibilities",
        "position of responsibility", "por", "co-curricular activities", "extra-curricular activities"
    ],
    "interests": ["interests", "hobbies"],
    "summary": ["summary", "professional summary", "objective", "profile"],
}

def _norm_text(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _strip_punct_lower(s: str) -> str:
    return re.sub(r"[^a-z0-9\s]", "", (s or "").lower()).strip()

def _is_allcapsish(s: str) -> bool:
    # "WORK EXPERIENCE" → True, "Work Experience" → False
    letters = [ch for ch in s if ch.isalpha()]
    return bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) >= 0.9

def _is_header_candidate(line: dict, local_median_size: float) -> bool:
    # Heuristics: bold OR (font bigger than local median) OR all-capsish; allow long when underlined/dashed
    text = line["text"]
    has_hr = bool(HR_RUN_RE.search(text))
    if not has_hr and len(text) > 120:  # headers are usually short, but allow long if decorated
        return False
    big = float(line.get("font_size", 0)) >= (local_median_size + 0.8)
    return has_hr or bool(line.get("is_bold")) or big or _is_allcapsish(text)

def _header_to_section_type(text: str) -> Optional[str]:
    # Clean decorative lines and punctuation before matching
    t = _clean_header_line(text)
    t_norm = _strip_punct_lower(t)
    if not t_norm:
        return None

    # Fast path: exact/startswith against synonyms
    for canonical, alts in HEADER_SYNONYMS.items():
        for a in alts:
            if t_norm == a or t_norm.startswith(a):
                return canonical

    # Fuzzy match as fallback
    best = (None, 0)
    for canonical, alts in HEADER_SYNONYMS.items():
        for a in alts:
            score = fuzz.token_set_ratio(t_norm, a)
            if score > best[1]:
                best = (canonical, score)
    return best[0] if best[1] >= 80 else None

def _looks_hr_or_headerish(ln: dict, local_median_size: float) -> bool:
    if HR_RUN_RE.search(ln["text"]):
        return True
    return _is_header_candidate(ln, local_median_size)


def _join_wrapped_lines(lines: List[dict], y_gap_thresh: float = 3.5) -> List[dict]:
    """Join clear continuations but never glue header/HR lines to body."""
    if not lines:
        return []
    med = _median_font_size(lines)
    out = [lines[0].copy()]
    for ln in lines[1:]:
        prev = out[-1]
        gap = ln["bbox"][1] - prev["bbox"][3]
        same_indent = abs(ln["left_indent"] - prev["left_indent"]) <= 4.0
        if _looks_hr_or_headerish(prev, med) or _looks_hr_or_headerish(ln, med):
            out.append(ln.copy())
            continue
        if gap <= y_gap_thresh and same_indent and not BULLET_PREFIX_RE.match(ln["text"]):
            prev["text"] = _norm_text(prev["text"] + " " + ln["text"])
            prev["bbox"] = (prev["bbox"][0], prev["bbox"][1], max(prev["bbox"][2], ln["bbox"][2]), ln["bbox"][3])
        else:
            out.append(ln.copy())
    return out

def _median_font_size(lines: List[dict]) -> float:
    xs = [float(l.get("font_size", 0)) for l in lines if l.get("font_size") is not None]
    if not xs: return 0.0
    xs = sorted(xs)
    mid = len(xs)//2
    return (xs[mid] if len(xs)%2==1 else (xs[mid-1]+xs[mid])/2)

def _page_range(lines: List[dict]) -> List[int]:
    if not lines: return []
    pages = sorted(set(l["page"] for l in lines))
    return [pages[0], pages[-1]] if pages else []

# ---------- Contact extraction ----------

def extract_contacts(lines: List[dict], top_n: int = 12) -> Dict[str, Any]:
    """
    Look only at the top of page-1 for contacts.
    """
    top = [l for l in lines if l["page"] == 1][:top_n]

    emails, phones, links = [], [], {"linkedin": None, "github": None}
    for l in top:
        for m in EMAIL_RE.findall(l["text"]):
            if m not in emails: emails.append(m)
        for m in URL_RE.findall(l["text"]):
            # keep full URL if available
            pass
        li = LINKEDIN_RE.search(l["text"])
        if li:
            links["linkedin"] = li.group(1)
        gh = GITHUB_RE.search(l["text"])
        if gh:
            links["github"] = gh.group(1)
        # phones (filter obvious false positives by length)
        for m in PHONE_RE.findall(l["text"]):
            ms = _norm_text(m)
            if _valid_phone(ms) and ms not in phones:
                phones.append(ms)

    # Fallback: scan entire document if top scan missed items
    if not emails or not phones or (not links.get("linkedin") and not links.get("github")):
        for l in lines:
            if not emails:
                for m in EMAIL_RE.findall(l["text"]):
                    if m not in emails:
                        emails.append(m)
            if not links.get("linkedin"):
                li = LINKEDIN_RE.search(l["text"])
                if li:
                    links["linkedin"] = li.group(1)
            if not links.get("github"):
                gh = GITHUB_RE.search(l["text"]) 
                if gh:
                    links["github"] = gh.group(1)
            if not phones:
                for m in PHONE_RE.findall(l["text"]):
                    ms = _norm_text(m)
                    if _valid_phone(ms) and ms not in phones:
                        phones.append(ms)

    # Name: among the first few lines, take the largest font_size line that is not mostly email/phone/keywords
    name_candidates = top[:5]
    if name_candidates:
        name_line = max(name_candidates, key=lambda l: float(l.get("font_size", 0)))
        raw_name = name_line["text"]
        # scrub email/phone/link tokens from name
        raw_name = EMAIL_RE.sub("", raw_name)
        raw_name = LINKEDIN_RE.sub("", raw_name)
        raw_name = GITHUB_RE.sub("", raw_name)
        raw_name = re.sub(r"\s+\|\s+", " ", raw_name)  # remove pipe-separated labels
        full_name = _norm_text(raw_name)
        full_name = _collapse_spaced_caps(full_name)
        full_name = _fix_inner_split_caps(full_name)
    else:
        full_name = None

    return {
        "full_name": full_name,
        "emails": emails,
        "phones": phones,
        "links": links
    }

# ---------- Section detection & collection ----------

def detect_sections(lines: List[dict]) -> List[Dict[str, Any]]:
    """
    Find headers and collect the lines under each until next header.
    """
    # Clean tiny noise-only lines
    clean = [l for l in lines if _norm_text(l["text"]) not in {"\\", "|"}]

    # Pre-join wrapped lines within each page to stabilize header matching
    by_page: Dict[int, List[dict]] = {}
    for l in clean:
        by_page.setdefault(l["page"], []).append(l)
    joined: List[dict] = []
    for pg, pls in sorted(by_page.items()):
        pls_sorted = sorted(pls, key=lambda l: (l.get("column", 1), l["bbox"][1], l["bbox"][0]))
        pls_joined = _join_wrapped_lines(pls_sorted)
        joined.extend(pls_joined)

    # Identify headers
    headers: List[Tuple[int, str]] = []  # (index in joined, canonical section_type)
    # compute a local median font per small window to detect "bigger than neighbors"
    WINDOW = 10
    for i, ln in enumerate(joined):
        win = joined[max(0, i-WINDOW): i+WINDOW+1]
        med = _median_font_size(win)
        if _is_header_candidate(ln, med):
            sec = _header_to_section_type(ln["text"])
            if sec:
                headers.append((i, sec))

    # Ensure headers are unique-ish in order (dedupe near-duplicates)
    deduped: List[Tuple[int, str]] = []
    last_idx = -999
    for idx, sec in headers:
        if idx - last_idx <= 2 and deduped and deduped[-1][1] == sec:
            # too close & same type → skip
            continue
        deduped.append((idx, sec))
        last_idx = idx

    # Collect sections
    sections: List[Dict[str, Any]] = []
    if not deduped:
        # no headers found: bucket everything into "other"
        big_text = "\n".join(l["text"] for l in joined)
        sections.append({"section_type": "other", "text": big_text, "page_range": _page_range(joined)})
        return sections

    for j, (h_idx, sec_type) in enumerate(deduped):
        start = h_idx + 1
        end = (deduped[j+1][0] if j+1 < len(deduped) else len(joined))
        body = joined[start:end]

        # second pass: join wrapped lines inside the body
        body = _join_wrapped_lines(body)

        # build text
        text = "\n".join(_norm_text(b["text"]) for b in body if _norm_text(b["text"]))
        pr = _page_range(body) if body else [joined[h_idx]["page"], joined[h_idx]["page"]]

        sections.append({
            "section_type": sec_type,
            "text": text,
            "page_range": pr
        })

    # Re-label prose-like skills to other (heuristic)
    def _looks_like_prose(txt: str) -> bool:
        if not txt:
            return False
        # If average line length is long and text contains verbs with past/ing forms, likely prose
        lines = [ln for ln in txt.splitlines() if ln.strip()]
        if not lines:
            return False
        avg_len = sum(len(ln) for ln in lines) / max(1, len(lines))
        has_verbs = bool(re.search(r"\b(implemented|built|developed|designed|evaluated|led|leading|managing|created|optimized|deployed|research(ed|ing))\b", txt, re.I))
        return avg_len > 60 and has_verbs

    for sec in sections:
        if sec.get("section_type") == "skills" and _looks_like_prose(sec.get("text", "")):
            sec["section_type"] = "other"

    return sections

# ---------- Entry points ----------

def process(lines_json_path: str) -> Dict[str, Any]:
    with open(lines_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    lines: List[dict] = payload["lines"]

    candidate = extract_contacts(lines)
    sections = detect_sections(lines)

    out = {
        "meta": {
            "source_path": payload["meta"]["source_path"],
            "pages": payload["meta"]["pages"],
        },
        "candidate": candidate,
        "sectioned_text": sections
    }
    return out

def main():
    ap = argparse.ArgumentParser(description="Step 2: Section & contact collection (text-first)")
    ap.add_argument("--in_json", required=True, help="Input JSON from Step 1 (lines)")
    ap.add_argument("--out_json", required=False, help="Output JSON with candidate + sectioned_text")
    args = ap.parse_args()

    # Derive output filename if not provided
    if args.out_json:
        out_json_path = args.out_json
        out_dir = os.path.dirname(out_json_path)
    else:
        # Use outputs folder as default
        out_dir = "outputs"
        in_base = os.path.splitext(os.path.basename(args.in_json))[0]
        out_json_path = os.path.join(out_dir, f"{in_base}_sections.json")
    os.makedirs(out_dir, exist_ok=True)

    result = process(args.in_json)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[ok] Sectioned output → {out_json_path}")

if __name__ == "__main__":
    main()