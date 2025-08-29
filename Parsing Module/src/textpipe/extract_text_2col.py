from __future__ import annotations
import argparse, json, os, re, unicodedata
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional
from typing import Set
import statistics

import fitz  # PyMuPDF

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Line:
    page: int
    text: str
    bbox: Tuple[float, float, float, float]  # x0,y0,x1,y1
    left_indent: float
    width: float
    is_bold: bool
    font_size: float
    column: int = 1  # 1 = left, 2 = right

# -----------------------------
# Normalization helpers
# -----------------------------
_CTRL_RE = re.compile(r"[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F\u0080-\u009F]")
_FA_MAP = {
    # Common FontAwesome glyph names we saw in Rohith's resume dump
    "faAt": "Email:",
    "faPhone": "Phone:",
    "faMapMarker": "Location:",
    "faLinkedin": "LinkedIn:",
}



_BULLET_CHARS = "•·●◦▪▸►▪︎▫︎-–—⋆★☆■□◆▶▪■◆❖»›·•"  # expanded set incl. star/diamond/box arrows

# --- Awards section helpers ---
_AWARDS_HDR_RE = re.compile(r"\bHONORS?\b.*\bAWARDS?\b|\bAWARDS?\b|\bHONORS?\b", re.I)
_AWARD_START_RE = re.compile(r"^(Star Award|Spot Award|Impact Award|[A-Z][A-Za-z0-9 .,&/+-]{0,60}\b(Award|Scholarship|Fellowship|Prize))", re.I)
# Detect common section headers to exit AWARDS scope
_SECTION_BREAK_RE = re.compile(r"^(Education|Experience|Work Experience|Projects|Skills|Certification[s]?|Publications?)\b", re.I)

# --- Header/Company cues (for splitting glued experience headers) ---
_HEADER_CUES = [
    r"Senior", r"Lead", r"Staff", r"Principal", r"Associate", r"Junior",
    r"Software", r"Data", r"Machine", r"AI", r"ML", r"NLP", r"Computer",
    r"Product", r"Engineer", r"Scientist", r"Consultant", r"Analyst", r"Manager",
    r"Decision Scientist", r"R&D", r"Research", r"Architect"
]
_HEADER_START_RE = re.compile(rf"\b(?:{'|'.join(_HEADER_CUES)})\b", re.I)

_COMPANY_CUE_RE  = re.compile(r"\b(Inc\.?|Ltd\.?|LLC|GmbH|Pvt\.?|Technologies|Labs|Analytics|Sigma|Gupshup|Fractal|Mu\s*Sigma)\b", re.I)

# Date cues used to avoid merging headers into bullets
_MONTH_RE = re.compile(r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\b", re.I)
_YEAR_RE  = re.compile(r"\b(19|20)\d{2}\b")

def _looks_like_header_text(txt: str) -> bool:
    if not txt:
        return False
    s = txt.strip()
    if not s:
        return False
    # If the line starts lowercase, treat it as a continuation (not a header)
    if s[:1].islower():
        return False
    # Real headers: month+year, known company tokens, or starts with a Titlecase role cue
    if _MONTH_RE.search(s) and _YEAR_RE.search(s):
        return True
    if _COMPANY_CUE_RE.search(s):
        return True
    # Starts with a known role/level cue in Titlecase only
    if re.match(rf"^(?:{'|'.join(_HEADER_CUES)})\b", s):
        return True
    return False


# Heuristics to decide if a line likely continues the previous bullet
_CONNECTOR_WORDS = {
    "and","or","to","for","with","on","in","at","of","that","which","as",
    "by","from","into","using","via","over","under","between","across","per","vs","vs."
}
_ACRONYM_CONTINUATIONS = {"ai","ml","nlp","cv","sql","rasa","reactjs","python","cpg","etl","kpi","api","kpis"}

def _is_continuation_text(txt: str) -> bool:
    if not txt:
        return False
    s = txt.strip()
    if not s:
        return False
    # starts with lowercase or punctuation typical of continuations
    if s[:1].islower() or s[:1] in {"(", "&", "/", "-", ":"}:
        return True
    # numeric continuation like "12 clients", "3% YOY"
    if re.match(r"^\d", s):
        return True
    # connector words at start
    w0 = s.split()[0].lower()
    if w0 in _CONNECTOR_WORDS:
        return True
    # common acronyms that often continue bullets
    if w0 in _ACRONYM_CONTINUATIONS:
        return True
    return False


def normalize_text(s: str) -> str:
    if not s:
        return ""
    # Unicode normalize to resolve ligatures (e.g., ﬁ → fi)
    s = unicodedata.normalize("NFKC", s)
    # Fix FA tokens that got split (e.g., "faMapMarke r" -> "faMapMarker")
    s = re.sub(r"fa([A-Za-z]{2,})\s+([A-Za-z]{1,})", lambda m: "fa" + m.group(1) + m.group(2), s)
    # Replace stray backslash+space prefix artifacts like "\\ faAt"
    s = re.sub(r"\\\s*(fa[A-Za-z]+)", lambda m: _FA_MAP.get(m.group(1), m.group(1)+":"), s)
    # Replace bare FA tokens at start-of-line
    s = re.sub(r"^(fa[A-Za-z]+)", lambda m: _FA_MAP.get(m.group(1), m.group(1)+":"), s)
    # Also catch leading backslash/space before FA tokens at start of line
    s = re.sub(r"^[\\\s]*(fa[A-Za-z]+)", lambda m: _FA_MAP.get(m.group(1), m.group(1) + ": "), s)

    # Normalize leading exotic bullet glyphs to a standard bullet
    # Covers stars, diamonds, boxes, arrows, cent/lozenge artifacts etc.
    s = re.sub(
        r"^\s*[⋆★☆◆◇■□❖▶▸►▪▫•·●◦»›¤¢§]+\s+",
        "• ",
        s
    )

    # Drop stray 'z' before Month Year (calendar icon artifact)
    s = re.sub(r"^\s*z\s+(?=(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\b)", "", s, flags=re.I)

    # Drop leading '*' before City, State (marker artifact)
    s = re.sub(r"^\s*\*\s+(?=[A-Z][a-z]+, )", "", s)

    # Remove male symbol (and similar) from body text
    s = re.sub(r"[♂]", "", s)

    # Fix spaced URL schemes
    s = re.sub(r"https:\s*//", "https://", s)
    # Normalize spaces around colons/commas introduced by FA mapping
    s = re.sub(r"\s*:\s*", ": ", s)
    s = re.sub(r"\s*,\s*", ", ", s)
    # Remove control chars
    s = _CTRL_RE.sub(" ", s)
    # Collapse spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def merge_wrapped_bullets(lines: List[Line]) -> List[Line]:

    merged: List[Line] = []
    for l in lines:
        t = l.text.strip()
        if merged:
            prev = merged[-1]
            prev_txt = prev.text.strip()
            prev_is_bullet = bool(prev_txt) and prev_txt[:1] in _BULLET_CHARS
            cur_is_bullet = bool(t) and t[:1] in _BULLET_CHARS
            # consider as wrapped continuation
            if prev_is_bullet and not cur_is_bullet and l.page == prev.page and l.column == prev.column:
                # Never merge if the next line looks like a new header/company/date line
                if _looks_like_header_text(t) or l.is_bold:
                    pass  # keep as a new line
                else:
                    # Always merge non-bullet continuations unless header/bold
                    pt = prev.text.rstrip()
                    ct = t.lstrip()
                    if pt.endswith('-'):
                        prev.text = pt[:-1] + ct
                    else:
                        prev.text = pt + " " + ct
                    prev.text = re.sub(r"\s{2,}", " ", prev.text).strip()
                    x0,y0,x1,y1 = prev.bbox
                    _,_,_,ny1 = l.bbox
                    prev.bbox = (x0,y0,x1,max(y1,ny1))
                    continue
        merged.append(l)
    return merged

# -----------------------------
# Education & split-title mergers
# -----------------------------
_DEGREE_RE = re.compile(
    r"^(B\.?\s?Tech\.?|B\.?\s?E\.?|B\.?\s?Sc\.?|M\.?\s?Tech\.?|M\.?\s?S\.?|M\.?\s?Sc\.?|Bachelor(?:'s)?|Master(?:'s)?|Bachelors?|Masters?)\b",
    re.I,
)
_IN_TOKEN_RE = re.compile(r"\b(in|of)\.?$", re.I)
_INSTITUTE_RE = re.compile(r"\b(University|College|Institute|Academy|School|IIT|NIT|SRM|II[Tl]|NIT)\b", re.I)

_TITLE_LEADING_RE = re.compile(r"^(Intern|Trainee|Associate|Senior|Junior|Lead|Staff|Principal)\b", re.I)
_ROLE_START_RE = re.compile(rf"^(?:{'|'.join(_HEADER_CUES)})\b", re.I)

def _merge_education_and_titles(lines: List[Line]) -> List[Line]:

    out: List[Line] = []
    i = 0
    n = len(lines)
    while i < n:
        cur = lines[i]
        t = cur.text.strip()
        merged_line = None

        # --- Case A: Education degree line followed by subject ---
        if _DEGREE_RE.match(t) or _IN_TOKEN_RE.search(t):
            if i + 1 < n:
                nxt = lines[i + 1]
                nt = nxt.text.strip()
                # Do not cross headers or bullets
                is_headerish = _looks_like_header_text(nt) or nxt.is_bold or (nt[:1] in _BULLET_CHARS)
                if (cur.page == nxt.page and cur.column == nxt.column and not is_headerish):
                    # Merge degree + subject
                    merged_text = re.sub(r"\s{2,}", " ", (t + " " + nt)).strip()
                    x0, y0, x1, y1 = cur.bbox
                    nx0, ny0, nx1, ny1 = nxt.bbox
                    new_bbox = (min(x0, nx0), min(y0, ny0), max(x1, nx1), max(y1, ny1))
                    merged_line = Line(**{**cur.__dict__})
                    merged_line.text = merged_text
                    merged_line.bbox = new_bbox
                    i += 2

                    # Optionally merge institution on the next line if it looks like one
                    if i < n:
                        nxt2 = lines[i]
                        nt2 = nxt2.text.strip()
                        if (nxt2.page == cur.page and nxt2.column == cur.column and
                            not (nt2[:1] in _BULLET_CHARS) and
                            not _looks_like_header_text(nt2) and _INSTITUTE_RE.search(nt2)):
                            merged_line.text = re.sub(r"\s{2,}", " ", (merged_line.text + ", " + nt2)).strip()
                            x0, y0, x1, y1 = merged_line.bbox
                            kx0, ky0, kx1, ky1 = nxt2.bbox
                            merged_line.bbox = (min(x0, kx0), min(y0, ky0), max(x1, kx1), max(y1, ky1))
                            i += 1

        if merged_line is None and _TITLE_LEADING_RE.match(t) and len(t.split()) <= 3:
            if i + 1 < n:
                nxt = lines[i + 1]
                nt = nxt.text.strip()
                # Consider a role start if it begins with a header cue, and is not clearly a new section header
                if (cur.page == nxt.page and cur.column == nxt.column and _ROLE_START_RE.match(nt)
                    and not (nt[:1] in _BULLET_CHARS) and not nxt.is_bold and not _looks_like_header_text(nt)):
                    merged_text = re.sub(r"\s{2,}", " ", (t + " " + nt)).strip()
                    x0, y0, x1, y1 = cur.bbox
                    nx0, ny0, nx1, ny1 = nxt.bbox
                    new_bbox = (min(x0, nx0), min(y0, ny0), max(x1, nx1), max(y1, ny1))
                    merged_line = Line(**{**cur.__dict__})
                    merged_line.text = merged_text
                    merged_line.bbox = new_bbox
                    i += 2

        if merged_line is not None:
            out.append(merged_line)
            continue

        out.append(cur)
        i += 1
    return out

# -----------------------------
# Post-processors for header splitting and cleanup
# -----------------------------

def _split_glued_headers(lines: List[Line]) -> List[Line]:
    """If a bullet line contains a glued role/company header (e.g., '... products Data Science Consultant, ...'),
    split into two lines at the header start so the new experience header begins on its own line."""
    out: List[Line] = []
    for l in lines:
        t = l.text
        m = _HEADER_START_RE.search(t)
        if not m:
            out.append(l)
            continue
        idx = m.start()
        # Only split if the header isn't at the very start (we only care about glued headers)
        if idx <= 0:
            out.append(l)
            continue
        prefix = t[:idx].rstrip()
        suffix = t[idx:].lstrip()
        # Require some meaningful text before and after
        if len(prefix) < 5 or len(suffix) < 5:
            out.append(l)
            continue
        # Extra guard: suffix should look like a header/company block
        looks_like_header = _looks_like_header_text(suffix)
        if not looks_like_header:
            out.append(l); continue
        # Emit split lines
        l1 = Line(**{**l.__dict__}); l1.text = prefix
        l2 = Line(**{**l.__dict__}); l2.text = suffix
        out.append(l1)
        out.append(l2)
    return out


def _final_text_cleanup(lines: List[Line]) -> List[Line]:
    for l in lines:
        # As a last resort, map any leading FA tokens with optional leading backslash/space
        l.text = re.sub(r"^[\\\s]*(fa[A-Za-z]+)", lambda m: _FA_MAP.get(m.group(1), m.group(1) + ": "), l.text)
        # Second-chance: normalize any remaining exotic leading bullets
        l.text = re.sub(r"^\s*[⋆★☆◆◇■□❖▶▸►▪▫•·●◦»›¤¢§]+\s+", "• ", l.text)
        # Unwrap hyphenated breaks across variants: '-', '‑' (non-breaking), soft hyphen, optional newline/whitespace
        l.text = re.sub(r"(\w)[\-‑\u00AD]\s*\n?\s*(\w)", r"\1\2", l.text)
        # Known fused token from sample
        l.text = re.sub(r"\bfailedevent\b", "failed event", l.text, flags=re.I)
        # Optional: very common fused adjective
        l.text = re.sub(r"\bperformanceorientated\b", "performance orientated", l.text, flags=re.I)
    return lines

# -----------------------------
# Second-pass bullet block collapse (guarantees bullet reflow)
# -----------------------------

def _collapse_bullet_blocks(lines: List[Line]) -> List[Line]:
    out: List[Line] = []
    in_awards = False
    for l in lines:
        t = l.text.strip()
        # Track whether we're inside an HONORS/AWARDS section
        if _AWARDS_HDR_RE.search(t):
            in_awards = True
        elif _SECTION_BREAK_RE.match(t):
            in_awards = False
        if out:
            prev = out[-1]
            prev_txt = prev.text.strip()
            prev_is_bullet = bool(prev_txt) and prev_txt[:1] in _BULLET_CHARS
            cur_is_bullet = bool(t) and t[:1] in _BULLET_CHARS

            # Treat plain award starters as bullets within HONORS/AWARDS
            if in_awards and not prev_is_bullet and _AWARD_START_RE.match(prev_txt):
                prev_is_bullet = True
                if prev_txt[:1] not in _BULLET_CHARS:
                    prev.text = "• " + prev.text.lstrip()
                    prev_txt = prev.text.strip()
            if in_awards and not cur_is_bullet and _AWARD_START_RE.match(t):
                cur_is_bullet = True
                if t[:1] not in _BULLET_CHARS:
                    # normalize current award header to have a bullet
                    l.text = "• " + t
                    t = l.text.strip()

            if prev_is_bullet and not cur_is_bullet and not _looks_like_header_text(t) and not l.is_bold:
                pt = prev.text.rstrip()
                ct = t.lstrip()
                if pt.endswith('-'):
                    prev.text = pt[:-1] + ct
                else:
                    prev.text = re.sub(r"\s{2,}", " ", (pt + " " + ct)).strip()
                x0, y0, x1, y1 = prev.bbox
                _, _, _, ny1 = l.bbox
                prev.bbox = (x0, y0, x1, max(y1, ny1))
                continue
        # Ensure award-start lines are bullet-prefixed when beginning a new block
        if in_awards and _AWARD_START_RE.match(t) and (not t[:1] in _BULLET_CHARS):
            l.text = "• " + t
        out.append(l)
    return out

# -----------------------------
# PDF parsing helpers
# -----------------------------

DefInterleaveGuard = None

def _interleave_by_y(left: List[Line], right: List[Line]) -> List[Line]:
    i = j = 0
    merged: List[Line] = []
    while i < len(left) and j < len(right):
        if left[i].bbox[1] <= right[j].bbox[1]:
            merged.append(left[i]); i += 1
        else:
            merged.append(right[j]); j += 1
    if i < len(left):
        merged.extend(left[i:])
    if j < len(right):
        merged.extend(right[j:])
    return merged

def _page_lines_from_pdf(page: "fitz.Page") -> List[Line]:
    info = page.get_text("dict")
    lines: List[Line] = []
    for block in info.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            texts, xs, ys, xe, ye = [], [], [], [], []
            bold_votes, sizes = [], []
            for span in line.get("spans", []):
                t = normalize_text(span.get("text", ""))
                if not t:
                    continue
                texts.append(t)
                x0,y0,x1,y1 = span.get("bbox", [0,0,0,0])
                xs.append(x0); ys.append(y0); xe.append(x1); ye.append(y1)
                sizes.append(float(span.get("size", 0)))
                font = (span.get("font", "") or "").lower()
                is_b = "bold" in font or "black" in font or "semibold" in font
                bold_votes.append(1 if is_b else 0)
            if not texts:
                continue
            x0, y0, x1, y1 = min(xs or [0]), min(ys or [0]), max(xe or [0]), max(ye or [0])
            line_text = " ".join(texts).strip()
            lines.append(Line(
                page=page.number + 1,
                text=line_text,
                bbox=(x0,y0,x1,y1),
                left_indent=x0,
                width=(x1-x0),
                is_bold=(sum(bold_votes) >= max(1, len(bold_votes)//2)),
                font_size=(sum(sizes)/max(1, len(sizes)))
            ))
    return lines


def _find_two_col_split(lines: List[Line]) -> Optional[float]:
    """Return x cutoff between two columns if we see a strong bi-modal x-mid gap, else None.
    First try a simple gap-based method; if inconclusive, fall back to a tiny k-means (k=2)
    on x-midpoints and validate separation & cluster balance. This helps catch layouts
    with a narrow right sidebar (e.g., SKILLS/HONORS) that our gap heuristic might miss.
    """
    if len(lines) < 8:
        return None

    xmids = [ (l.bbox[0] + l.bbox[2]) / 2.0 for l in lines ]
    if not xmids:
        return None

    # --- Pass 1: original gap heuristic ---
    mids = sorted(xmids)
    span = max(mids) - min(mids)
    if span <= 0:
        return None
    gaps = [mids[i+1] - mids[i] for i in range(len(mids)-1)]
    if gaps:
        max_gap = max(gaps)
        if span > 0 and (max_gap / span) >= 0.33:
            cut_idx = gaps.index(max_gap)
            return mids[cut_idx]

    # --- Pass 2: tiny k-means on x-midpoints (k=2) ---
    # Initialize centers at robust percentiles to avoid outliers
    lo, hi = statistics.quantiles(mids, n=4)[0], statistics.quantiles(mids, n=4)[-1]
    c1, c2 = float(lo), float(hi)
    for _ in range(10):
        left_pts = [x for x in xmids if abs(x - c1) <= abs(x - c2)]
        right_pts = [x for x in xmids if abs(x - c2) < abs(x - c1)]
        if not left_pts or not right_pts:
            break
        c1_new = sum(left_pts) / len(left_pts)
        c2_new = sum(right_pts) / len(right_pts)
        if abs(c1_new - c1) < 0.5 and abs(c2_new - c2) < 0.5:
            c1, c2 = c1_new, c2_new
            break
        c1, c2 = c1_new, c2_new

    left_pts = [x for x in xmids if abs(x - c1) <= abs(x - c2)]
    right_pts = [x for x in xmids if abs(x - c2) < abs(x - c1)]
    if not left_pts or not right_pts:
        return None

    # Validate: both clusters should have at least 20% of lines
    min_cluster_frac = min(len(left_pts), len(right_pts)) / max(1, len(xmids))
    if min_cluster_frac < 0.2:
        return None

    # Validate: separation should be meaningfully larger than within-cluster spread
    def _spread(vals: List[float]) -> float:
        if len(vals) <= 1:
            return 0.0
        mean_v = sum(vals) / len(vals)
        return (sum((v - mean_v) ** 2 for v in vals) / len(vals)) ** 0.5

    sep = abs(c2 - c1)
    within = _spread(left_pts) + _spread(right_pts) + 1e-6
    separation_ratio = sep / within
    if separation_ratio < 1.2:
        return None

    # Extra hint: if the average width of right cluster lines is much smaller than left,
    # it's likely a sidebar; we still just return a mid cut between centers.
    # Compute average width per cluster using the same assignment
    left_widths, right_widths = [], []
    for l in lines:
        xm = (l.bbox[0] + l.bbox[2]) / 2.0
        if abs(xm - c1) <= abs(xm - c2):
            left_widths.append(l.width)
        else:
            right_widths.append(l.width)
    # Not used directly in the threshold, but helps future tuning/logging if needed
    # left_avg_w = sum(left_widths)/max(1, len(left_widths))
    # right_avg_w = sum(right_widths)/max(1, len(right_widths))

    # Return a cut midway between the two centroids
    return (c1 + c2) / 2.0


def _assign_columns(lines: List[Line], cut_x: Optional[float]) -> None:
    if cut_x is None:
        for l in lines:
            l.column = 1
        return
    for l in lines:
        xm = (l.bbox[0]+l.bbox[2])/2.0
        l.column = 1 if xm <= cut_x else 2


def extract_pdf_two_col(path: str, interleave: bool = False, collect_debug: bool = False) -> Dict[str, Any]:
    doc = fitz.open(path)
    all_lines: List[Line] = []
    pages_text: List[str] = []
    per_page_cuts: Dict[int, Optional[float]] = {}
    per_page_lines: Dict[int, List[Line]] = {}

    for p in range(len(doc)):
        page = doc[p]
        plines = _page_lines_from_pdf(page)
        # find a page-specific split
        cut = _find_two_col_split(plines)
        per_page_cuts[p+1] = cut
        _assign_columns(plines, cut)
        per_page_lines[p+1] = list(plines)
        # deterministic order: left column (y,x), then right column (y,x) per page
        left = [l for l in plines if l.column == 1]
        right = [l for l in plines if l.column == 2]
        left.sort(key=lambda l: (l.bbox[1], l.bbox[0]))
        right.sort(key=lambda l: (l.bbox[1], l.bbox[0]))

        left_merged  = merge_wrapped_bullets(left)
        right_merged = merge_wrapped_bullets(right)

        # Merge education degree+subject and split titles
        left_merged  = _merge_education_and_titles(left_merged)
        right_merged = _merge_education_and_titles(right_merged)

        # Split glued experience headers (title/company) if they appear mid-line after a bullet
        left_clean  = _split_glued_headers(left_merged)
        right_clean = _split_glued_headers(right_merged)

        # Final unwrapping / token cleanup
        left_clean  = _final_text_cleanup(left_clean)
        right_clean = _final_text_cleanup(right_clean)

        ordered = _interleave_by_y(left_clean, right_clean) if interleave else (left_clean + right_clean)
        ordered = _collapse_bullet_blocks(ordered)
        all_lines.extend(ordered)
        pages_text.append("\n".join([l.text for l in ordered]))

    payload = {
        "meta": {"source_path": path, "pages": len(doc), "strategy": "two_column_first", "interleave": interleave},
        "lines": [asdict(l) for l in all_lines],
        "text_per_page": pages_text,
        "text_all": re.sub(r"(^\s*[\u2022\u00B7\u25CF\u25E6\u25AA\-\u2013\u2014]\s.*)\n(?!\s*[\u2022\u00B7\u25CF\u25E6\u25AA\-\u2013\u2014]\s)(?!\s*$)\s*(\S.*)", r"\1 \2", "\n\n".join(pages_text), flags=re.M),
    }
    if collect_debug:
        payload["_debug"] = {
            "per_page_cuts": per_page_cuts,
            "per_page_counts": {pg: {"left": sum(1 for l in per_page_lines.get(pg, []) if l.column==1),
                                      "right": sum(1 for l in per_page_lines.get(pg, []) if l.column==2)} for pg in per_page_lines}
        }
    if collect_debug and "_debug" in payload:
        cuts = payload["_debug"].get("per_page_cuts", {})
        counts = payload["_debug"].get("per_page_counts", {})
        for pg, cut in cuts.items():
            # Heuristic: if cut is None but page has >15 lines and at least 6 with left_indent > median+60, flag it
            page_lines = [l for l in payload["lines"] if l["page"] == pg]
            if not page_lines:
                continue
            meds = statistics.median([ (ln["bbox"][0] + ln["bbox"][2]) / 2.0 for ln in page_lines ])
            far_right = sum(1 for ln in page_lines if ((ln["bbox"][0] + ln["bbox"][2]) / 2.0) > meds + 60)
            if cut is None and len(page_lines) > 15 and far_right >= 6:
                print(f"[warn] page {pg}: no split detected but {far_right} far-right lines found (possible sidebar)")
    return payload


# -----------------------------
# CLI
# -----------------------------

def detect_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".pdf"}:
        return "pdf"
    raise ValueError("This 2-col extractor currently supports only PDF files.")


def main():
    ap = argparse.ArgumentParser(description="Separate extractor tuned for 2-column PDF resumes")
    ap.add_argument("--path", required=True, help="Path to PDF resume")
    ap.add_argument("--out_json", required=True, help="Path to write JSON output")
    ap.add_argument("--out_txt", help="Optional: write plain-text dump")
    ap.add_argument("--debug", action="store_true", help="Print brief per-page column split info")
    ap.add_argument("--interleave", action="store_true", help="Interleave left/right by vertical position instead of left-then-right")
    ap.add_argument("--viz_pdf", help="Optional: write a debug PDF overlay with the column cut and line boxes")
    args = ap.parse_args()

    _ = detect_type(args.path)
    collect_debug = bool(args.debug or args.viz_pdf)
    payload = extract_pdf_two_col(args.path, interleave=args.interleave, collect_debug=collect_debug)

    if collect_debug and "_debug" in payload:
        cuts = payload["_debug"].get("per_page_cuts", {})
        counts = payload["_debug"].get("per_page_counts", {})
        for pg, cut in cuts.items():
            # Heuristic: if cut is None but page has >15 lines and at least 6 with left_indent > median+60, flag it
            page_lines = [l for l in payload["lines"] if l["page"] == pg]
            if not page_lines:
                continue
            meds = statistics.median([ (ln["bbox"][0] + ln["bbox"][2]) / 2.0 for ln in page_lines ])
            far_right = sum(1 for ln in page_lines if ((ln["bbox"][0] + ln["bbox"][2]) / 2.0) > meds + 60)
            if cut is None and len(page_lines) > 15 and far_right >= 6:
                print(f"[warn] page {pg}: no split detected but {far_right} far-right lines found (possible sidebar)")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    if args.out_txt:
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write(payload["text_all"])  # already normalized & merged

    if args.debug:
        # quick visual: count lines per column per page
        by_page: Dict[int, Dict[int, int]] = {}
        for l in payload["lines"]:
            pg = l["page"]; col = l.get("column", 1)
            by_page.setdefault(pg, {}).setdefault(col, 0)
            by_page[pg][col] += 1
        for pg in sorted(by_page.keys()):
            cut_val = payload.get("_debug", {}).get("per_page_cuts", {}).get(pg)
            print(f"[page {pg}] left={by_page[pg].get(1,0)} right={by_page[pg].get(2,0)} cut_x={cut_val}")

    print(f"[ok] 2-col extract → {args.out_json}")
    if args.out_txt:
        print(f"[ok] Plain text → {args.out_txt}")

if __name__ == "__main__":
    main()
