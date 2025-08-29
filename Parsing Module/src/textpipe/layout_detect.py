# src/textpipe/layout_detect.py
from __future__ import annotations
from typing import Dict, Any, Optional, List, Tuple
import fitz  # PyMuPDF
import statistics

# We import the page line builder and the split finder from your 2-col module
from .extract_text_2col import _page_lines_from_pdf, _find_two_col_split, Line

def detect_resume_layout(pdf_path: str, sample_pages: Optional[int] = None, debug: bool = False) -> Dict[str, Any]:
    """
    Decide if a PDF resume is single-column or two-column by probing pages
    with the existing _find_two_col_split() heuristic.
    Returns: {"layout": "single"|"two_column", "confidence": float, "pages": N, "cuts_found": K}
    """
    doc = fitz.open(pdf_path)
    total_pages = len(doc)
    pages_to_check = list(range(total_pages))
    if sample_pages and sample_pages < total_pages:
        # sample first N and last page to avoid missing mixed layouts
        pages_to_check = sorted(set(list(range(min(sample_pages, total_pages))) + [total_pages-1]))

    cuts = 0
    right_heavy_pages = 0
    checked = 0

    for p in pages_to_check:
        page = doc[p]
        plines: List[Line] = _page_lines_from_pdf(page)
        # If very few lines, skip (cover pages / image-only pages)
        if len(plines) < 8:
            continue

        cut = _find_two_col_split(plines)
        checked += 1
        if cut is None:
            continue

        # --- Per-page validation: cluster balance and column width balance ---
        xmids = [ (l.bbox[0] + l.bbox[2]) / 2.0 for l in plines ]
        left_idxs  = [i for i,x in enumerate(xmids) if x <= cut]
        right_idxs = [i for i,x in enumerate(xmids) if x >  cut]
        if not left_idxs or not right_idxs:
            continue

        min_cluster_frac = min(len(left_idxs), len(right_idxs)) / max(1, len(plines))
        if min_cluster_frac < 0.30:  # stricter than k-means default
            # Treat as single-column-like page; ignore this cut
            continue

        # Compute visual width of each side to avoid skinny-label vs body-text false positives
        def side_width(idxs):
            x0s = [plines[i].bbox[0] for i in idxs]
            x1s = [plines[i].bbox[2] for i in idxs]
            return (max(x1s) - min(x0s)) if x0s and x1s else 0.0
        left_w  = side_width(left_idxs)
        right_w = side_width(right_idxs)
        min_w, max_w = (min(left_w, right_w), max(left_w, right_w))
        if max_w <= 0 or min_w < 0.25 * max_w:
            # One side too skinny compared to the other — likely single column with labels/margins
            continue

        # Passed per-page checks — count this page as a valid two-column hit
        cuts += 1

        right_frac = len(right_idxs) / max(1, len(plines))
        if right_frac >= 0.20:
            right_heavy_pages += 1

        if debug:
            print(f"[layout] page={p+1} cut_x={cut:.1f} right_frac={right_frac:.2f} lines={len(plines)} left_w={left_w:.1f} right_w={right_w:.1f} min_cluster_frac={min_cluster_frac:.2f}")

    if checked == 0:
        # not enough textual content; default to single
        return {"layout": "single", "confidence": 0.5, "pages": total_pages, "cuts_found": 0}

    # Decisions:
    # - strong two-column if ≥60% of probed pages have a valid cut AND the right column actually carries content on ≥1 page
    # - else single
    frac_cut = cuts / checked
    two_col_like = (frac_cut >= 0.60) and (right_heavy_pages >= 1)

    confidence = 0.5 + 0.5 * min(1.0, frac_cut)  # simple monotone mapping
    if two_col_like:
        return {"layout": "two_column", "confidence": confidence, "pages": total_pages, "cuts_found": cuts}
    return {"layout": "single", "confidence": 1.0 - confidence/2, "pages": total_pages, "cuts_found": cuts}