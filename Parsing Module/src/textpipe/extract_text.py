from __future__ import annotations
import argparse, json, os, re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Any, Optional

# PDF/DOC support only
import fitz  # PyMuPDF
from docx import Document as DocxDocument

# --- Regex for dates and locations ---
DATE_RE = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}"
    r"(?:\s*[-–]\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}|"
    r"\s*[-–]\s*(?:Present|Current|present|current))?"
    r"|\d{4}(?:\s*[-–]\s*(?:\d{4}|Present|Current|present|current))?)"
)

LOCATION_RE = re.compile(r".+,\s*(?:[A-Z]{2}|[A-Za-z]+)(?:\s*/\s*.+)?$")


@dataclass
class Line:
    page: int
    text: str
    bbox: Tuple[float, float, float, float]
    left_indent: float
    width: float
    is_bold: bool
    font_size: float


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


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
                t = _norm_space(span.get("text", ""))
                if not t:
                    continue
                texts.append(t)
                x0, y0, x1, y1 = span.get("bbox", [0, 0, 0, 0])
                xs.append(x0)
                ys.append(y0)
                xe.append(x1)
                ye.append(y1)
                sizes.append(float(span.get("size", 0)))
                font = (span.get("font", "") or "").lower()
                is_b = "bold" in font or "black" in font or "semibold" in font
                bold_votes.append(1 if is_b else 0)

            if not texts:
                continue

            x0, y0, x1, y1 = min(xs or [0]), min(ys or [0]), max(xe or [0]), max(ye or [0])
            line_text = _norm_space(" ".join(texts))
            lines.append(
                Line(
                    page=page.number + 1,
                    text=line_text,
                    bbox=(x0, y0, x1, y1),
                    left_indent=x0,
                    width=(x1 - x0),
                    is_bold=(sum(bold_votes) >= max(1, len(bold_votes) // 2)),
                    font_size=(sum(sizes) / max(1, len(sizes))),
                )
            )
    return lines


def _detect_columns(lines: List[Line]) -> int:
    if len(lines) < 6:
        return 1
    mids = sorted([((l.bbox[0] + l.bbox[2]) / 2.0) for l in lines])
    span = max(mids) - min(mids) if mids else 1.0
    gaps = [mids[i + 1] - mids[i] for i in range(len(mids) - 1)]
    if not gaps or span <= 0:
        return 1
    if max(gaps) / span > 0.33:
        return 2
    return 1


def _assign_column(lines: List[Line]) -> None:
    if not lines:
        return
    mids = sorted([((l.bbox[0] + l.bbox[2]) / 2.0) for l in lines])
    span = max(mids) - min(mids) if mids else 1.0
    gaps = [mids[i + 1] - mids[i] for i in range(len(mids) - 1)]
    if not gaps or span <= 0 or (max(gaps) / span) <= 0.33:
        for l in lines:
            setattr(l, "column", 1)
        return
    cut_idx = gaps.index(max(gaps))
    cut_x = mids[cut_idx]
    for l in lines:
        xm = (l.bbox[0] + l.bbox[2]) / 2.0
        setattr(l, "column", 1 if xm <= cut_x else 2)


def extract_from_pdf(path: str, collect_debug: bool = False) -> Dict[str, Any]:
    if collect_debug:
        print(f"[debug] extract_from_pdf() called with path: {path}")
    doc = fitz.open(path)
    all_lines: List[Line] = []
    for p in range(len(doc)):
        page = doc[p]
        plines = _page_lines_from_pdf(page)
        if _detect_columns(plines) == 2:
            _assign_column(plines)
        else:
            for l in plines:
                setattr(l, "column", 1)
        plines.sort(key=lambda l: (getattr(l, "column", 1), l.bbox[1], l.bbox[0]))
        all_lines.extend(plines)

    all_lines.sort(key=lambda l: (l.page, getattr(l, "column", 1), l.bbox[1], l.bbox[0]))

    # --- Fix: merge right-aligned spans with anchor line ---
    grouped: List[Line] = []
    used = set()

    for i, l in enumerate(all_lines):
        if i in used:
            continue

        # Find other spans on same page + same vertical band
        same_y = []
        for j, r in enumerate(all_lines):
            if j <= i or j in used:
                continue
            if l.page == r.page and abs(r.bbox[1] - l.bbox[1]) < 3:
                same_y.append((j, r))

        if same_y:
            grouped.append(l)
            for j, r in same_y:
                # Stack the right-aligned spans below
                grouped.append(
                    Line(
                        page=r.page,
                        text=r.text,
                        bbox=r.bbox,
                        left_indent=r.left_indent,
                        width=r.width,
                        is_bold=r.is_bold,
                        font_size=r.font_size,
                    )
                )
            used.update([j for j, _ in same_y])
        else:
            grouped.append(l)

    all_lines = grouped

    # --- Build output text ---
    pages_text = []
    for pg in range(1, len(doc) + 1):
        page_text = "\n".join([ln.text for ln in all_lines if ln.page == pg])
        pages_text.append(page_text)

    return {
        "meta": {"source_path": path, "pages": len(doc)},
        "lines": [asdict(l) for l in all_lines],
        "text_per_page": pages_text,
        "text_all": "\n\n".join(pages_text),
    }


def extract_from_docx(path: str) -> Dict[str, Any]:
    doc = DocxDocument(path)
    lines: List[Line] = []
    y = 0.0
    for para in doc.paragraphs:
        t = _norm_space(para.text)
        if not t:
            y += 12
            continue
        sizes, bold_votes = [], []
        for run in para.runs:
            if run.font.size:
                sizes.append(float(run.font.size.pt))
            bold_votes.append(1 if run.bold else 0)
        font_size = (sum(sizes) / len(sizes)) if sizes else 11.0
        is_bold = sum(bold_votes) >= max(1, len(bold_votes) // 2)
        width = 600.0
        lines.append(
            Line(
                page=1,
                text=t,
                bbox=(0.0, y, width, y + font_size + 4),
                left_indent=0.0,
                width=width,
                is_bold=is_bold,
                font_size=font_size,
            )
        )
        y += font_size + 8
    lines.sort(key=lambda l: (l.page, l.bbox[1], l.bbox[0]))
    return {
        "meta": {"source_path": path, "pages": 1},
        "lines": [asdict(l) for l in lines],
        "text_per_page": ["\n".join([l.text for l in lines])],
        "text_all": "\n".join([l.text for l in lines]),
    }


def detect_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    raise ValueError(f"Unsupported file type: {ext}")


def main():
    ap = argparse.ArgumentParser(description="Extract text + ordered lines from PDF/DOCX")
    ap.add_argument("--path", required=True, help="Path to PDF/DOCX resume")
    ap.add_argument("--out_json", required=True, help="Path to write JSON (lines + text)")
    args = ap.parse_args()

    ftype = detect_type(args.path)
    if ftype == "pdf":
        payload = extract_from_pdf(args.path)
    else:
        payload = extract_from_docx(args.path)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[ok] Extracted text & lines → {args.out_json}")


if __name__ == "__main__":
    main()