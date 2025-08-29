from __future__ import annotations
import argparse, json, os
from typing import Dict, Any

# Layout detector
from src.textpipe.layout_detect import detect_resume_layout

# Two extractors
from src.textpipe.extract_text_2col import extract_pdf_two_col
from src.textpipe.extract_text import extract_from_pdf  # <-- rename if your function differs


def main():
    ap = argparse.ArgumentParser(description="Auto-detect resume layout and extract accordingly")
    ap.add_argument("--path", required=True, help="Path to PDF resume")
    ap.add_argument("--out_json", required=True, help="Path to write JSON output")
    ap.add_argument("--out_txt", help="Optional: write plain-text dump")
    ap.add_argument("--debug", action="store_true", help="Print detector logs")
    ap.add_argument("--interleave", action="store_true",
                    help="(2-col only) Interleave left/right by vertical position")
    ap.add_argument("--probe_pages", type=int, default=2,
                    help="How many pages to sample for layout detection (default: 2)")
    args = ap.parse_args()

    print(f"[DEBUG] Parsed arguments: {args}", flush=True)

    # Convert paths to strings safely
    args.path = str(args.path) if not isinstance(args.path, str) else args.path
    args.out_json = str(args.out_json) if not isinstance(args.out_json, str) else args.out_json
    if args.out_txt:
        args.out_txt = str(args.out_txt) if not isinstance(args.out_txt, str) else args.out_txt

    print(f"[DEBUG] Type of args.path: {type(args.path)}", flush=True)
    print(f"[DEBUG] Type of args.out_json: {type(args.out_json)}", flush=True)
    if args.out_txt:
        print(f"[DEBUG] Type of args.out_txt: {type(args.out_txt)}", flush=True)

    print(f"[DEBUG] Input PDF path: {args.path}", flush=True)
    print(f"[DEBUG] Output JSON path: {args.out_json}", flush=True)
    if args.out_txt:
        print(f"[DEBUG] Output TXT path: {args.out_txt}", flush=True)

    # 1) Detect layout fast (uses your existing heuristics)
    print(f"[DEBUG] Running layout detection...", flush=True)
    report = detect_resume_layout(args.path, sample_pages=args.probe_pages, debug=args.debug)
    layout = report.get("layout", "single")
    print(f"[DEBUG] Layout detection report: {report}", flush=True)
    if args.debug:
        print(f"[auto] layout={layout} confidence={report.get('confidence'):.2f} "
              f"cuts_found={report.get('cuts_found')} pages={report.get('pages')}")

    # 2) Route to the right extractor
    print(f"[DEBUG] Routing to extractor: {layout}", flush=True)
    if layout == "two_column":
        payload = extract_pdf_two_col(args.path, interleave=args.interleave, collect_debug=args.debug)
    else:
        payload = extract_from_pdf(args.path, collect_debug=args.debug)
    print(f"[DEBUG] Extraction completed. Keys: {list(payload.keys())}", flush=True)

    # 3) Persist outputs
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    print(f"[DEBUG] Writing JSON output...", flush=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[DEBUG] JSON written to {args.out_json}", flush=True)

    if args.out_txt:
        print(f"[DEBUG] Writing plain text output...", flush=True)
        with open(args.out_txt, "w", encoding="utf-8") as f:
            f.write(payload["text_all"])
        print(f"[DEBUG] TXT written to {args.out_txt}", flush=True)

    print(f"[ok] {layout} extract → {args.out_json}")
    if args.out_txt:
        print(f"[ok] Plain text → {args.out_txt}")


if __name__ == "__main__":
    main()