from __future__ import annotations
import os, json, argparse, hashlib, time
from typing import Any, Dict, List
from datetime import datetime
import requests

# -------- Config (env) --------
BASE_URL = (os.getenv("NGROK_INGEST_BASE", "").strip() or None)
TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "20"))
RETRY = int(os.getenv("HTTP_RETRY", "2"))
SLEEP_BETWEEN = float(os.getenv("ROW_SLEEP", "0.05"))

def _ensure_base_url(override: str | None = None) -> str:
    """Validate and finalize the BASE_URL from CLI override or env.
    Raises a clear error if not set or contains placeholders."""
    global BASE_URL
    base = (override or BASE_URL or "").strip()
    if not base or "<" in base or base.lower().startswith("http://<"):
        raise RuntimeError(
            "NGROK_INGEST_BASE is not set. Set it in your environment or pass --base_url. "
            "Example: https://de997fc7b262.ngrok-free.app"
        )
    BASE_URL = base.rstrip("/")
    return BASE_URL

# -------- Helpers --------
def _iso_date(s: str | None) -> str | None:
    """Normalize 'YYYY-MM'| 'YYYY-MM-DD' -> 'YYYY-MM-DD' (first-of-month if needed)."""
    if not s:
        return None
    s = s.strip()
    if len(s) == 7 and s[4] == "-":        # YYYY-MM
        return f"{s}-01"
    if len(s) == 10 and s[4] == "-" and s[7] == "-":  # YYYY-MM-DD
        return s
    return None

def _first(*vals):
    for v in vals:
        if v:
            return v
    return None

def _id(prefix: str, *parts: str) -> str:
    """Stable short id from content (deterministic across runs)."""
    h = hashlib.sha1("|".join([p or "" for p in parts]).encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{h}"

def _post(table: str, rows: List[Dict[str, Any]], *, dry: bool = False) -> None:
    """POST rows to {BASE}/ingest/{table}. Accepts list; wraps as {'data': [...]}."""
    if not rows:
        return
    base = _ensure_base_url()
    url = f"{base}/ingest/{table}"
    payload = {"data": rows}
    if dry:
        print(f"\n--- DRY RUN: POST {url} ---")
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    headers = {"Content-Type": "application/json"}

    last_err = None
    for attempt in range(1, RETRY + 2):
        try:
            resp = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
            if resp.status_code < 300:
                print(f"→ POST {table}: {len(rows)} rows OK")
                return
            last_err = f"{resp.status_code} {resp.text.strip()}"
        except Exception as e:
            last_err = str(e)
        time.sleep(0.4 * attempt)

    raise RuntimeError(f"POST {url} failed after retries: {last_err}")

# -------- Canonical -> Schema mappers --------
def map_experience(candidate_id: str, experiences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, e in enumerate(experiences or []):
        # bullets may be list[str] or list[{'bullet_text': str}]
        bullets: List[str] = []
        for b in (e.get("bullets") or []):
            if isinstance(b, dict) and b.get("bullet_text"):
                bullets.append(b["bullet_text"])
            elif isinstance(b, str):
                bullets.append(b)

        out.append({
            "id": _id("exp", candidate_id, str(i), e.get("company_raw") or e.get("company_norm") or e.get("title_raw") or ""),
            "candidate_id": candidate_id,
            "company": _first(e.get("company_raw"), e.get("company_norm")),
            "title": _first(e.get("title_raw"), e.get("title_norm")),
            "location": e.get("location"),
            "from_date": _iso_date(e.get("start_date")),
            "to_date": _iso_date(e.get("end_date")),
            "is_current": bool(e.get("is_current")),
            "bullets": bullets or None,
            "updated_at": datetime.utcnow().isoformat() + "Z",
            "seniority": e.get("seniority"),
        })
    return out

def map_education(candidate_id: str, schools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, ed in enumerate(schools or []):
        out.append({
            "id": _id("edu", candidate_id, str(i), ed.get("institution_raw") or ""),
            "candidate_id": candidate_id,
            "school": ed.get("institution_raw"),
            "degree": _first(ed.get("degree_raw"), ed.get("degree_level")),
            "major": ed.get("field_raw"),
            "location": ed.get("location") or None,
            "from_date": _iso_date(ed.get("start_date")),
            "to_date": _iso_date(ed.get("end_date")),
            "gpa": ed.get("gpa"),
            "updated_at": datetime.utcnow().isoformat() + "Z",
        })
    return out

def map_projects(candidate_id: str, projects: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for i, p in enumerate(projects or []):
        highlights: List[str] = []
        for b in (p.get("bullets") or []):
            if isinstance(b, dict) and b.get("bullet_text"):
                highlights.append(b["bullet_text"])
            elif isinstance(b, str):
                highlights.append(b)
        if p.get("summary"):
            highlights = [p["summary"], *highlights]

        out.append({
            "id": _id("proj", candidate_id, str(i), p.get("name") or ""),
            "candidate_id": candidate_id,
            "name": p.get("name"),
            "from_date": _iso_date(p.get("start_date")),
            "to_date": _iso_date(p.get("end_date")),
            "highlights": highlights or None,
            "updated_at": datetime.utcnow().isoformat() + "Z",
        })
    return out

def map_skills(candidate_id: str, skills: Any) -> List[Dict[str, Any]]:
    """If canonical lacks categories, collapse to {'General': [...]} of unique names."""
    if isinstance(skills, dict) and "categories" in skills:
        cats = skills["categories"]
    else:
        names: List[str] = []
        if isinstance(skills, dict):
            for v in skills.values():
                if isinstance(v, list):
                    names.extend([str(x) for x in v])
                elif isinstance(v, str):
                    names.extend([s.strip() for s in v.split(",") if s.strip()])
        else:
            for s in (skills or []):
                if isinstance(s, dict) and s.get("name"):
                    names.append(str(s["name"]))
                elif isinstance(s, str):
                    names.append(s)
        cats = {"General": sorted(list({n for n in names if n}))}

    return [{
        "id": _id("skills", candidate_id, "0"),
        "candidate_id": candidate_id,
        "categories": cats,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }]

# -------- Public: importable function for your pipeline --------
def send_to_api(table: str, data: List[Dict[str, Any]] | Dict[str, Any], *, dry_run: bool = False) -> None:
    """Public API your pipeline can call. Table = experience|education|projects|skills."""
    _ensure_base_url()
    rows = data if isinstance(data, list) else [data]
    _post(table, rows, dry=dry_run)

# -------- CLI --------
def main():
    ap = argparse.ArgumentParser(description="Push canonical JSON(s) to friend's ingestion API.")
    ap.add_argument("--in_path", default=os.path.join(os.getcwd(), "outputs", "canonical_files"),
                    help="Path to a canonical JSON file or a folder of *.json (default: ./outputs/canonical_files)")
    ap.add_argument("--tables", default="experience,education,projects,skills",
                    help="Comma-separated tables to send")
    ap.add_argument("--dry_run", action="store_true", help="Print payloads; do not POST")
    ap.add_argument("--base_url", default=None, help="Override NGROK_INGEST_BASE (e.g., https://xxxx.ngrok-free.app)")
    args = ap.parse_args()

    target = _ensure_base_url(args.base_url)
    print(f"Using endpoint base: {target}")

    # collect files
    paths: List[str] = []
    if os.path.isdir(args.in_path):
        for n in sorted(os.listdir(args.in_path)):
            if n.lower().endswith(".json"):
                paths.append(os.path.join(args.in_path, n))
    else:
        paths = [args.in_path]

    tables = {t.strip() for t in args.tables.split(",") if t.strip()}
    totals = {t: 0 for t in tables}

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            doc = json.load(f)

        cand = doc.get("candidate") or {}
        cand_id = cand.get("candidate_id") or cand.get("resume_text_hash") or cand.get("full_name") or "unknown"

        if "experience" in tables:
            rows = map_experience(cand_id, doc.get("experience") or [])
            _post("experience", rows, dry=args.dry_run)
            totals["experience"] += len(rows)

        if "education" in tables:
            rows = map_education(cand_id, doc.get("education") or [])
            _post("education", rows, dry=args.dry_run)
            totals["education"] += len(rows)

        if "projects" in tables:
            rows = map_projects(cand_id, doc.get("projects") or [])
            _post("projects", rows, dry=args.dry_run)
            totals["projects"] += len(rows)

        if "skills" in tables:
            rows = map_skills(cand_id, doc.get("skills") or [])
            _post("skills", rows, dry=args.dry_run)
            totals["skills"] += len(rows)

        time.sleep(SLEEP_BETWEEN)

    print("Done. Sent rows:", totals)

if __name__ == "__main__":
    main()