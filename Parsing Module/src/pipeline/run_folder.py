from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from typing import Dict, List, Optional
from pathlib import Path

SUPPORTED_EXT = {".pdf", ".docx", ".txt"}

def _now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def _stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def _mkdir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    proc = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed ({' '.join(cmd)}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    else:
        if proc.stdout.strip():
            print(proc.stdout.strip())

def _maybe_run(cmd: List[str], label: str) -> bool:
    try:
        print(f"[step] {label}: {' '.join(cmd)}")
        _run(cmd)
        return True
    except Exception as e:
        print(f"[warn] {label} failed: {e}")
        return False

def build_out_paths(doc_path: str, batch_out_dir: str) -> Dict[str, str]:
    st = _stem(doc_path)
    doc_dir = os.path.join(batch_out_dir, st)
    _mkdir(doc_dir)
    paths = {
        "doc_dir": doc_dir,
        "layout_json": os.path.join(doc_dir, f"{st}_layout.json"),
        "lines_json": os.path.join(doc_dir, f"{st}_lines.json"),
        "text_txt": os.path.join(doc_dir, f"{st}_text.txt"),
        "sections_json": os.path.join(doc_dir, f"{st}_sections_nli.json"),
        "ner_spacy": os.path.join(doc_dir, f"{st}_ner_spacy.json"),
        "ner_hf": os.path.join(doc_dir, f"{st}_ner_hf.json"),
        "ner_ens": os.path.join(doc_dir, f"{st}_ner_ens.json"),
        "canon_spacy": os.path.join(doc_dir, f"{st}_spacy_canonical.json"),
        "canon_hf": os.path.join(doc_dir, f"{st}_hf_canonical.json"),
        "canon_ens": os.path.join(doc_dir, f"{st}_ens_canonical.json"),
        "compare_json": os.path.join(doc_dir, f"{st}_ner_compare.json"),
    }
    print(f"[debug] Created output paths in build_out_paths for document '{doc_path}':")
    for k, v in paths.items():
        print(f"  {k}: {v}")
    return paths

def process_one(doc_path: str, batch_out_dir: str, min_conf: float, device: Optional[int], job_id: Optional[str] = None) -> Dict[str, str]:
    py = sys.executable
    print(f"[debug] Building output paths for doc_path: {doc_path}, batch_out_dir: {batch_out_dir}")
    paths = build_out_paths(doc_path, batch_out_dir)
    print(f"[debug] Output paths dictionary created:\n{json.dumps(paths, indent=2)}")

    print(paths)

    print("[debug] Starting layout detection and text extraction (run_auto)")
    print(f"[debug] Checking if input file exists: {doc_path} -> {os.path.exists(doc_path)}")
    run_auto_cmd = [
        str(py), "-m", "src.pipeline.run_auto",
        "--path", str(doc_path),
        "--out_json", str(paths["lines_json"]),
        "--out_txt", str(paths["text_txt"])
    ]
    print(f"[debug] Will run run_auto with command: {run_auto_cmd}")
    ok_auto = _maybe_run(run_auto_cmd, label="run_auto")

    if ok_auto:
        print(f"[debug] run_auto step completed successfully. Output files:")
        print(f"  lines_json exists: {os.path.exists(paths['lines_json'])}, size: {os.path.getsize(paths['lines_json']) if os.path.exists(paths['lines_json']) else 'N/A'}")
        print(f"  text_txt exists: {os.path.exists(paths['text_txt'])}, size: {os.path.getsize(paths['text_txt']) if os.path.exists(paths['text_txt']) else 'N/A'}")
    else:
        print("[warn] run_auto step failed. Aborting further processing.")
        return paths

    print("[debug] Starting sectioning with NLI model")
    print(f"[debug] Checking if input lines_json exists: {paths['lines_json']} -> {os.path.exists(paths['lines_json'])}")
    sec_cmd = [
        py, "-m", "src.postprocess.sectioner_nli",
        "--in_json", paths["lines_json"],
        "--out_json", paths["sections_json"],
        "--min_conf", str(min_conf),
    ]
    if device is not None:
        sec_cmd += ["--device", str(device)]
    print(f"[debug] Will run sectioner_nli with command: {sec_cmd}")
    ok_sectioner = _maybe_run(sec_cmd, label="sectioner_nli")
    if ok_sectioner:
        print(f"[debug] sectioner_nli step completed successfully. Output file:")
        print(f"  sections_json exists: {os.path.exists(paths['sections_json'])}, size: {os.path.getsize(paths['sections_json']) if os.path.exists(paths['sections_json']) else 'N/A'}")
    else:
        print("[warn] sectioner_nli step failed. Aborting further processing.")
        return paths

    print("[debug] Starting NER extraction steps")
    ner_steps = [
        ("ner_spacy", "src.ner.extract"),
        ("ner_hf", "src.ner.extract_hf"),
        ("ner_ens", "src.ner.extract_ensemble"),
    ]

    for ner_key, module in ner_steps:
        print(f"[debug] Running NER step: {ner_key}")
        print(f"[debug] Checking if input sections_json exists: {paths['sections_json']} -> {os.path.exists(paths['sections_json'])}")
        ner_cmd = [
            py, "-m", module,
            "--in_json", paths["sections_json"],
            "--out_json", paths[ner_key],
        ]
        print(f"[debug] Will run {ner_key} with command: {ner_cmd}")
        ok_ner = _maybe_run(ner_cmd, label=ner_key)
        if ok_ner:
            print(f"[debug] {ner_key} step completed successfully. Output file:")
            print(f"  {ner_key} exists: {os.path.exists(paths[ner_key])}, size: {os.path.getsize(paths[ner_key]) if os.path.exists(paths[ner_key]) else 'N/A'}")
        else:
            print(f"[warn] {ner_key} step failed.")

    print("[debug] Starting canonical conversion for each NER method")
    for ner_key, canon_key in [
        ("ner_spacy", "canon_spacy"),
        ("ner_hf", "canon_hf"),
        ("ner_ens", "canon_ens"),
    ]:
        exists = os.path.exists(paths[ner_key])
        size = os.path.getsize(paths[ner_key]) if exists else 'N/A'
        print(f"[debug] Checking if {ner_key} output exists and non-empty: {paths[ner_key]} -> exists: {exists}, size: {size}")
        if exists and size != 0:
            canon_cmd = [
                py, "-m", "src.transform.convert_schema",
                "--in_json", paths[ner_key],
                "--out_json", paths[canon_key],
                "--parser_version", "v0.3.0",
            ]
            print(f"[debug] Will run convert_schema for {ner_key} with command: {canon_cmd}")
            ok_canon = _maybe_run(canon_cmd, label=f"convert_schema:{ner_key}")
            if ok_canon:
                print(f"[debug] convert_schema for {ner_key} completed successfully. Output file:")
                print(f"  {canon_key} exists: {os.path.exists(paths[canon_key])}, size: {os.path.getsize(paths[canon_key]) if os.path.exists(paths[canon_key]) else 'N/A'}")
            else:
                print(f"[warn] convert_schema for {ner_key} failed.")
        else:
            print(f"[debug] Skipping convert_schema for {ner_key} due to missing or empty input.")

    # Ingest canonical ENS to database using ingest_one with job_id
    exists_canon_ens = os.path.exists(paths["canon_ens"])
    print(f"[debug] Checking if canonical ENS file exists: {paths['canon_ens']} -> {exists_canon_ens}")
    if exists_canon_ens:
        print("[debug] Preparing to ingest canonical ENS data into database via ingest_one")
        try:
            from src.db.ingest_canonical import ingest_one, get_client
            client = get_client()
            # Add detailed debugging about job_id and file being ingested
            print(f"[debug] Calling ingest_one with ENS file: {paths['canon_ens']} and job_id: {job_id}")
            ingest_one(sb=client, in_json=paths["canon_ens"], job_id=job_id)
            print("[ok] Canonical ENS data ingested successfully using ingest_one")
        except Exception as e:
            print(f"[warn] Ingestion via ingest_one failed: {e}")
    else:
        print(f"[debug] Canonical ENS file not found at {paths['canon_ens']}, skipping ingestion.")

    print("[debug] Comparing NER outputs")
    compare_cmd = [
        py, "-m", "src.eval.compare_ner",
        "--spacy", paths["ner_spacy"],
        "--hf", paths["ner_hf"],
        "--ens", paths["ner_ens"],
        "--out_json", paths["compare_json"]
    ]
    print(f"[debug] Will run compare_ner with command: {compare_cmd}")
    ok_compare = _maybe_run(compare_cmd, label="compare_ner")
    if ok_compare:
        print(f"[debug] compare_ner step completed successfully. Output file:")
        print(f"  compare_json exists: {os.path.exists(paths['compare_json'])}, size: {os.path.getsize(paths['compare_json']) if os.path.exists(paths['compare_json']) else 'N/A'}")
    else:
        print("[warn] compare_ner step failed.")

    return paths

def main():
    ap = argparse.ArgumentParser(description="Pipeline for a single resume file")
    ap.add_argument("--in_file", required=True, help="Path to the input resume file")
    ap.add_argument("--out_root", default="outputs")
    ap.add_argument("--min_conf", type=float, default=0.45)
    ap.add_argument("--device", type=int, default=None)
    args = ap.parse_args()

    batch_dir = os.path.join(args.out_root, f"in_{_now_stamp()}")
    _mkdir(batch_dir)

    summary = {"batch_dir": batch_dir, "items": []}

    print(f"\n=== Processing Single File: {os.path.basename(args.in_file)} ===")
    paths = process_one(args.in_file, batch_dir, args.min_conf, args.device, job_id=args.job_id)
    summary["items"].append({"input": args.in_file, "outputs": paths})

    with open(os.path.join(batch_dir, "batch_summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=2)
    print(f"\n[ok] File processing complete -> {batch_dir}")

if __name__ == "__main__":
    main()