import os
import json
import hashlib
from datetime import datetime
from typing import List
import shutil

from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredWordDocumentLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import pdfplumber
from langchain.schema import Document

# --------------------
# Config via .env or defaults
# --------------------
load_dotenv()
DATA_FOLDER = os.getenv("DATA_FOLDER", "data")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 900))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 200))

CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "company-docs")

# Manifest just to detect changes and skip unnecessary re-embedding (Chroma only)
MANIFEST_PATH = ".chroma_manifest.json"
SUPPORTED_EXTS = {".pdf", ".docx", ".txt"}

# Infer expected embedding dimension
EMBED_DIM = 1536 if "3-small" in EMBEDDING_MODEL else (3072 if "3-large" in EMBEDDING_MODEL else 1536)

# --------------------
# Helpers
# --------------------

def file_sha1(path: str) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_text(s: str) -> str:
    # light cleanup to improve chunking/embeddings
    return " ".join((s or "").replace("\u00a0", " ").split())


def list_files(root: str) -> List[str]:
    files = []
    for name in sorted(os.listdir(root)):
        if name.startswith("."):
            continue
        p = os.path.join(root, name)
        ext = os.path.splitext(name)[1].lower()
        if os.path.isfile(p) and ext in SUPPORTED_EXTS:
            files.append(p)
    return files


import pdfplumber
from langchain.schema import Document

def load_with_pdfplumber(path: str, filename: str):
    docs = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages):
            # Paragraph text
            text = page.extract_text() or ""
            if text.strip():
                for para in text.split("\n\n"):
                    if para.strip():
                        docs.append(Document(
                            page_content=normalize_text(para),
                            metadata={
                                "source": filename,
                                "source_path": os.path.abspath(path),
                                "page": i,
                                "page_display": i+1,
                                "type": "paragraph",
                                "block_type": "paragraph"
                            }
                        ))

            # Tables
            tables = page.extract_tables()
            for tbl in tables:
                if tbl:
                    rows = [[cell or "" for cell in row] for row in tbl if row]
                    if rows:
                        header = "| " + " | ".join(rows[0]) + " |"
                        separator = "| " + " | ".join(["---"] * len(rows[0])) + " |"
                        body = ["| " + " | ".join(r) + " |" for r in rows[1:]]
                        table_str = "\n".join([header, separator] + body)
                    docs.append(Document(
                        page_content=normalize_text(table_str),
                        metadata={
                            "source": filename,
                            "source_path": os.path.abspath(path),
                            "page": i,
                            "page_display": i+1,
                            "type": "table",
                            "block_type": "table"
                        }
                    ))
    return docs


# loading file and metadata

def load_docs(path_list: List[str]):
    documents = []
    for p in path_list:
        ext = os.path.splitext(p)[1].lower()
        filename = os.path.basename(p)
        try:
            if ext == ".pdf":
                file_docs = load_with_pdfplumber(p, filename)
            elif ext == ".docx":
                loader = UnstructuredWordDocumentLoader(p)
                file_docs = loader.load()
            else:  # .txt
                loader = TextLoader(p, autodetect_encoding=True)
                file_docs = loader.load()

            for d in file_docs:
                d.page_content = normalize_text(d.page_content)
                d.metadata["source"] = filename
                d.metadata["source_path"] = os.path.abspath(p)
                if "page" in d.metadata and isinstance(d.metadata["page"], int):
                    d.metadata["page_display"] = int(d.metadata["page"]) + 1
        except Exception as e:
            print(f"! Skipping {filename}: {e}")
            continue
        documents.extend(file_docs)
    return documents


# remember what files are embedded

def read_manifest() -> dict:
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def write_manifest(entries: List[dict]):
    with open(MANIFEST_PATH, "w", encoding="utf-8") as f:
        json.dump({"files": entries, "updated_at": datetime.utcnow().isoformat() + "Z"}, f, indent=2)


# --------------------
# Main build function
# --------------------

def create_vectorstore():
    if not os.path.exists(DATA_FOLDER):
        raise FileNotFoundError(f"DATA_FOLDER not found: {DATA_FOLDER}")

    paths = list_files(DATA_FOLDER)
    if not paths:
        print("No supported files found in 'data/'. Supported: .pdf, .docx, .txt")
        return

    # Build current file inventory
    current = []
    for p in paths:
        current.append({
            "path": os.path.abspath(p),
            "name": os.path.basename(p),
            "sha1": file_sha1(p),
            "mtime": os.path.getmtime(p),
        })

    # Compare with manifest to detect changes
    manifest = read_manifest()
    prev = manifest.get("files", [])

    def _key(x):
        return (x["name"], x["sha1"])  # name+hash uniquely identify content

    unchanged = set(map(_key, prev)).intersection(set(map(_key, current)))
    changed = len(unchanged) != len(current)

    # (Re)build Chroma collection from scratch for simplicity & correctness
    if changed:
        print("Changes detected. Rebuilding Chroma collection…")
        # Remove existing persisted DB dir to ensure a clean rebuild
        try:
            shutil.rmtree(CHROMA_DIR)
        except FileNotFoundError:
            pass
    else:
        print("No changes detected. Chroma collection appears up-to-date.")
        return

    print("Building embeddings & upserting to Chroma… this may take a moment…")
    docs = load_docs(paths)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n• ", "\n- ", "\n", " ", ""],  # prefer keeping bullets whole
    )
    chunks = splitter.split_documents(docs)

    # Optional prune: drop tiny / duplicate chunks
    MIN_CHARS = 60
    seen = set()
    cleaned = []
    for d in chunks:
        txt = (d.page_content or "").strip()
        if len(txt) < MIN_CHARS:
            continue
        h = hashlib.sha1(txt.encode("utf-8")).hexdigest()
        if h in seen:
            continue
        seen.add(h)
        d.metadata["chunk_sha1"] = h
        cleaned.append(d)
    chunks = cleaned

    embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DIR,
        collection_name=CHROMA_COLLECTION,
    )
    # Ensure data is flushed to disk
    try:
        vectorstore.persist()
    except Exception:
        pass

    print(f"Upserted {len(chunks)} chunks from {len(paths)} files to Chroma collection '{CHROMA_COLLECTION}' (dir='{CHROMA_DIR}').")


if __name__ == "__main__":
    create_vectorstore()