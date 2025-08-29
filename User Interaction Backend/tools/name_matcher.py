import os
import psycopg2
from typing import List, Dict
from fuzzywuzzy import fuzz
from dotenv import load_dotenv

load_dotenv()

SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")

def fetch_all_candidate_names() -> List[Dict[str, str]]:
    conn = psycopg2.connect(SUPABASE_DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT full_name, primary_email FROM candidate")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return [{"name": r[0], "email": r[1]} for r in rows if r[0] and r[1]]

def match_names_to_emails(query_names: List[str], threshold: int = 85) -> List[Dict[str, str]]:
    candidates = fetch_all_candidate_names()
    seen_emails = set()
    matches = []

    for name in query_names:
        for cand in candidates:
            if fuzz.partial_ratio(name.lower(), cand["name"].lower()) >= threshold:
                if cand["email"] not in seen_emails:
                    seen_emails.add(cand["email"])
                    matches.append({
                        "query_name": name,
                        "matched_name": cand["name"],
                        "email": cand["email"]
                    })

    return matches