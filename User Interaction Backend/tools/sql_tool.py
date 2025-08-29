import os
import psycopg2
from typing import List, Dict, Optional
from openai import OpenAI

from dotenv import load_dotenv
load_dotenv()

# Load env variables
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("LLM_MODEL")
SUPABASE_DB_URL = os.getenv("SUPABASE_DB_URL")
GPT_MODEL = os.getenv("GPT_MODEL", "gpt-4o-mini")

# Chosen schema prompt
SCHEMA_TEXT = (
    "Tables (Resume DB v2): "
    "candidate(pk=candidate_id UUID, full_name, primary_email CITEXT, alt_emails CITEXT[], primary_phone, "
    "location_city, location_state, location_country, linkedin_url, github_url, portfolio_url, resume_text_hash UNIQUE, "
    "source_file_id TEXT, parsed_at TIMESTAMPTZ, created_at TIMESTAMPTZ); "

    "resume_source(pk=source_file_id TEXT, filename TEXT, bytes BIGINT, mime_type TEXT, uploaded_at TIMESTAMPTZ); "
    "  • One resume_source → many candidates (via source_file_id) "

    "experience(pk=exp_id UUID, candidate_id → candidate.candidate_id, company_norm, title_raw, "
    "employment_type, start_date DATE, end_date DATE, exp_summary TEXT, created_at TIMESTAMPTZ); "
    "  • One candidate → many experiences "

    "education(pk=edu_id UUID, candidate_id → candidate.candidate_id, degree_raw, degree_level, field_raw, field_norm, institution_raw, institution_norm, "
    "start_date DATE, end_date DATE, gpa TEXT, grading_scale TEXT, created_at TIMESTAMPTZ); "
    "  • One candidate → many education entries "

    "project(pk=project_id UUID, candidate_id → candidate.candidate_id, name, role, start_date DATE, end_date DATE, repo_url TEXT, tech_stack TEXT[], "
    "project_summary TEXT, created_at TIMESTAMPTZ); "
    "  • One candidate → many projects "

    "publication(pk=pub_id UUID, candidate_id → candidate.candidate_id, title TEXT, year INT, url TEXT, created_at TIMESTAMPTZ); "
    "  • One candidate → many publications "

    "achievement(pk=ach_id UUID, candidate_id → candidate.candidate_id, title, description TEXT, date DATE, created_at TIMESTAMPTZ); "
    "  • One candidate → many achievements "

    "certification(pk=cert_id UUID, candidate_id → candidate.candidate_id, name, detailed_text, issue_date DATE, credential_id TEXT, url TEXT, created_at TIMESTAMPTZ); "
    "  • One candidate → many certifications "

    "candidate_language(pk=(candidate_id UUID, language TEXT), level TEXT); "
    "  • One candidate → many languages "

    "raw_section_text(pk=section_id UUID, candidate_id → candidate.candidate_id, section_type TEXT CHECK experience|project|education|skills|summary|certification|publication|achievement|language|other, "
    "text TEXT, source_file_id TEXT, created_at TIMESTAMPTZ); "
    "  • One candidate → many section_texts "

    "skill(pk=(candidate_id UUID, name_norm TEXT[])); "
    "  • One candidate → many normalized skills "
)

def run_sql_query(sql: str, params: Optional[List] = None) -> List[Dict]:
    conn = psycopg2.connect(SUPABASE_DB_URL)
    cur = conn.cursor()

    try:
        if params:
            cur.execute(sql, params)
        else:
            cur.execute(sql)  # safe if no %s placeholders
        columns = [desc[0] for desc in cur.description]
        rows = [dict(zip(columns, row)) for row in cur.fetchall()]
        return rows
    finally:
        cur.close()
        conn.close()

def generate_sql_from_query(natural_query: str) -> str:
    client = OpenAI(api_key=OPENAI_API_KEY)

    system_prompt = (
        "You are an elite SQL generator. Produce a single correct SQL statement for the given database schema and"
        " question. Target SQL dialect: postgres. "
        "You MUST avoid any data modification (no INSERT/UPDATE/DELETE) and no transaction/commit statements."
        " The query MUST be SELECT-only. Return ONLY the SQL with no explanation, no markdown, and no other text."
    )

    user_prompt = (
        f"Schema\n---\n{SCHEMA_TEXT}\n\n"
        f"Question\n---\n{natural_query}"
    )

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    resp = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=messages,
        max_tokens=800
    )

    return resp.choices[0].message.content.strip()

from typing import List, Dict, Optional, Union

def execute_sql_for_intent(natural_query: str, candidate_emails: Optional[List[Union[str, Dict]]] = None) -> List[Dict]:
    sql = generate_sql_from_query(natural_query)

    print(f"[SQL DEBUG] Generated SQL:\n{sql}",flush=True)

    if candidate_emails:
        # Extract just the email string, whether it's a list of strings or dicts
        email_list = [c["email"] if isinstance(c, dict) else c for c in candidate_emails]
        placeholders = ", ".join(["%s"] * len(email_list))

        sql = f"""
        WITH base AS (
            {sql}
        )
        SELECT * FROM base WHERE primary_email IN ({placeholders})
        """

        return run_sql_query(sql, email_list)

    return run_sql_query(sql)