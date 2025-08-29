import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()



# Load OpenAI client and model
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GPT_MODEL = os.getenv("GPT_MODEL")

client = OpenAI(api_key=OPENAI_API_KEY)

def classify_intent(query: str) -> str:
    print(f"[INTENT DEBUG] Classifying intent for query: {query}")

    system_prompt = """
You are an assistant that classifies user queries so the user can get answers related to resume data wh.
Your job is to direct queries to the correct backend tool. The assistant has two main tools:

1. sql_tool (structured database of applied candidates resume data)
   - Use this when the answer depends on **structured fields** stored in a Postgres DB.
   - Example queries: filtering by degree, date ranges, job titles, company names, skills, locations, counts.
   - Schema includes: candidate (name, email, location, phone, LinkedIn/GitHub URLs, source file),
     experience (company, title, dates, summary),
     education (degree, field, institution, GPA),
     projects, certifications, achievements, publications, languages, skills.
   - Best for: precise queries like "How many candidates have a Master's in Computer Science?",
     "List candidates with Python skill and 3+ years at Google", "Show me candidates in New York".

2. vector_tool (semantic embeddings in ChromaDB of resume data of applied candidates)
   - Use this when the query involves **unstructured or descriptive text**.
   - Example queries: personality traits, project summaries, work style, achievements phrased in free text,
     "Who has led AI/ML projects?", "Find candidates with strong leadership qualities".
   - Best for: subjective, fuzzy, or full-text search over resume sections (experience, projects, achievements, raw text).

3. general
   - Use this for greetings, onboarding, help requests, or usage instructions.
   - Example: "Hi", "How do I use this assistant?", "What can you do?"

Return ONLY one of: `sql`, `semantic`,or `general`.
"""

    messages = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": query})

    completion = client.chat.completions.create(
        model=GPT_MODEL,
        messages=messages
    )

    intent = completion.choices[0].message.content.strip().lower()
    print(f"[INTENT DEBUG] LLM returned intent: {intent}", flush=True)
    return intent