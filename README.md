# HR Automation System – Resume Q&A & Shortlisting Assistant

This project is a HR automation system that enables:
- Resume parsing and structured data storage
- Intelligent Q&A over company/resume documents
- A Next.js frontend for user interaction
- A backend service for chatbot query handling

Each component works as an independent service but integrates into a unified workflow.

---

## Project Structure

- **Company_Document_QnA/**  
  Handles document-based Q&A (company policies, job descriptions, resumes).  

- **Parsing_Module/**  
  Backend pipeline for parsing resumes into structured JSON + relational DB + vector DB entries.  

- **SigmaHire_Frontend/**  
  Next.js web application for recruiters/interviewers.  

- **User_Interaction_Backend/**  
  FastAPI backend serving chatbot and Q&A APIs, consumed by the frontend.  

---

## Getting Started

### 1. Company Document Q&A

**Purpose:** Answer questions from resumes and company documents using embeddings + LLM.  

**Setup & Run**
\`\`\`bash
cd Company_Document_QnA
pip install -r requirements.txt
python app.py
\`\`\`

---

### 2. Parsing Module

**Purpose:** Parse resumes (PDF/DOCX) into structured JSON + relational DB + vector DB entries.  

**Setup & Run**
\`\`\`bash
cd Parsing_Module
pip install -r requirements.txt
uvicorn app:app --reload --port 8001
\`\`\`

- Starts a backend parsing server.  
- The frontend will call this API to trigger parsing, storage, and embedding.  
- Pipeline: **Parse → Normalize → Store → Embed → Index**  

---

### 3. SigmaHire Frontend (Next.js)

**Purpose:** Web UI for recruiters to interact with parsed resumes, shortlist candidates, and query the system.  

**Setup & Run**
\`\`\`bash
cd SigmaHire_Frontend
npm install
npm run dev
\`\`\`

- Runs at [http://localhost:3000](http://localhost:3000).  
- Communicates with the Parsing Module + User Interaction Backend.  

---

### 4. User Interaction Backend (FastAPI)

**Purpose:** Acts as the central chatbot/Q&A backend. Serves APIs for:  
- Resume Q&A  
- Candidate shortlisting  
- Context-aware conversations  

**Setup & Run**
\`\`\`bash
cd User_Interaction_Backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8002
\`\`\`

---

## 🔗 How Components Work Together

1. **Parsing Module**: Resumes are uploaded → parsed → stored in DB + vector index.  
2. **User Interaction Backend**: Provides APIs for querying resumes & company docs.  
3. **SigmaHire Frontend**: UI for recruiters to query and shortlist candidates.  
4. **Company Document Q&A**: Provides document-level intelligence for policies and resumes.  

All services run in parallel and communicate via REST APIs.  
The flow is: **Frontend → Backend → Parsing/DB → Response to User**.  

---

## 🛠️ Tech Stack

- **Backend**: Python, FastAPI, LangChain, ChromaDB, Supabase (Postgres)  
- **Frontend**: Next.js, React, Tailwind, Framer Motion  
- **Parsing**: PyMuPDF, docx, unstructured, spaCy, Transformers  
- **Infra**: Uvicorn, npm, pip  

---

## 📌 Notes

- Each service is modular and can be scaled or deployed independently.  
- Environment variables (API keys, DB URLs) should be configured in `.env` files inside each component.    

---

## 👨‍💻 Authors
Abhay
Samarthya 
