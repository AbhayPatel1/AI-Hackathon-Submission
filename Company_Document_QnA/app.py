import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationTokenBufferMemory
from langchain.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
# ---- Spell correction
from symspellpy import SymSpell, Verbosity

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
CHROMA_DIR = os.getenv("CHROMA_DIR", "chroma")
CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "company-docs")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")

st.set_page_config(page_title="Company Docs Q&A", page_icon="📄", layout="wide")

# ---- Spell correction (SymSpell)
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dict_path = os.path.join("data", "frequency_dictionary_en_82_765.txt")
if os.path.exists(dict_path):
    sym_spell.load_dictionary(dict_path, term_index=0, count_index=1)

def correct_query_spelling(query: str) -> str:
    tokens = query.split()
    corrected = []
    for w in tokens:
        suggestions = sym_spell.lookup(w, Verbosity.CLOSEST, max_edit_distance=2)
        if suggestions:
            corrected.append(suggestions[0].term)
        else:
            corrected.append(w)
    return " ".join(corrected)

# ---- Global styles & top nav (company-like UI)
st.markdown(
    """
    <style>
      /* Base layout */
      .stApp {
          background: #f9fafb !important;
          font-family: 'Inter', sans-serif;
          background-image: linear-gradient(#f3f4f6 1px, transparent 1px), linear-gradient(to right, #f3f4f6 1px, transparent 1px);
          background-size: 20px 20px;
      }

      /* Top nav */
      .top-nav {
          display: flex;
          justify-content: space-between;
          align-items: center;
          padding: 14px 20px;
          background-color: white;
          border-bottom: 1px solid #e5e7eb;
      }
      .brand {
          font-size: 1.2rem;
          font-weight: 700;
          color: #4f46e5;
      }
      .brand .logo {
          width: 30px;
          height: 30px;
          border-radius: 6px;
          background: linear-gradient(135deg,#6366f1,#22d3ee);
      }
      .nav-links a {
          margin-left: 20px;
          color: #4b5563;
          text-decoration: none;
          font-weight: 500;
      }
      .nav-links a:hover {
          color: #1f2937;
      }

      /* Hero */
      .hero {
          background: white;
          border-radius: 12px;
          padding: 24px;
          text-align: center;
          box-shadow: 0 1px 3px rgba(0,0,0,0.06);
      }
      .hero h1 {
          font-size: 2rem;
          font-weight: 700;
          color: #1f2937;
      }
      .hero p {
          font-size: 1rem;
          color: #4b5563;
      }

      /* Chat bubbles (reduced padding, no white outer container, no emoji) */
      .stChatMessage {
          background: transparent !important;
          border-radius: 16px !important;
          box-shadow: none !important;
          padding: 0 !important;
          margin-bottom: 8px !important;
          color: #1f2937 !important;
          border: none !important;
          max-width: 800px;
          margin: 0 auto !important;
      }
      .stChatMessage p, .stChatMessage li, .stChatMessage span, .stChatMessage div {color: #1f2937 !important;}
      /* Remove background color from bot response bubble */
      .bot-msg {
        background: none !important;
      }

      /* PROFESSIONAL USER MESSAGE BACKGROUND */
      .user-message {
        background: linear-gradient(90deg, #dbeafe 0%, #eff6ff 100%) !important;
      }

      /* Shrink and center conversation window */
      .chat-message, .chat-input {
        max-width: 800px;
        margin: auto;
      }

      /* Remove emojis and large icons */
      .chat-avatar {
        display: none !important;
      }
      .stChatMessageAvatar {
        display: none !important;
      }

      /* Input box background match */
      .stChatInputContainer {
        background-color: #f8f9fa !important;
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px;
        max-width: 800px;
        margin: auto;
        margin-top: -10px !important;
        display: flex;
        align-items: center;
      }

      /* Reduce padding/margin around chat messages */
      .chat-message {
        padding: 4px 8px;
        margin: 8px 0;
      }

      /* Buttons */
      button[kind="primary"] {
          background-color: #4f46e5 !important;
          color: white !important;
          font-weight: 600;
          border-radius: 8px !important;
      }

      /* Footer */
      .footer {
          margin-top: 40px;
          color: #6b7280;
          font-size: 0.875rem;
          text-align: center;
      }

      /* Shrink container for chat UI */
      .shrink-container {
        max-width: 800px;
        margin: 0 auto;
      }

      /* Permanent black background for the Sources tab */
      .stExpander, .streamlit-expander {
          background-color: #000000 !important;
          color: white !important;
          border: 1px solid #333 !important;
      }
      .stExpander > summary, .streamlit-expander summary {
          color: white !important;
      }

    </style>
    <div class="top-nav">
      <div class="brand"><div class="logo"></div> Sigmoid — Knowledge Hub</div>
      <div class="nav-links">
        <a href="#">Home</a>
        <a href="#">Policies</a>
        <a href="#">Benefits</a>
        <a href="#">IT Help</a>
      </div>
    </div>
    <div class="hero">
      <h1>Company Documents Q&A</h1>
      <p>Ask about HR, benefits, WFH, reimbursements, and more — grounded in your official docs.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("")

# ---- Load vector store (ChromaDB)
embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    collection_name=CHROMA_COLLECTION,
    embedding_function=embeddings,
)

with st.sidebar:
    st.markdown("### 🏢 Sigmoid Intranet")
    st.caption("Private company knowledge assistant")
    st.divider()
    st.markdown("**Vector DB:** Chroma (persistent)")
    st.caption(f"Collection: `{CHROMA_COLLECTION}`\nDir: `{CHROMA_DIR}`")
    if st.button("Start new topic", use_container_width=True):
        st.session_state.chat = []
        st.session_state.last_question = ""
        st.rerun()
    st.divider()
    st.markdown("**Tips**")
    st.caption("Ask naturally. For new subject, click *Start new topic*.")

# ---- Retriever (diverse contexts)
retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 10, "fetch_k": 30}
)

# ---- Helper: aggregate sources with semantic scores
def aggregate_sources_with_scores(question: str, k: int = 10):
    """
    similarity_search_with_relevance_scores -> aggregate page numbers per file
    and keep the max score per file. Returns:
      { filename: {"pages": set[int], "score": float} }
    """
    out = {}
    if not question:
        return out
    try:
        results = vectorstore.similarity_search_with_relevance_scores(question, k=k)
    except Exception:
        return out
    for doc, score in results:
        meta = getattr(doc, "metadata", {}) or {}
        fn = meta.get("source", "Unknown")
        page = meta.get("page_display", meta.get("page", None))
        # normalize page
        page_num = None
        if isinstance(page, int):
            page_num = page
        else:
            try:
                page_num = int(page)
            except Exception:
                page_num = None
        if fn not in out:
            out[fn] = {"pages": set(), "score": float(score) if score is not None else 0.0}
        else:
            if score is not None:
                out[fn]["score"] = max(out[fn]["score"], float(score))
        if page_num is not None:
            out[fn]["pages"].add(page_num)
    return out

# ---- LLM + memory
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)

# Follow-up detection (Option B)
detect_prompt = PromptTemplate.from_template(
    """
    You are a classifier. Determine if the NEW_QUESTION depends on the PREVIOUS_QUESTION for meaning.
    Answer only "YES" or "NO".

    PREVIOUS_QUESTION: {previous}
    NEW_QUESTION: {current}
    """
)
def is_followup(prev_q: str, new_q: str) -> bool:
    prev_q = (prev_q or "").strip()
    new_q = (new_q or "").strip()
    if not prev_q:
        return False
    msg = detect_prompt.format(previous=prev_q, current=new_q)
    try:
        resp = llm.invoke(msg).content.strip().upper()
        return resp.startswith("Y")
    except Exception:
        return False

memory = ConversationTokenBufferMemory(
    llm=llm,
    memory_key="chat_history",
    input_key="question",
    output_key="answer",
    max_token_limit=4000,
    return_messages=True,
)

# ---- QA prompt
QA_TEMPLATE = """
You are a helpful assistant answering strictly from the provided context.
If the answer is not in the context, say you don't know and suggest the closest relevant info you DO have.

Include concrete numbers, limits, eligibility, processes, and exceptions when present.
Write concise bulleted answers when appropriate.

Context:
{context}

Chat history:
{chat_history}

Question:
{question}

Answer:
"""
qa_prompt = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template=QA_TEMPLATE.strip()
)

condense_prompt = PromptTemplate.from_template(
    """
    Given the following conversation and a follow-up question, rephrase the follow-up question to be a standalone question that can be used for document retrieval.

    Chat history:
    {chat_history}

    Follow-up question:
    {question}

    Rephrased question:
    """
)

# ---- Streaming handler for live token updates
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.text = ""
    def on_llm_new_token(self, token, **kwargs):
        self.text += token
        self.container.markdown(self.text)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    combine_docs_chain_kwargs={"prompt": qa_prompt},
    condense_question_prompt=condense_prompt,
)

# ---- UI state
if "chat" not in st.session_state:
    st.session_state.chat = []
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

#
# ---- Render existing history first (with layout limit)
st.markdown("""
<style>
  .chat-container {
    max-width: 700px;
    margin: auto;
  }
  .stChatInput input {
    max-width: 700px;
    margin: auto;
    display: block;
  }
  .stChatMessage {
    background: none !important;
    box-shadow: none !important;
  }
  .stChatMessageAvatar {
    display: none !important;
  }
</style>
<div class='chat-container'>
""", unsafe_allow_html=True)

for msg in st.session_state.get("chat", []):
    role = msg[0]
    content = msg[1]
    with st.chat_message(role):
        if role == "user":
            st.markdown(
                f"""<div style='display:flex; justify-content:flex-end; margin-bottom:6px;'>
                <div class="user-message chat-message" style='border-radius: 14px; padding: 6px 12px; font-size: 17px; font-weight:500; color:#1f2937; text-align:right;'>
                    {content}
                </div>
                </div>""",
                unsafe_allow_html=True
            )
        else:
            st.markdown(content, unsafe_allow_html=False)
    if role != "user" and len(msg) > 2 and msg[2]:
        source_docs = msg[2]
        composed_q = msg[3] if len(msg) > 3 else None
        agg = aggregate_sources_with_scores(composed_q) if composed_q else {}
        with st.expander("Sources"):
            files_pages = {}
            for d in source_docs:
                meta = getattr(d, "metadata", {}) or {}
                fn = meta.get("source", "Unknown")
                page = meta.get("page_display", meta.get("page", None))
                page_num = None
                if isinstance(page, int):
                    page_num = page
                else:
                    try:
                        page_num = int(page)
                    except Exception:
                        page_num = None
                files_pages.setdefault(fn, set())
                if page_num is not None:
                    files_pages[fn].add(page_num)

            top_fn = None
            if agg:
                try:
                    top_fn = max(agg.items(), key=lambda x: x[1]['score'])[0]
                except Exception:
                    top_fn = None

            def render_entry(fn, pages_set):
                if pages_set:
                    page_list = ", ".join(str(p) for p in sorted(pages_set))
                    st.markdown(f"<span style='color:#1f2937'><b>{fn}</b> (pages: {page_list})</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:#1f2937'><b>{fn}</b> (pages: —)</span>", unsafe_allow_html=True)

            if top_fn and top_fn in files_pages:
                render_entry(top_fn, files_pages[top_fn])
            else:
                for fn, pages in files_pages.items():
                    render_entry(fn, pages)


#
# ---- Chat input (with layout limit)
query = st.chat_input("Ask about WFH, leave, benefits, reimbursements…")
if query:
    # Apply spell correction
    query = correct_query_spelling(query)
    # Handle quick greetings/thanks (also show user bubble)
    low = query.lower().strip()
    if low in ["hi", "hello", "hey"]:
        with st.chat_message("user"):
            st.markdown(
                f"""<div style='display:flex; justify-content:flex-end; margin-bottom:6px;'>
                <div class="user-message chat-message" style='border-radius: 14px; padding: 6px 12px; font-size: 17px; font-weight:500; color:#1f2937; text-align:right;'>
                    {query}
                </div>
                </div>""",
                unsafe_allow_html=True
            )
        with st.chat_message("assistant"):
            st.markdown(
                "<div style='display:flex; justify-content:flex-start; margin-bottom:6px;'><div class='bot-msg chat-message' style='border-radius: 14px; padding: 6px 12px; font-size: 17px; color:#1f2937; text-align:left;'>Hello! How can I help you today?</div></div>",
                unsafe_allow_html=True,
            )
        st.session_state.chat.append(("user", query))
        st.session_state.chat.append(("assistant", "Hello! How can I help you today?"))
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
    if low in ["thanks", "thank you", "thx"]:
        with st.chat_message("user"):
            st.markdown(
                f"""<div style='display:flex; justify-content:flex-end; margin-bottom:6px;'>
                <div class="user-message chat-message" style='border-radius: 14px; padding: 6px 12px; font-size: 17px; font-weight:500; color:#1f2937; text-align:right;'>
                    {query}
                </div>
                </div>""",
                unsafe_allow_html=True
            )
        with st.chat_message("assistant"):
            st.markdown(
                "<div style='display:flex; justify-content:flex-start; margin-bottom:6px;'><div class='bot-msg chat-message' style='border-radius: 14px; padding: 6px 12px; font-size: 17px; color:#1f2937; text-align:left;'>You're welcome!</div></div>",
                unsafe_allow_html=True,
            )
        st.session_state.chat.append(("user", query))
        st.session_state.chat.append(("assistant", "You're welcome!"))
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()

    # 1) Show user's message immediately
    with st.chat_message("user"):
        st.markdown(
            f"""<div style='display:flex; justify-content:flex-end; margin-bottom:6px;'>
            <div class="user-message chat-message" style='border-radius: 14px; padding: 6px 12px; font-size: 17px; font-weight:500; color:#1f2937; text-align:right;'>
                {query}
            </div>
            </div>""",
            unsafe_allow_html=True
        )

    # 2) Option B follow-up merge
    if is_followup(st.session_state.last_question, query):
        composed = f"Previous question: {st.session_state.last_question}\nFollow-up: {query}"
    else:
        composed = query

    # 3) Stream assistant tokens
    with st.chat_message("assistant"):
        stream_area = st.empty()
        handler = StreamHandler(stream_area)
        result = qa.invoke({"question": composed}, config={"callbacks": [handler]})
        final_answer = handler.text if handler.text else result.get("answer", "")
        # Render assistant's message using Streamlit markdown for rich formatting
        stream_area.markdown(final_answer, unsafe_allow_html=False)

    # 4) Sources (current turn) — top file only with pages
    source_docs = result.get("source_documents", [])
    agg = aggregate_sources_with_scores(composed)
    if source_docs:
        with st.expander("Sources"):
            files_pages = {}
            for d in source_docs:
                meta = getattr(d, "metadata", {}) or {}
                fn = meta.get("source", "Unknown")
                page = meta.get("page_display", meta.get("page", None))
                page_num = None
                if isinstance(page, int):
                    page_num = page
                else:
                    try:
                        page_num = int(page)
                    except Exception:
                        page_num = None
                files_pages.setdefault(fn, set())
                if page_num is not None:
                    files_pages[fn].add(page_num)

            top_fn = None
            if agg:
                try:
                    top_fn = max(agg.items(), key=lambda x: x[1]['score'])[0]
                except Exception:
                    top_fn = None

            def render_entry(fn, pages_set):
                if pages_set:
                    page_list = ", ".join(str(p) for p in sorted(pages_set))
                    st.markdown(f"<span style='color:#1f2937'><b>{fn}</b> (pages: {page_list})</span>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<span style='color:#1f2937'><b>{fn}</b> (pages: —)</span>", unsafe_allow_html=True)

            if top_fn and top_fn in files_pages:
                render_entry(top_fn, files_pages[top_fn])
            else:
                for fn, pages in files_pages.items():
                    render_entry(fn, pages)

    # 5) Persist turn
    st.session_state.chat.append(("user", query))
    st.session_state.chat.append(("assistant", final_answer, source_docs, composed))
    st.session_state.last_question = query
st.markdown("</div>", unsafe_allow_html=True)

# ---- Footer
st.markdown("""
<div class="footer">© 2025 Sigmoid Analytics — Internal Use Only</div>
""", unsafe_allow_html=True)