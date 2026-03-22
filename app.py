"""
Annual Report AI Chatbot
========================
A RAG (Retrieval-Augmented Generation) chatbot that lets you upload
annual reports (PDF) and ask questions about them using natural language.

Built with: Streamlit · LangChain · Groq · FAISS · HuggingFace Embeddings

Author: Muhammad Ayman Abd Rahman
"""

import os
import re
import numpy as np
import streamlit as st
from dotenv import load_dotenv

import pdfplumber
from rank_bm25 import BM25Okapi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

load_dotenv()

# ── Configuration ────────────────────────────────────────────────────────────

CONFIG = {
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.1,
    "max_tokens": 2048,
    "chunk_size": 800,
    "chunk_overlap": 300,
    "semantic_k": 8,
    "bm25_top": 4,           # guaranteed BM25 keyword slots
    "semantic_top": 6,        # semantic slots after re-ranking
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
}

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# ── Page Config & Styling ────────────────────────────────────────────────────

st.set_page_config(
    page_title="Annual Report Chatbot",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.3rem 0 0 0; opacity: 0.85; font-size: 0.95rem; }
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #f8fafc 0%, #eef2f7 100%);
    }
    .status-ready    { background: #d1fae5; color: #065f46; padding: 0.5rem 1rem; border-radius: 8px; text-align: center; font-weight: 600; }
    .status-waiting  { background: #fef3c7; color: #92400e; padding: 0.5rem 1rem; border-radius: 8px; text-align: center; font-weight: 600; }
    .source-box {
        background: #f1f5f9;
        border-left: 3px solid #2d6a9f;
        padding: 0.6rem 1rem;
        border-radius: 0 8px 8px 0;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: #334155;
    }
    .config-info {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: #64748b;
    }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ─────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={"device": "cpu"},
    )


def extract_words_text(page):
    words = page.extract_words(keep_blank_chars=True, x_tolerance=3, y_tolerance=3)
    if not words:
        return ""
    lines = []
    current_line = [words[0]]
    for w in words[1:]:
        if abs(w["top"] - current_line[-1]["top"]) < 5:
            current_line.append(w)
        else:
            current_line.sort(key=lambda x: x["x0"])
            lines.append(" ".join(ww["text"] for ww in current_line))
            current_line = [w]
    if current_line:
        current_line.sort(key=lambda x: x["x0"])
        lines.append(" ".join(ww["text"] for ww in current_line))
    return "\n".join(lines)


def extract_pdf_text(pdf_file) -> str:
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            page_parts = []

            standard_text = page.extract_text()
            if standard_text and standard_text.strip():
                page_parts.append(standard_text.strip())

            words_text = extract_words_text(page)
            if words_text and words_text.strip():
                if standard_text:
                    standard_tokens = set(standard_text.lower().split())
                    words_tokens = set(words_text.lower().split())
                    new_tokens = words_tokens - standard_tokens
                    if len(new_tokens) > len(words_tokens) * 0.1:
                        page_parts.append(f"[Additional text from page layout]\n{words_text.strip()}")
                else:
                    page_parts.append(words_text.strip())

            tables = page.extract_tables()
            if tables:
                for table in tables:
                    rows = []
                    for row in table:
                        if row:
                            cells = [str(cell).strip() if cell else "" for cell in row]
                            if any(c for c in cells):
                                rows.append(" | ".join(cells))
                    if rows:
                        page_parts.append("[Table]\n" + "\n".join(rows))

            if page_parts:
                combined = "\n\n".join(page_parts)
                text += f"\n[Page {page_num}]\n{combined}\n"

    return text


def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_text(text)


def build_vector_store(chunks, embeddings):
    return FAISS.from_texts(chunks, embeddings)


def tokenize(text):
    """Simple tokenizer for BM25 — lowercase, split on non-alphanumeric."""
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def build_bm25_index(chunks):
    """Build a BM25 index over the raw chunks."""
    tokenized = [tokenize(chunk) for chunk in chunks]
    return BM25Okapi(tokenized)


def bm25_search(query, bm25_index, chunks, top_n=4):
    """
    BM25 keyword retrieval — industry-standard term-frequency scoring.
    Automatically handles:
    - Term frequency (tf): words appearing more in a chunk score higher
    - Inverse document frequency (idf): rare terms are weighted more
    - Document length normalization: short data-rich chunks aren't penalized
    No hardcoded rules or query expansion needed.
    """
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    scores = bm25_index.get_scores(query_tokens)

    # Boost chunks that contain numerical/monetary data
    # BM25 alone doesn't know that "RM701.8 million" is more useful
    # than a paragraph discussing revenue conceptually
    for i, chunk in enumerate(chunks):
        if scores[i] > 0:
            chunk_lower = chunk.lower()
            has_money = bool(re.search(r'RM[\d,.]+|[\d,]+\.?\d*\s*(million|billion|sen|%)', chunk_lower))
            has_standalone_numbers = bool(re.search(r'\b\d{3,}\b', chunk))  # 3+ digit numbers like 710, 681
            has_key_section = bool(re.search(r'key highlights|financial highlights', chunk_lower))

            if has_money:
                scores[i] *= 1.5
            if has_standalone_numbers:
                scores[i] *= 1.3   # boost data chunks with plain numbers
            if has_key_section:
                scores[i] *= 1.5
            # Short chunks with data are more likely to be summary/highlight sections
            # than long narrative chunks — give them a density bonus
            if len(chunk) < 500 and (has_money or has_standalone_numbers):
                scores[i] *= 1.3

    top_indices = np.argsort(scores)[::-1][:top_n]
    results = []
    for idx in top_indices:
        if scores[idx] > 0:
            results.append(Document(page_content=chunks[idx]))
    return results


def format_docs(docs):
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def rerank_semantic(query, docs, embeddings, top_n):
    if not docs:
        return []
    seen = set()
    unique = []
    for doc in docs:
        h = hash(doc.page_content[:200])
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    docs = unique
    if len(docs) <= top_n:
        return docs

    query_emb = embeddings.embed_query(query)
    doc_embs = embeddings.embed_documents([d.page_content for d in docs])

    qv = np.array(query_emb)
    scores = []
    for dv in doc_embs:
        dv = np.array(dv)
        sim = np.dot(qv, dv) / (np.linalg.norm(qv) * np.linalg.norm(dv) + 1e-10)
        scores.append(sim)

    ranked = np.argsort(scores)[::-1][:top_n]
    return [docs[i] for i in ranked]


def deduplicate_docs(docs):
    seen = set()
    unique = []
    for doc in docs:
        h = hash(doc.page_content[:200])
        if h not in seen:
            seen.add(h)
            unique.append(doc)
    return unique


# ── RAG Prompt ───────────────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful AI assistant that answers questions about annual reports.
Use ONLY the provided context to answer. If the context doesn't contain enough information,
say so honestly — do not make up information.

When answering:
- Be specific and cite numbers, figures, and monetary values when available in the context
- Keep answers clear and concise
- If asked about something not in the report, say it's not covered in the provided document
- When names and titles/roles are mentioned together, identify them correctly (e.g. Chairman, CEO, Director)
- If the context contains financial tables or figures, extract and present the relevant numbers
- When page numbers are visible in the context (e.g. [Page 45]), reference them in your answer
- Data separated by | in the context represents table columns — read them as structured data
- If the user asks in Malay or another language, respond in the same language using data from the context
- Be careful to distinguish between similar financial metrics:
  * PBT = Profit Before Tax
  * PAT = Profit After Tax
  * PATAMI = Profit After Tax, Zakat and Minority Interest
  Do not confuse these — each has a different figure
- Look for Key Highlights or summary sections which often contain the most important headline figures

Context from the annual report:
{context}

Previous conversation:
{chat_history}"""),
    ("human", "{question}"),
])


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/financial-analytics.png", width=64)
    st.markdown("### 📊 Annual Report Chatbot")
    st.markdown("---")

    if GROQ_API_KEY:
        st.markdown("🔑 **API Key:** ✅ Loaded from `.env`")
    else:
        st.markdown("🔑 **API Key:** ❌ Missing")
        st.error("Add your Groq API key to the `.env` file:\n\n`GROQ_API_KEY=gsk_your_key_here`")

    st.markdown("---")
    st.markdown("### 📄 Upload Annual Report")
    uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"],
        help="Upload the annual report you want to chat with.")
    st.markdown("---")

    if "vector_store" in st.session_state:
        st.markdown('<div class="status-ready">✅ Knowledge base ready</div>', unsafe_allow_html=True)
        st.caption(f"📦 {st.session_state.get('chunk_count', '?')} chunks indexed")
        st.caption(f"📄 {st.session_state.get('file_name', '')}")
    else:
        st.markdown('<div class="status-waiting">⏳ Awaiting PDF upload</div>', unsafe_allow_html=True)

    process_btn = st.button("🚀 Process PDF", use_container_width=True,
                            disabled=not uploaded_file or not GROQ_API_KEY)
    st.markdown("---")

    with st.expander("ℹ️ Model Info"):
        st.markdown(f"""
<div class="config-info">
<strong>Model:</strong> {CONFIG['model']}<br>
<strong>Temperature:</strong> {CONFIG['temperature']}<br>
<strong>Chunks:</strong> {CONFIG['chunk_size']}ch / {CONFIG['chunk_overlap']}ov<br>
<strong>Retrieval:</strong> Hybrid (FAISS semantic {CONFIG['semantic_top']} + BM25 keyword {CONFIG['bm25_top']})<br>
<strong>Embeddings:</strong> {CONFIG['embedding_model']}<br>
<strong>PDF Parser:</strong> pdfplumber (text + words + tables)
</div>
""", unsafe_allow_html=True)

# ── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>📊 Annual Report AI Chatbot</h1>
    <p>Upload an annual report and ask anything — powered by RAG + Groq</p>
</div>
""", unsafe_allow_html=True)

# ── Process PDF ──────────────────────────────────────────────────────────────

if process_btn and uploaded_file and GROQ_API_KEY:
    with st.status("Processing your annual report...", expanded=True) as status:
        st.write("📖 Extracting text, infographics, and tables from PDF...")
        raw_text = extract_pdf_text(uploaded_file)

        if not raw_text.strip():
            st.error("Could not extract text. PDF may be scanned/image-based.")
            st.stop()

        st.write(f"   ✅ Extracted {len(raw_text):,} characters")

        st.write("✂️ Splitting into chunks...")
        chunks = chunk_text(raw_text)
        st.write(f"   ✅ Created {len(chunks)} chunks")

        st.write("🧠 Building vector + BM25 indexes...")
        embeddings = load_embeddings()
        vector_store = build_vector_store(chunks, embeddings)
        bm25_index = build_bm25_index(chunks)

        st.session_state["vector_store"] = vector_store
        st.session_state["bm25_index"] = bm25_index
        st.session_state["raw_chunks"] = chunks
        st.session_state["chunk_count"] = len(chunks)
        st.session_state["file_name"] = uploaded_file.name
        st.session_state["messages"] = []

        status.update(label="✅ Knowledge base ready!", state="complete", expanded=False)
    st.rerun()

# ── Chat Interface ───────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state["messages"] = []

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📎 Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f'<div class="source-box"><strong>Chunk {i}:</strong> {src[:300]}{"..." if len(src) > 300 else ""}</div>',
                        unsafe_allow_html=True,
                    )

if prompt := st.chat_input("Ask a question about the annual report..."):
    if not GROQ_API_KEY:
        st.warning("Please add your Groq API key to the `.env` file.")
        st.stop()
    if "vector_store" not in st.session_state:
        st.warning("Please upload and process a PDF first.")
        st.stop()

    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                llm = ChatGroq(
                    groq_api_key=GROQ_API_KEY,
                    model_name=CONFIG["model"],
                    temperature=CONFIG["temperature"],
                    max_tokens=CONFIG["max_tokens"],
                )

                # ── HYBRID RETRIEVAL ──
                # BM25 results guaranteed in context (keyword precision)
                # Semantic results re-ranked separately (meaning coverage)

                # 1. BM25 keyword search — guaranteed slots
                bm25_docs = bm25_search(
                    query=prompt,
                    bm25_index=st.session_state["bm25_index"],
                    chunks=st.session_state["raw_chunks"],
                    top_n=CONFIG["bm25_top"],
                )

                # 2. Semantic search via MMR — then re-rank
                retriever = st.session_state["vector_store"].as_retriever(
                    search_type="mmr",
                    search_kwargs={
                        "k": CONFIG["semantic_k"],
                        "lambda_mult": 0.7,
                    },
                )
                semantic_candidates = retriever.invoke(prompt)

                embeddings = load_embeddings()
                sem_docs = rerank_semantic(
                    query=prompt,
                    docs=semantic_candidates,
                    embeddings=embeddings,
                    top_n=CONFIG["semantic_top"],
                )

                # 3. Combine: BM25 first (guaranteed), then semantic
                all_docs = deduplicate_docs(bm25_docs + sem_docs)

                context = format_docs(all_docs)
                sources = [doc.page_content for doc in all_docs]

                # Chat history
                chat_history = ""
                recent_msgs = st.session_state["messages"][-8:]
                if len(recent_msgs) > 1:
                    for m in recent_msgs[:-1]:
                        role = "User" if m["role"] == "user" else "Assistant"
                        chat_history += f"{role}: {m['content']}\n"

                chain = RAG_PROMPT | llm | StrOutputParser()

                answer = chain.invoke({
                    "context": context,
                    "chat_history": chat_history if chat_history else "None",
                    "question": prompt,
                })

                st.markdown(answer)

                if sources:
                    with st.expander("📎 Sources"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(
                                f'<div class="source-box"><strong>Chunk {i}:</strong> {src[:300]}{"..." if len(src) > 300 else ""}</div>',
                                unsafe_allow_html=True,
                            )

                st.session_state["messages"].append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                })

            except Exception as e:
                error_msg = str(e)
                if "api_key" in error_msg.lower() or "auth" in error_msg.lower():
                    st.error("❌ Invalid API key. Check your `.env` file.")
                elif "rate_limit" in error_msg.lower():
                    st.error("⏱️ Rate limited. Wait a moment and try again.")
                else:
                    st.error(f"❌ Error: {error_msg}")

# ── Empty State ──────────────────────────────────────────────────────────────

if "vector_store" not in st.session_state:
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("#### 1️⃣ Setup")
        st.markdown("Add your Groq API key to `.env`. Get a free key at [console.groq.com](https://console.groq.com).")
    with col2:
        st.markdown("#### 2️⃣ Upload PDF")
        st.markdown("Drop your annual report PDF in the sidebar.")
    with col3:
        st.markdown("#### 3️⃣ Start Chatting")
        st.markdown("Ask any question — financials, strategy, governance, ESG, and more.")
    st.markdown("---")
    st.markdown('<center><sub>Built with Streamlit · LangChain · Groq · FAISS · BM25 · HuggingFace · pdfplumber</sub></center>', unsafe_allow_html=True)
