# 📊 Annual Report AI Chatbot

A **RAG (Retrieval-Augmented Generation)** chatbot that lets you upload annual reports as PDFs and ask questions about them in natural language. Built with a hybrid retrieval pipeline (FAISS semantic + BM25 keyword search) for accurate financial data extraction.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red?logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-1.2+-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PDF INGESTION                                │
│                                                                     │
│  PDF Upload ──► pdfplumber ──► Raw Text + Tables + Infographic Text │
│                  │                                                  │
│                  ├── extract_text()    (standard text)               │
│                  ├── extract_words()   (scattered/infographic text)  │
│                  └── extract_tables()  (structured table data)       │
│                                                                     │
│  Raw Text ──► RecursiveCharacterTextSplitter ──► Chunks (800ch/300ov)│
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DUAL INDEXING                                   │
│                                                                     │
│  Chunks ──► HuggingFace Embeddings ──► FAISS Vector Store           │
│         │   (multilingual-MiniLM-L12)                               │
│         │                                                           │
│         └── BM25Okapi Tokenizer ──► BM25 Inverted Index             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    HYBRID RETRIEVAL                                  │
│                                                                     │
│  User Query                                                         │
│      │                                                              │
│      ├──► BM25 Search ──► Top 4 (guaranteed, data-boosted)          │
│      │    • Term frequency + inverse document frequency             │
│      │    • 1.5x boost for RM/monetary values                      │
│      │    • 1.3x boost for 3+ digit numbers                        │
│      │    • 1.5x boost for "Key Highlights" sections               │
│      │    • 1.3x density bonus for short data-rich chunks           │
│      │                                                              │
│      └──► FAISS MMR Search ──► 8 candidates ──► Re-rank ──► Top 6  │
│           • Maximal Marginal Relevance (λ=0.7)                      │
│           • Cosine similarity re-ranking                            │
│                                                                     │
│  BM25 (4) + Semantic (6) ──► Deduplicate ──► Final Context (~8-10)  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      GENERATION                                     │
│                                                                     │
│  Context + Chat History + Question ──► Groq LLM ──► Answer          │
│                                        (Llama 3.3 70B)              │
│                                                                     │
│  System prompt enforces:                                            │
│  • Answer ONLY from context (no hallucination)                      │
│  • Distinguish PBT / PAT / PATAMI                                  │
│  • Reference page numbers                                          │
│  • Respond in user's language (Malay/English)                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Evaluation Results

Automated evaluation using 15 test cases across 5 difficulty levels, scored on 4 dimensions:

### Overall Scores

| Metric | Score | Description |
|--------|-------|-------------|
| **Context Recall** | 0.707 | Retrieval finds relevant chunks ~71% of the time |
| **Answer Relevancy** | 0.608 | Answers address the question ~61% of the time |
| **Faithfulness** | 0.848 | 85% of claims are grounded in retrieved context |
| **Hallucination Pass** | **15/15** | Zero hallucinations — never fabricates information |

### Scores by Difficulty Level

| Level | Questions | Recall | Relevancy | Faithfulness | Hallucination |
|-------|-----------|--------|-----------|-------------|---------------|
| L1 - Direct Extraction | 4 | 0.904 | 0.739 | 1.000 | 4/4 |
| L2 - Specific Lookup | 4 | 0.807 | 0.611 | 0.875 | 4/4 |
| L3 - Synthesis | 2 | 0.840 | 0.519 | 0.834 | 2/2 |
| L4 - Hallucination Traps | 3 | 0.551 | 0.515 | 0.778 | 3/3 |
| L5 - Malay Language | 2 | 0.211 | 0.572 | 0.611 | 2/2 |

### Key Findings

- **Zero hallucinations** across all test runs — the most critical metric for financial document QA
- **Perfect scores** on company name, operating revenue, PATAMI, and return on equity
- **Hybrid retrieval** (BM25 + semantic) significantly outperforms pure semantic search for financial data
- **Malay queries** work for revenue lookups but struggle with employee data (BM25 tokens don't cross languages; semantic search compensates partially)
- **Chairman identification** requires cross-referencing scattered governance sections — a known limitation of chunk-based RAG

## Tech Stack

| Component | Technology | Purpose | Cost |
|-----------|-----------|---------|------|
| Frontend | Streamlit | Chat interface | Free |
| LLM | Groq API (Llama 3.3 70B) | Answer generation | Free tier |
| Embeddings | HuggingFace `paraphrase-multilingual-MiniLM-L12-v2` | Multilingual vector embeddings | Free (local) |
| Vector Store | FAISS | Semantic similarity search | Free (local) |
| Keyword Search | BM25Okapi (`rank-bm25`) | Term-frequency retrieval | Free (local) |
| PDF Parser | pdfplumber | Text + table + infographic extraction | Free |
| Orchestration | LangChain (LCEL) | Prompt → LLM → Parse pipeline | Free |

**Total cost: $0**

## Quick Start

### Prerequisites
- Python 3.9+
- A free Groq API key from [console.groq.com](https://console.groq.com)

### Installation

```bash
git clone https://github.com/your-username/annual-report-chatbot.git
cd annual-report-chatbot

python -m venv venv
source venv/bin/activate       # Windows Git Bash: source venv/Scripts/activate
                                # Windows PowerShell: .\venv\Scripts\Activate

pip install -r requirements.txt

cp .env.example .env
# Edit .env and add your Groq API key: GROQ_API_KEY=gsk_your_key_here
```

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Run Evaluation

```bash
python eval_ragas.py path/to/annual_report.pdf
```

Outputs scores to terminal + timestamped JSON file.

## Usage

1. **API key** auto-loads from `.env` — sidebar shows ✅ if detected
2. **Upload PDF** in the sidebar and click **Process PDF**
3. **Ask questions** — try these:
   - *"What was the total operating revenue?"*
   - *"How many employees does the company have?"*
   - *"Compare revenue this year vs last year"*
   - *"What was the PATAMI?"*
   - *"Apakah hasil pendapatan syarikat?"* (Malay)

Each answer includes expandable **📎 Sources** showing the retrieved chunks.

## Configuration

All settings are in the `CONFIG` dict at the top of `app.py`:

```python
CONFIG = {
    "model": "llama-3.3-70b-versatile",   # Groq model
    "temperature": 0.1,                     # Lower = more factual
    "chunk_size": 800,                      # Characters per chunk
    "chunk_overlap": 300,                   # Overlap between chunks
    "semantic_k": 8,                        # FAISS MMR candidates
    "bm25_top": 4,                          # Guaranteed BM25 slots
    "semantic_top": 6,                      # Re-ranked semantic slots
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
}
```

## Project Structure

```
annual-report-chatbot/
├── app.py              # Main Streamlit application
├── eval_ragas.py       # Automated RAG evaluation script
├── debug_extract.py    # PDF extraction diagnostic tool
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variable template
├── .gitignore          # Git ignore rules
└── README.md           # This file
```

## Development Journey

This project was built iteratively across 9 versions, with each version addressing specific failures identified through systematic testing:

| Version | Change | Problem Solved |
|---------|--------|---------------|
| v1 | PyPDF2 + basic similarity search | Baseline — missed CEO, employees |
| v2 | Tuned chunk overlap (200→300), top-k (4→6) | Fixed CEO identification, broke revenue |
| v3 | MMR retrieval + cosine re-ranking | Recovered revenue, but inconsistent |
| v4 | pdfplumber with layout mode | Layout mode added noise, made everything worse |
| v5 | pdfplumber default + table extraction | Clean extraction, but retrieval still missed infographic data |
| v6 | Hybrid extraction (text + words + tables) | Confirmed data IS extracted — problem is retrieval |
| v7 | Added keyword search alongside semantic | Keyword results got demoted by re-ranker |
| v8 | Guaranteed keyword slots (not re-ranked) + query expansion | Fixed PATAMI, company name; employees still inconsistent |
| v9 | **BM25 replaces hardcoded keyword expansion** + multilingual embeddings + data-chunk boosting | Clean, generalizable solution with no hardcoded rules |

### Key Lessons Learned

1. **Extraction ≠ Retrieval ≠ Generation** — Debugging RAG requires isolating which component is failing. The `debug_extract.py` script proved the data was extracted correctly; the problem was always retrieval.

2. **Semantic search alone is not enough for financial data** — Embedding models prioritize meaning over exact terms. "How many employees?" is semantically similar to "employees are the foundation of our success" (narrative) AND "NUMBER OF EMPLOYEES 710" (data). BM25 bridges this gap with term-frequency scoring.

3. **Infographic pages need special extraction** — Standard `extract_text()` misses scattered text in designed layouts. The `extract_words()` method reads individual text objects and reconstructs lines by Y-coordinate, catching data from Key Highlights pages.

4. **Guaranteed slots > unified re-ranking** — When combining keyword and semantic results, putting everything through one re-ranker lets the semantic scorer demote data-rich but short chunks. Keeping BM25 results as guaranteed context slots solved this.

5. **BM25 data boosting is a generalizable pattern** — Instead of hardcoding query expansions ("if user asks about employees, also search for headcount"), boosting chunks that contain numerical data (RM values, 3+ digit numbers) is a general signal that works across all query types.

6. **Zero hallucination is achievable with prompt engineering** — The system prompt's instruction to "answer ONLY from context" combined with low temperature (0.1) achieved 15/15 hallucination pass rate consistently.

## Limitations

- **Scanned PDFs** — Image-based PDFs require OCR preprocessing (e.g., `pytesseract`)
- **Complex tables** — Multi-level header tables may not extract cleanly
- **Cross-reference questions** — Answers requiring synthesis across distant sections (e.g., "Who is the Chairman?") depend on retrieval finding all relevant chunks
- **Groq free tier** — 100K tokens/day limit; heavy usage may trigger rate limits
- **Malay keyword retrieval** — BM25 operates on exact tokens, so Malay queries rely on semantic search for cross-language matching

## Future Enhancements

- [ ] Cross-encoder re-ranking (e.g., `ms-marco-MiniLM-L-6-v2`) for higher precision
- [ ] Multi-PDF support for cross-report comparison
- [ ] OCR integration for scanned documents
- [ ] Persistent vector store (save/load FAISS index to disk)
- [ ] Deployment to Streamlit Community Cloud
- [ ] Fine-tuned embeddings on financial document corpus

## License

MIT License — free to use, modify, and distribute.

---

*Built by Muhammad Ayman Abd Rahman*
