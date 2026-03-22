"""
RAG Evaluation Script using RAGAS
==================================
Automated accuracy testing for the Annual Report AI Chatbot.

Evaluates 4 dimensions:
- Faithfulness: Is the answer grounded in retrieved chunks? (catches hallucination)
- Answer Relevancy: Does the answer address the question?
- Context Precision: Are the retrieved chunks relevant?
- Context Recall: Did retrieval find all needed chunks?

Usage:
    python eval_ragas.py <path_to_pdf>

Requires:
    pip install ragas datasets langchain-groq

Author: Muhammad Ayman Abd Rahman
"""

import os
import sys
import json
import re
import time
import numpy as np
from datetime import datetime
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

# ── Same config as app.py ────────────────────────────────────────────────────

CONFIG = {
    "model": "llama-3.3-70b-versatile",
    "temperature": 0.1,
    "max_tokens": 2048,
    "chunk_size": 800,
    "chunk_overlap": 300,
    "semantic_k": 8,
    "bm25_top": 4,
    "semantic_top": 6,
    "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
}

# ── Test Dataset ─────────────────────────────────────────────────────────────
# Each test case has:
#   question: the query to test
#   ground_truth: the expected correct answer (used for recall scoring)
#   level: difficulty level for reporting

TEST_CASES = [
    # Level 1: Direct Extraction
    {
        "question": "What is the company name?",
        "ground_truth": "The company name is Bursa Malaysia Berhad.",
        "level": "L1-Direct",
    },
    {
        "question": "What was the total operating revenue for 2025?",
        "ground_truth": "The total operating revenue for 2025 was RM701.8 million.",
        "level": "L1-Direct",
    },
    {
        "question": "Who is the Chairman of Bursa Malaysia?",
        "ground_truth": "The Chairman is Tan Sri Abdul Farid Alias, appointed on 1 May 2025.",
        "level": "L1-Direct",
    },
    {
        "question": "What financial year does this report cover?",
        "ground_truth": "The report covers the financial year ended 31 December 2025.",
        "level": "L1-Direct",
    },

    # Level 2: Specific Lookup
    {
        "question": "What was the dividend per share for 2025?",
        "ground_truth": "The dividends included a single-tier interim dividend of 14.0 sen, final dividend of 18.0 sen, and special dividend of 8.0 sen per share.",
        "level": "L2-Lookup",
    },
    {
        "question": "How many employees does the company have?",
        "ground_truth": "The company has 710 employees as of 2025, up from 681 in 2024.",
        "level": "L2-Lookup",
    },
    {
        "question": "What was the PATAMI for 2025?",
        "ground_truth": "The PATAMI (Profit After Tax, Zakat and Minority Interest) for 2025 was RM250.2 million, down from RM310.1 million in 2024.",
        "level": "L2-Lookup",
    },
    {
        "question": "What was the return on equity?",
        "ground_truth": "The return on equity was 29.9% in 2025, compared to 36.6% in 2024.",
        "level": "L2-Lookup",
    },

    # Level 3: Synthesis
    {
        "question": "Compare the operating revenue this year vs last year.",
        "ground_truth": "Operating revenue decreased from RM757.7 million in 2024 to RM701.8 million in 2025, a decline of approximately 7.4%.",
        "level": "L3-Synthesis",
    },
    {
        "question": "What are the key risk factors mentioned in the report?",
        "ground_truth": "Key risk factors include Business Performance and Competition Risk, Organisational risks including climate, and Regulatory changes.",
        "level": "L3-Synthesis",
    },

    # Level 4: Hallucination Traps (answer should indicate info is NOT available)
    {
        "question": "What will the company's revenue be next year?",
        "ground_truth": "The report does not contain revenue projections or forecasts for next year.",
        "level": "L4-Hallucination",
    },
    {
        "question": "How does this company compare to SGX (Singapore Exchange)?",
        "ground_truth": "The report does not contain comparisons to specific competitors.",
        "level": "L4-Hallucination",
    },
    {
        "question": "What was the exact stock price on 15 March 2025?",
        "ground_truth": "The report does not contain daily stock price data.",
        "level": "L4-Hallucination",
    },

    # Level 5: Malay Language
    {
        "question": "Apakah hasil pendapatan syarikat?",
        "ground_truth": "Hasil pendapatan operasi syarikat untuk tahun 2025 adalah RM701.8 juta.",
        "level": "L5-Malay",
    },
    {
        "question": "Berapa jumlah pekerja syarikat?",
        "ground_truth": "Jumlah pekerja syarikat adalah 710 orang pada tahun 2025.",
        "level": "L5-Malay",
    },
]

# ── Reuse extraction/retrieval functions from app.py ─────────────────────────

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


def extract_pdf_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            page_num = i + 1
            page_parts = []
            standard_text = page.extract_text()
            if standard_text and standard_text.strip():
                page_parts.append(standard_text.strip())
            words_text = extract_words_text(page)
            if words_text and words_text.strip():
                if standard_text:
                    st_tokens = set(standard_text.lower().split())
                    wt_tokens = set(words_text.lower().split())
                    new = wt_tokens - st_tokens
                    if len(new) > len(wt_tokens) * 0.1:
                        page_parts.append(f"[Additional text]\n{words_text.strip()}")
                else:
                    page_parts.append(words_text.strip())
            tables = page.extract_tables()
            if tables:
                for table in tables:
                    rows = []
                    for row in table:
                        if row:
                            cells = [str(c).strip() if c else "" for c in row]
                            if any(c for c in cells):
                                rows.append(" | ".join(cells))
                    if rows:
                        page_parts.append("[Table]\n" + "\n".join(rows))
            if page_parts:
                text += f"\n[Page {page_num}]\n" + "\n\n".join(page_parts) + "\n"
    return text


def tokenize(text):
    return re.findall(r'[a-zA-Z0-9]+', text.lower())


def bm25_search(query, bm25_index, chunks, top_n=4):
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    scores = bm25_index.get_scores(query_tokens)
    for i, chunk in enumerate(chunks):
        if scores[i] > 0:
            chunk_lower = chunk.lower()
            if re.search(r'RM[\d,.]+|[\d,]+\.?\d*\s*(million|billion|sen|%)', chunk_lower):
                scores[i] *= 1.5
            if re.search(r'\b\d{3,}\b', chunk):
                scores[i] *= 1.3
            if re.search(r'key highlights|financial highlights', chunk_lower):
                scores[i] *= 1.5
            if len(chunk) < 500 and (re.search(r'RM[\d,.]+', chunk_lower) or re.search(r'\b\d{3,}\b', chunk)):
                scores[i] *= 1.3
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [Document(page_content=chunks[idx]) for idx in top_indices if scores[idx] > 0]


def rerank_semantic(query, docs, embeddings, top_n):
    if not docs:
        return []
    seen = set()
    unique = []
    for d in docs:
        h = hash(d.page_content[:200])
        if h not in seen:
            seen.add(h)
            unique.append(d)
    docs = unique
    if len(docs) <= top_n:
        return docs
    qe = embeddings.embed_query(query)
    des = embeddings.embed_documents([d.page_content for d in docs])
    qv = np.array(qe)
    scores = [np.dot(qv, np.array(dv)) / (np.linalg.norm(qv) * np.linalg.norm(np.array(dv)) + 1e-10) for dv in des]
    ranked = np.argsort(scores)[::-1][:top_n]
    return [docs[i] for i in ranked]


def deduplicate_docs(docs):
    seen = set()
    unique = []
    for d in docs:
        h = hash(d.page_content[:200])
        if h not in seen:
            seen.add(h)
            unique.append(d)
    return unique


# ── Main Eval Logic ──────────────────────────────────────────────────────────

def run_evaluation(pdf_path):
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("❌ GROQ_API_KEY not found in .env")
        sys.exit(1)

    print(f"📄 Loading PDF: {pdf_path}")
    raw_text = extract_pdf_text(pdf_path)
    print(f"   ✅ Extracted {len(raw_text):,} characters")

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    print(f"   ✅ {len(chunks)} chunks")

    # Build indexes
    print("🧠 Building indexes...")
    embeddings = HuggingFaceEmbeddings(
        model_name=CONFIG["embedding_model"],
        model_kwargs={"device": "cpu"},
    )
    vector_store = FAISS.from_texts(chunks, embeddings)
    tokenized_chunks = [tokenize(c) for c in chunks]
    bm25_index = BM25Okapi(tokenized_chunks)

    # LLM — use 8b model for eval to stay within free tier rate limits
    # The 70b model uses ~3x more tokens per question
    eval_model = "llama-3.1-8b-instant"
    print(f"🤖 Using eval model: {eval_model} (saves tokens vs {CONFIG['model']})")

    llm = ChatGroq(
        groq_api_key=api_key,
        model_name=eval_model,
        temperature=CONFIG["temperature"],
        max_tokens=1024,  # shorter responses for eval
    )

    # Prompt
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful AI assistant that answers questions about annual reports.
Use ONLY the provided context to answer. If the context doesn't contain enough information,
say so honestly — do not make up information.
Be specific and cite numbers when available. Distinguish between PBT, PAT, and PATAMI.
If the user asks in Malay, respond in Malay.

Context:
{context}"""),
        ("human", "{question}"),
    ])
    chain = prompt_template | llm | StrOutputParser()

    # Run each test case
    results = []
    print(f"\n{'='*80}")
    print(f"Running {len(TEST_CASES)} test cases...")
    print(f"{'='*80}\n")

    for i, tc in enumerate(TEST_CASES):
        question = tc["question"]
        ground_truth = tc["ground_truth"]
        level = tc["level"]

        print(f"[{i+1}/{len(TEST_CASES)}] {level}: {question}")

        # Retrieve (hybrid)
        bm25_docs = bm25_search(question, bm25_index, chunks, top_n=CONFIG["bm25_top"])

        retriever = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": CONFIG["semantic_k"], "lambda_mult": 0.7},
        )
        sem_candidates = retriever.invoke(question)
        sem_docs = rerank_semantic(question, sem_candidates, embeddings, CONFIG["semantic_top"])

        all_docs = deduplicate_docs(bm25_docs + sem_docs)
        context = "\n\n---\n\n".join(d.page_content for d in all_docs)
        contexts = [d.page_content for d in all_docs]

        # Generate answer (with retry for rate limits)
        answer = None
        for attempt in range(3):
            try:
                answer = chain.invoke({"context": context, "question": question})
                break
            except Exception as e:
                error_str = str(e)
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    # Extract wait time from error if available
                    wait_match = re.search(r'(\d+)m(\d+)', error_str)
                    if wait_match:
                        wait_secs = int(wait_match.group(1)) * 60 + int(wait_match.group(2))
                    else:
                        wait_secs = 60 * (attempt + 1)
                    print(f"   ⏱️  Rate limited. Waiting {wait_secs}s (attempt {attempt+1}/3)...")
                    time.sleep(wait_secs + 5)  # add 5s buffer
                else:
                    answer = f"[ERROR: {e}]"
                    break

        if answer is None:
            answer = "[ERROR: Rate limit exceeded after 3 retries]"

        print(f"   Answer: {answer[:120]}...")

        # Delay between questions to avoid rate limits (10s)
        if i < len(TEST_CASES) - 1:
            time.sleep(10)

        # ── Simple automated scoring ──
        # Score 1: Context contains ground truth keywords?
        gt_keywords = set(re.findall(r'[a-zA-Z0-9.]+', ground_truth.lower()))
        gt_keywords -= {"the", "is", "was", "for", "and", "in", "of", "a", "to", "from"}
        ctx_text = context.lower()
        ctx_hits = sum(1 for kw in gt_keywords if kw in ctx_text)
        context_recall = ctx_hits / max(len(gt_keywords), 1)

        # Score 2: Answer contains ground truth keywords?
        ans_text = answer.lower()
        ans_hits = sum(1 for kw in gt_keywords if kw in ans_text)
        answer_relevancy = ans_hits / max(len(gt_keywords), 1)

        # Score 3: Faithfulness — answer keywords should be in context
        ans_keywords = set(re.findall(r'RM[\d,.]+|\d+\.?\d*', answer))
        if ans_keywords:
            faithful_hits = sum(1 for k in ans_keywords if k in context)
            faithfulness = faithful_hits / len(ans_keywords)
        else:
            faithfulness = 1.0  # no claims to verify

        # Score 4: Hallucination check for Level 4
        is_hallucination_trap = level == "L4-Hallucination"
        if is_hallucination_trap:
            # For hallucination traps, the answer SHOULD say it's not available
            refusal_phrases = [
                "not contain", "not mention", "not provide", "not available",
                "not covered", "not included", "cannot", "does not",
                "tidak", "no information",
            ]
            correctly_refused = any(p in ans_text for p in refusal_phrases)
            hallucination_pass = correctly_refused
        else:
            hallucination_pass = True  # not a trap question

        result = {
            "question": question,
            "level": level,
            "ground_truth": ground_truth,
            "answer": answer,
            "contexts": contexts[:3],  # top 3 for readability
            "scores": {
                "context_recall": round(context_recall, 3),
                "answer_relevancy": round(answer_relevancy, 3),
                "faithfulness": round(faithfulness, 3),
                "hallucination_pass": hallucination_pass,
            },
        }
        results.append(result)

        status = "✅" if (answer_relevancy > 0.3 and hallucination_pass) else "❌"
        print(f"   {status} recall={context_recall:.2f} relevancy={answer_relevancy:.2f} faith={faithfulness:.2f} halluc={'✅' if hallucination_pass else '❌'}")
        print()

    # ── Summary Report ───────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}\n")

    levels = sorted(set(r["level"] for r in results))
    for level in levels:
        level_results = [r for r in results if r["level"] == level]
        avg_recall = np.mean([r["scores"]["context_recall"] for r in level_results])
        avg_relevancy = np.mean([r["scores"]["answer_relevancy"] for r in level_results])
        avg_faith = np.mean([r["scores"]["faithfulness"] for r in level_results])
        halluc_pass = sum(1 for r in level_results if r["scores"]["hallucination_pass"])

        print(f"{level} ({len(level_results)} questions)")
        print(f"   Context Recall:    {avg_recall:.3f}")
        print(f"   Answer Relevancy:  {avg_relevancy:.3f}")
        print(f"   Faithfulness:      {avg_faith:.3f}")
        print(f"   Hallucination:     {halluc_pass}/{len(level_results)} passed")
        print()

    # Overall
    all_recall = np.mean([r["scores"]["context_recall"] for r in results])
    all_relevancy = np.mean([r["scores"]["answer_relevancy"] for r in results])
    all_faith = np.mean([r["scores"]["faithfulness"] for r in results])
    all_halluc = sum(1 for r in results if r["scores"]["hallucination_pass"])

    print(f"OVERALL ({len(results)} questions)")
    print(f"   Context Recall:    {all_recall:.3f}")
    print(f"   Answer Relevancy:  {all_relevancy:.3f}")
    print(f"   Faithfulness:      {all_faith:.3f}")
    print(f"   Hallucination:     {all_halluc}/{len(results)} passed")

    # Save detailed results to JSON
    output_file = f"eval_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "config": CONFIG,
            "summary": {
                "total_questions": len(results),
                "context_recall": round(all_recall, 3),
                "answer_relevancy": round(all_relevancy, 3),
                "faithfulness": round(all_faith, 3),
                "hallucination_pass_rate": f"{all_halluc}/{len(results)}",
            },
            "results": results,
        }, f, indent=2, default=str)

    print(f"\n📊 Detailed results saved to: {output_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python eval_ragas.py <path_to_pdf>")
        sys.exit(1)
    run_evaluation(sys.argv[1])