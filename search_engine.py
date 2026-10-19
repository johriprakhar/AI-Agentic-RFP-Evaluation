"""
Semantic search engine for RFP supplier proposal documents.
Supports both non-LLM (BM25 + TF-IDF lexical-semantic scoring) and LLM-powered deep semantic search.
Tracks 1-based page and line numbers for every extracted chunk.
"""

import json
import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import pymupdf

from llm_providers import chat_json

# High-frequency English stop words
STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
    "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "could",
    "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has",
    "have", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if",
    "in", "into", "is", "it", "its", "itself", "just", "me", "more", "most", "my", "myself", "no", "nor",
    "not", "now", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over",
    "own", "s", "same", "she", "should", "so", "some", "such", "than", "that", "the", "their", "theirs",
    "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under",
    "until", "up", "very", "was", "we", "were", "what", "when", "where", "which", "while", "who", "whom",
    "why", "will", "with", "would", "you", "your", "yours"
}


def tokenize(text: str) -> List[str]:
    """Extract alphanumeric words and symbols (e.g. $240,000, ISO-27001)."""
    words = re.findall(r"[a-zA-Z0-9_\$]+", text.lower())
    return [w for w in words if len(w) > 1 and w not in STOP_WORDS]


def stem(word: str) -> str:
    """Lightweight English suffix normalizer for common morphological variations."""
    for suffix in ("tions", "tion", "ments", "ment", "ingly", "ings", "ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


def extract_document_chunks(
    pdf_bytes: bytes,
    filename: str,
    supplier_name: Optional[str] = None,
    window_lines: int = 5,
    step_lines: int = 3,
) -> List[Dict[str, Any]]:
    """
    Extract text page-by-page and chunk into line-level segments.
    Each chunk records:
      - document_name / filename
      - page number (1-based)
      - start_line & end_line (1-based on that page)
      - text: combined string
      - lines: list of (line_num, line_str)
    """
    doc_name = supplier_name or os.path.splitext(os.path.basename(filename))[0].replace("_", " ").title()
    chunks = []

    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page_idx, page in enumerate(doc, 1):
            raw_text = page.get_text("text")
            raw_lines = raw_text.splitlines()

            # Record non-empty lines with their 1-based line number on this page
            indexed_lines = [(i + 1, line.strip()) for i, line in enumerate(raw_lines) if line.strip()]
            if not indexed_lines:
                continue

            # Sliding window of lines to preserve context across sentences
            total_lines = len(indexed_lines)
            for i in range(0, total_lines, step_lines):
                window_slice = indexed_lines[i:i + window_lines]
                if not window_slice:
                    continue

                start_line = window_slice[0][0]
                end_line = window_slice[-1][0]
                chunk_text = " ".join(item[1] for item in window_slice)

                chunks.append({
                    "chunk_id": f"{doc_name}_p{page_idx}_l{start_line}",
                    "document_name": doc_name,
                    "filename": os.path.basename(filename),
                    "page": page_idx,
                    "start_line": start_line,
                    "end_line": end_line,
                    "text": chunk_text,
                    "lines": [{"line_num": num, "text": text} for num, text in window_slice],
                })

                # If this window reached the end of page lines, stop
                if i + window_lines >= total_lines:
                    break

    return chunks


def search_without_llm(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int = 10,
) -> Dict[str, Any]:
    """
    Semantic information retrieval using BM25 with term frequency saturation,
    inverse document frequency (IDF), document length normalization,
    n-gram matching, and query term coverage.
    """
    query_tokens = tokenize(query)
    query_stems = [stem(t) for t in query_tokens]
    if not query_tokens or not chunks:
        return {
            "query": query,
            "mode": "without_llm",
            "top_document": None,
            "top_document_score": 0.0,
            "document_rankings": [],
            "results": [],
        }

    # Document statistics
    N = len(chunks)
    chunk_tokens = [tokenize(c["text"]) for c in chunks]
    chunk_stems = [[stem(t) for t in tokens] for tokens in chunk_tokens]
    chunk_lens = [len(tokens) for tokens in chunk_tokens]
    avg_dl = max(1.0, sum(chunk_lens) / max(1, N))

    # Document frequency per stem
    df = Counter()
    for stems in chunk_stems:
        for s in set(stems):
            df[s] += 1

    # Inverted Document Frequency (BM25 IDF formula)
    idf = {}
    for s, count in df.items():
        idf[s] = math.log((N - count + 0.5) / (count + 0.5) + 1.0)

    k1 = 1.5
    b = 0.75

    raw_scores = []
    clean_query = query.strip().lower()

    for idx, c in enumerate(chunks):
        stems = chunk_stems[idx]
        dl = chunk_lens[idx]
        tf = Counter(stems)

        score = 0.0
        matched_terms = set()
        for q_stem, q_token in zip(query_stems, query_tokens):
            if q_stem in tf:
                matched_terms.add(q_token)
                freq = tf[q_stem]
                numerator = freq * (k1 + 1)
                denominator = freq + k1 * (1 - b + b * (dl / avg_dl))
                score += idf.get(q_stem, 0.5) * (numerator / denominator)

        # Exact phrase and bigram bonuses
        clean_text = c["text"].lower()
        if len(clean_query) > 3 and clean_query in clean_text:
            score += 4.5
        else:
            for i in range(len(query_tokens) - 1):
                bigram = f"{query_tokens[i]} {query_tokens[i+1]}"
                if bigram in clean_text:
                    score += 1.8

        # Coverage factor (percentage of distinct query terms present)
        coverage = len(matched_terms) / max(1, len(set(query_stems)))
        score = score * (0.6 + 0.4 * coverage)

        raw_scores.append((idx, score, matched_terms))

    raw_scores.sort(key=lambda x: x[1], reverse=True)
    max_score = raw_scores[0][1] if raw_scores and raw_scores[0][1] > 0 else 1.0

    scored_chunks = []
    for idx, r_score, matched in raw_scores:
        if r_score <= 0.05:
            continue
        pct = min(100.0, round((r_score / max_score) * 98.0, 1))
        scored_chunks.append({
            **chunks[idx],
            "score": pct,
            "raw_score": r_score,
            "matched_terms": sorted(list(matched)),
        })

    # Aggregate by document
    doc_rankings, top_doc = _aggregate_document_scores(scored_chunks, chunks)

    return {
        "query": query,
        "mode": "without_llm",
        "top_document": top_doc["document_name"] if top_doc else None,
        "top_document_score": top_doc["score"] if top_doc else 0.0,
        "top_document_summary": (
            f"{top_doc['document_name']} matched {top_doc['match_count']} relevant chunks with peak score of {top_doc['score']}%."
            if top_doc else "No matching document found."
        ),
        "document_rankings": doc_rankings,
        "results": scored_chunks[:top_k],
    }


def search_with_llm(
    query: str,
    chunks: List[Dict[str, Any]],
    llm_config: Dict[str, Any],
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    Semantic search using the configured LLM.
    Uses candidate retrieval to pre-select candidate chunks, then asks the LLM
    to semantically evaluate, score (0-100), extract relevant quotes, and explain matches.
    """
    if not chunks:
        return {
            "query": query,
            "mode": "with_llm",
            "top_document": None,
            "top_document_score": 0.0,
            "document_rankings": [],
            "results": [],
        }

    # Step 1: Pre-filter candidate chunks using BM25 so prompt remains within model token limits
    pre_filtered = search_without_llm(query, chunks, top_k=18)["results"]
    if not pre_filtered:
        # If strict lexical found no candidates, sample top chunks across all documents
        sampled = []
        seen_docs = set()
        for c in chunks:
            if c["document_name"] not in seen_docs:
                sampled.append(c)
                seen_docs.add(c["document_name"])
            if len(sampled) >= 12:
                break
        pre_filtered = sampled or chunks[:12]

    # Step 2: Prepare candidates for LLM prompt
    candidates = []
    for c in pre_filtered:
        candidates.append({
            "chunk_id": c["chunk_id"],
            "document_name": c["document_name"],
            "page": c["page"],
            "lines": f"{c['start_line']}-{c['end_line']}",
            "text": c["text"],
        })

    prompt = f"""
You are an expert procurement proposal analyst and semantic search engine.
A user is searching across uploaded supplier RFP proposals.

User Search Query:
"{query}"

Candidate Document Chunks:
{json.dumps(candidates, ensure_ascii=False, indent=2)}

Task Instructions:
1. Evaluate each chunk for semantic relevance to the search query. Understand synonyms, procurement terms, and implied requirements.
2. Assign a "relevance_score" from 0 to 100 for each chunk (0 = completely irrelevant, 100 = directly answers or matches the query).
3. Provide a brief 1-sentence "reasoning" explaining what in the chunk matches.
4. Extract the exact key "matched_quote" from the chunk.
5. Identify the "most_relevant_document" that best satisfies the user query overall.
6. Provide an "overall_answer" summarizing what the documents state regarding the query.

Return JSON ONLY with this schema:
{{
  "most_relevant_document": "Document Name",
  "overall_answer": "Concise 1-2 sentence direct answer to the query based on the proposals",
  "chunk_evaluations": [
    {{
      "chunk_id": "chunk_id_here",
      "relevance_score": 95,
      "reasoning": "Explicitly details the annual support fee and fixed implementation pricing.",
      "matched_quote": "Fixed implementation fee: $240,000. Annual support: $36,000."
    }}
  ]
}}
"""

    system_prompt = (
        "You are an objective document search engine. Return JSON only. "
        "Score relevance strictly based on the provided document text."
    )

    try:
        llm_reply = chat_json(llm_config, system_prompt, prompt)
    except Exception as e:
        # Fallback to without-LLM results if LLM call fails
        fallback = search_without_llm(query, chunks, top_k=top_k)
        fallback["mode"] = "with_llm (fallback to BM25)"
        fallback["llm_error"] = str(e)
        return fallback

    evaluations = {
        item.get("chunk_id"): item
        for item in llm_reply.get("chunk_evaluations", [])
        if item.get("chunk_id")
    }

    # Map scores back to original chunk objects
    chunk_map = {c["chunk_id"]: c for c in pre_filtered}
    scored_results = []
    for chunk_id, ev in evaluations.items():
        if chunk_id in chunk_map:
            score = float(ev.get("relevance_score", 0))
            if score > 5:
                scored_results.append({
                    **chunk_map[chunk_id],
                    "score": round(score, 1),
                    "reasoning": ev.get("reasoning", ""),
                    "matched_quote": ev.get("matched_quote", ""),
                })

    scored_results.sort(key=lambda x: x["score"], reverse=True)
    doc_rankings, top_doc = _aggregate_document_scores(scored_results, chunks)

    most_relevant_doc = llm_reply.get("most_relevant_document") or (top_doc["document_name"] if top_doc else None)
    top_score = top_doc["score"] if top_doc else (scored_results[0]["score"] if scored_results else 0.0)

    return {
        "query": query,
        "mode": "with_llm",
        "top_document": most_relevant_doc,
        "top_document_score": top_score,
        "overall_answer": llm_reply.get("overall_answer", ""),
        "document_rankings": doc_rankings,
        "results": scored_results[:top_k],
    }


def _aggregate_document_scores(
    scored_chunks: List[Dict[str, Any]],
    all_chunks: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Calculate aggregate relevance ranking per document from individual chunk scores."""
    all_docs = sorted(list({c["document_name"] for c in all_chunks}))
    doc_chunk_scores = {doc: [] for doc in all_docs}

    for c in scored_chunks:
        doc_chunk_scores[c["document_name"]].append(c["score"])

    doc_rankings = []
    for doc, scores in doc_chunk_scores.items():
        if not scores:
            doc_rankings.append({
                "document_name": doc,
                "score": 0.0,
                "max_score": 0.0,
                "avg_top_score": 0.0,
                "match_count": 0,
            })
            continue

        scores_sorted = sorted(scores, reverse=True)
        top1 = scores_sorted[0]
        # Weighted aggregate: 70% top chunk score + 30% average of top 3 chunk scores
        top3_avg = sum(scores_sorted[:3]) / min(3, len(scores_sorted))
        composite = round(0.7 * top1 + 0.3 * top3_avg, 1)

        doc_rankings.append({
            "document_name": doc,
            "score": composite,
            "max_score": top1,
            "avg_top_score": round(top3_avg, 1),
            "match_count": len(scores),
        })

    doc_rankings.sort(key=lambda x: (x["score"], x["match_count"]), reverse=True)
    top_doc = doc_rankings[0] if doc_rankings and doc_rankings[0]["score"] > 0 else None
    return doc_rankings, top_doc
