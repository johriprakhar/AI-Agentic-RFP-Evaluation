"""
Pure lexical retrieval: tokenisation, line-indexed chunking and BM25 ranking.

This module has no LLM dependency on purpose. It is imported both by
`search_engine` (which layers semantic re-ranking on top) and by
`pdf_processor` (which uses it to pick the most relevant passages when a
proposal is too long to send to a model whole). Keeping it model-free means the
document-preparation path never drags in a provider SDK and stays testable
offline.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pymupdf

# BM25 parameters: term-frequency saturation and length normalisation.
K1 = 1.5
B = 0.75

#: Sliding-window chunk geometry, in lines.
WINDOW_LINES = 5
STEP_LINES = 3

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
    "why", "will", "with", "would", "you", "your", "yours",
}


def tokenize(text: str) -> List[str]:
    """Extract alphanumeric words and currency symbols (e.g. $240,000, ISO-27001)."""
    words = re.findall(r"[a-zA-Z0-9_\$]+", str(text or "").lower())
    return [w for w in words if len(w) > 1 and w not in STOP_WORDS]


def stem(word: str) -> str:
    """Lightweight English suffix normaliser for common morphological variants."""
    for suffix in ("tions", "tion", "ments", "ment", "ingly", "ings", "ing", "ies", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) > len(suffix) + 2:
            return word[:-len(suffix)]
    return word


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def document_name_for(filename: str, supplier_name: Optional[str] = None) -> str:
    if supplier_name:
        return supplier_name
    return os.path.splitext(os.path.basename(filename))[0].replace("_", " ").title()


def chunks_from_pages(
    pages: Iterable[Dict[str, Any]],
    filename: str,
    supplier_name: Optional[str] = None,
    window_lines: int = WINDOW_LINES,
    step_lines: int = STEP_LINES,
) -> List[Dict[str, Any]]:
    """
    Build overlapping, line-indexed chunks from already-extracted pages.

    Line numbers are taken from each page's ``raw_text`` when present so they
    stay faithful to the PDF: stripping a page first would silently shift every
    line number by however many blank lines preceded the content.

    Each chunk records its document, 1-based page, 1-based start/end line, the
    joined text and the individual numbered lines, which is what lets the UI
    cite "Page 2, Lines 14–16" instead of just naming a file.
    """
    doc_name = document_name_for(filename, supplier_name)
    chunks: List[Dict[str, Any]] = []

    for page in pages or []:
        page_number = int(page.get("page", 0) or 0)
        source = page.get("raw_text")
        if source is None:
            source = page.get("text", "")
        raw_lines = str(source or "").splitlines()

        indexed = [(i + 1, line.strip()) for i, line in enumerate(raw_lines) if line.strip()]
        if not indexed:
            continue

        total = len(indexed)
        for start in range(0, total, step_lines):
            window = indexed[start:start + window_lines]
            if not window:
                continue
            chunks.append({
                "chunk_id": f"{doc_name}_p{page_number}_l{window[0][0]}",
                "document_name": doc_name,
                "filename": os.path.basename(filename),
                "page": page_number,
                "start_line": window[0][0],
                "end_line": window[-1][0],
                "text": " ".join(text for _, text in window),
                "lines": [{"line_num": num, "text": text} for num, text in window],
            })
            if start + window_lines >= total:
                break

    return chunks


def extract_document_chunks(
    pdf_bytes: bytes,
    filename: str,
    supplier_name: Optional[str] = None,
    window_lines: int = WINDOW_LINES,
    step_lines: int = STEP_LINES,
) -> List[Dict[str, Any]]:
    """Read a PDF and return line-indexed chunks. Convenience wrapper over `chunks_from_pages`."""
    pages: List[Dict[str, Any]] = []
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as doc:
        for index, page in enumerate(doc, 1):
            raw_text = page.get_text("text")
            pages.append({"page": index, "raw_text": raw_text, "text": raw_text.strip()})
    return chunks_from_pages(pages, filename, supplier_name, window_lines, step_lines)


# ---------------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------------


class BM25Index:
    """
    An Okapi BM25 index over a chunk collection.

    The collection statistics (document frequencies, IDF, average length) are
    computed once at construction, so ranking N queries against the same
    proposals costs one pass per query instead of one rebuild per query. That
    matters for criteria-targeted condensation, which issues one query per
    active criterion.
    """

    def __init__(self, chunks: Sequence[Dict[str, Any]]):
        self.chunks: List[Dict[str, Any]] = list(chunks)
        self.chunk_stems: List[List[str]] = []
        self.chunk_tf: List[Counter] = []
        self.chunk_lens: List[int] = []
        self.lowered: List[str] = []

        for chunk in self.chunks:
            tokens = tokenize(chunk.get("text", ""))
            stems = [stem(token) for token in tokens]
            self.chunk_stems.append(stems)
            self.chunk_tf.append(Counter(stems))
            self.chunk_lens.append(len(tokens))
            self.lowered.append(str(chunk.get("text", "")).lower())

        self.count = len(self.chunks)
        self.avg_len = max(1.0, sum(self.chunk_lens) / max(1, self.count))

        document_frequency: Counter = Counter()
        for stems in self.chunk_stems:
            for unique_stem in set(stems):
                document_frequency[unique_stem] += 1

        self.idf: Dict[str, float] = {
            term: math.log((self.count - freq + 0.5) / (freq + 0.5) + 1.0)
            for term, freq in document_frequency.items()
        }

    def __len__(self) -> int:
        return self.count

    def score(self, query: str) -> List[Tuple[int, float, set]]:
        """
        Score every chunk against a query.

        Returns ``(index, score, matched_terms)`` triples in descending score
        order. On top of textbook BM25 the score adds a verbatim-phrase bonus, a
        bigram bonus and a query-coverage multiplier, which together stop a
        chunk that happens to repeat one rare term from outranking a chunk that
        actually addresses the whole query.
        """
        query_tokens = tokenize(query)
        if not query_tokens or not self.count:
            return []

        query_stems = [stem(token) for token in query_tokens]
        distinct_stems = max(1, len(set(query_stems)))
        clean_query = str(query or "").strip().lower()

        scored: List[Tuple[int, float, set]] = []
        for index in range(self.count):
            tf = self.chunk_tf[index]
            length = self.chunk_lens[index]
            norm = K1 * (1 - B + B * (length / self.avg_len))

            total = 0.0
            matched: set = set()
            for query_stem, query_token in zip(query_stems, query_tokens):
                frequency = tf.get(query_stem)
                if not frequency:
                    continue
                matched.add(query_token)
                total += self.idf.get(query_stem, 0.5) * (
                    (frequency * (K1 + 1)) / (frequency + norm)
                )

            text_lower = self.lowered[index]
            if len(clean_query) > 3 and clean_query in text_lower:
                total += 4.5
            else:
                for position in range(len(query_tokens) - 1):
                    bigram = f"{query_tokens[position]} {query_tokens[position + 1]}"
                    if bigram in text_lower:
                        total += 1.8

            coverage = len(matched) / distinct_stems
            total *= 0.6 + 0.4 * coverage
            scored.append((index, total, matched))

        scored.sort(key=lambda row: row[1], reverse=True)
        return scored

    def rank(self, query: str, top_k: int = 10, minimum: float = 0.05) -> List[Dict[str, Any]]:
        """Score a query and return chunk copies carrying a 0–100 relevance percentage."""
        scored = self.score(query)
        if not scored:
            return []
        best = scored[0][1] if scored[0][1] > 0 else 1.0

        ranked: List[Dict[str, Any]] = []
        for index, raw_score, matched in scored:
            if raw_score <= minimum:
                continue
            ranked.append({
                **self.chunks[index],
                "score": min(100.0, round((raw_score / best) * 98.0, 1)),
                "raw_score": raw_score,
                "matched_terms": sorted(matched),
            })
            if top_k is not None and len(ranked) >= top_k:
                break
        return ranked


def aggregate_document_scores(
    scored_chunks: Sequence[Dict[str, Any]],
    all_chunks: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Roll chunk scores up to a per-document ranking::

        corroboration = mean(2nd and 3rd best passages) / best passage   (0 if alone)
        composite     = best passage x (0.85 + 0.15 x corroboration)

    The peak term dominates deliberately: the single most relevant passage is
    what answers a query, so the document holding it should win even if a rival
    mentions the topic more often. The 15% corroboration term then separates
    documents with comparable peaks, favouring the one that backs the match up
    across several passages.

    Expressing corroboration as a *ratio* to the peak matters. The obvious
    formulation — a weighted blend of the peak and the mean of the top three —
    behaves backwards: a document with one matching passage has a top-three mean
    equal to its peak, scoring the maximum, and every additional supporting
    passage drags that mean down. Under the ratio form a document can only gain
    from corroboration, never lose, and because only the second and third
    passages count, length alone still buys nothing.
    """
    documents = sorted({chunk["document_name"] for chunk in all_chunks})
    per_document: Dict[str, List[float]] = {name: [] for name in documents}
    for chunk in scored_chunks:
        per_document.setdefault(chunk["document_name"], []).append(chunk["score"])

    rankings: List[Dict[str, Any]] = []
    for name, scores in per_document.items():
        if not scores:
            rankings.append({
                "document_name": name, "score": 0.0, "max_score": 0.0,
                "avg_top_score": 0.0, "support_score": 0.0, "match_count": 0,
            })
            continue

        ordered = sorted(scores, reverse=True)
        peak = ordered[0]
        support = ordered[1:3]
        support_mean = sum(support) / len(support) if support else 0.0
        corroboration = min(1.0, support_mean / peak) if peak > 0 else 0.0

        rankings.append({
            "document_name": name,
            "score": round(min(100.0, peak * (0.85 + 0.15 * corroboration)), 1),
            "max_score": peak,
            "avg_top_score": round(sum(ordered[:3]) / min(3, len(ordered)), 1),
            "support_score": round(support_mean, 1),
            "match_count": len(scores),
        })

    rankings.sort(key=lambda row: (row["score"], row["match_count"]), reverse=True)
    top = rankings[0] if rankings and rankings[0]["score"] > 0 else None
    return rankings, top


def search_without_llm(
    query: str,
    chunks: Sequence[Dict[str, Any]],
    top_k: int = 10,
) -> Dict[str, Any]:
    """Offline BM25 search across chunks, with per-document aggregation."""
    if not str(query or "").strip() or not chunks:
        return {
            "query": query, "mode": "without_llm", "top_document": None,
            "top_document_score": 0.0, "document_rankings": [], "results": [],
        }

    index = BM25Index(chunks)
    ranked = index.rank(query, top_k=None)
    rankings, top = aggregate_document_scores(ranked, chunks)

    return {
        "query": query,
        "mode": "without_llm",
        "top_document": top["document_name"] if top else None,
        "top_document_score": top["score"] if top else 0.0,
        "top_document_summary": (
            f"{top['document_name']} matched {top['match_count']} relevant chunks with a "
            f"composite score of {top['score']}%."
            if top else "No matching document found."
        ),
        "document_rankings": rankings,
        "results": ranked[:top_k],
    }


# ---------------------------------------------------------------------------
# Criteria-targeted selection
# ---------------------------------------------------------------------------


def select_relevant_chunks(
    chunks: Sequence[Dict[str, Any]],
    queries: Sequence[str],
    budget_chars: int,
    per_query: int = 12,
    overhead_per_chunk: int = 0,
) -> List[Dict[str, Any]]:
    """
    Pick the passages most worth sending to a model under a character budget.

    One BM25 query is issued per criterion against a single shared index, the
    per-criterion winners are interleaved by rank so that every criterion is
    represented before any criterion gets a second passage, and the survivors
    are finally re-sorted into reading order. Interleaving is what prevents a
    verbose criterion from consuming the whole budget and leaving another
    criterion with no evidence at all.

    ``overhead_per_chunk`` lets the caller charge each selected passage for the
    citation markers it will later be wrapped in, so the budget reflects the
    string that actually gets sent rather than the passage text alone.
    """
    if not chunks or budget_chars <= 0:
        return []

    index = BM25Index(chunks)
    by_id: Dict[str, Dict[str, Any]] = {}
    ordered_hits: List[List[str]] = []

    for query in queries:
        hits = index.rank(query, top_k=per_query)
        ordered_hits.append([hit["chunk_id"] for hit in hits])
        for hit in hits:
            existing = by_id.get(hit["chunk_id"])
            if existing is None or hit["raw_score"] > existing["raw_score"]:
                by_id[hit["chunk_id"]] = hit

    # Interleave: round 0 takes each criterion's best chunk, round 1 the next, ...
    selected: List[Dict[str, Any]] = []
    used_chars = 0
    taken: set = set()
    for round_index in range(per_query):
        for hits in ordered_hits:
            if round_index >= len(hits):
                continue
            chunk_id = hits[round_index]
            if chunk_id in taken:
                continue
            candidate = by_id[chunk_id]
            size = len(candidate["text"]) + 1 + overhead_per_chunk
            if used_chars + size > budget_chars:
                continue
            taken.add(chunk_id)
            selected.append(candidate)
            used_chars += size

    # Nothing fitted (a single chunk larger than the budget): keep the best one.
    if not selected and by_id:
        best = max(by_id.values(), key=lambda hit: hit["raw_score"])
        selected = [best]

    selected.sort(key=lambda chunk: (chunk["page"], chunk["start_line"]))
    return selected
