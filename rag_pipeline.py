"""
Core RAG pipeline shared by build_index.py and streamlit_app.py.

Two pipelines, matching the experiments in news_rag_pipeline.ipynb:

  v1 (baseline) : sentence-level chunks  -> dense vector search (Qdrant)
  v2 (improved) : 2-3 sentence chunks    -> hybrid search (dense + BM25, fused with RRF)
                                         -> cross-encoder reranking

Both feed the same grounded prompt to a local LLM served by Ollama, and the
answer is checked by three citation guardrails.
"""

import json
import os
import re
from pathlib import Path

import numpy as np
import requests

# ── Configuration ──────────────────────────────────────────────────────────────
ARTICLES_PATH  = Path(__file__).parent / "articles.jsonl"
COLLECTION_V1  = "news_chunks"      # sentence chunks
COLLECTION_V2  = "news_chunks_v2"   # 2-3 sentence chunks

EMBED_MODEL_ID = "microsoft/harrier-oss-v1-0.6b"
EMBED_DIM      = 1024
RERANKER_ID    = "cross-encoder/ms-marco-MiniLM-L-6-v2"

OLLAMA_URL     = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
DEFAULT_MODEL  = "gemma4:latest"
OLLAMA_TIMEOUT = int(os.getenv("OLLAMA_TIMEOUT", "300"))   # seconds; large models on CPU are slow

QDRANT_URL     = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_PATH    = os.getenv("QDRANT_PATH")  # set to use embedded Qdrant (no Docker)

MIN_WORDS      = 8    # sentences shorter than this are dropped (bylines, captions, ...)
PAYLOAD_KEYS   = ["chunk_id", "article_id", "title", "source", "topic", "url", "text"]


# ── Data loading ───────────────────────────────────────────────────────────────
def load_articles(path: Path = ARTICLES_PATH) -> list[dict]:
    articles = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                articles.append(json.loads(line))
    return articles


def get_qdrant_client():
    from qdrant_client import QdrantClient
    if QDRANT_PATH:
        return QdrantClient(path=QDRANT_PATH)
    return QdrantClient(url=QDRANT_URL)


def load_embed_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL_ID, trust_remote_code=True)


# ── Chunking ───────────────────────────────────────────────────────────────────
def _sentences(body: str) -> list[str]:
    import nltk
    nltk.download("punkt_tab", quiet=True)
    return nltk.tokenize.sent_tokenize(body)


def _chunk(article: dict, idx: int, text: str) -> dict:
    return {
        "chunk_id":   f"{article['article_id']}_{idx}",
        "article_id": int(article["article_id"]),
        "title":      article["title"],
        "source":     article["source"],
        "topic":      article["topic"],
        "url":        article["url"],
        "text":       text,
    }


def chunk_article(article: dict) -> list[dict]:
    """v1: one chunk per sentence (the chunk index is the sentence index)."""
    return [
        _chunk(article, i, sent)
        for i, sent in enumerate(_sentences(article["body"]))
        if len(sent.split()) >= MIN_WORDS
    ]


def chunk_article_v2(article: dict, short_threshold: int = 40) -> list[dict]:
    """v2: group sentences into 2-sentence chunks.
    If the combined word count is still below short_threshold, extend to 3 sentences.
    This gives richer context per chunk without losing granularity for long sentences.
    """
    sentences = [s for s in _sentences(article["body"]) if len(s.split()) >= MIN_WORDS]
    chunks, idx, i = [], 0, 0
    while i < len(sentences):
        group = [sentences[i]]
        i += 1
        if i < len(sentences):                      # always add a 2nd sentence
            group.append(sentences[i]); i += 1
        if i < len(sentences) and len(" ".join(group).split()) < short_threshold:
            group.append(sentences[i]); i += 1     # extend to 3rd if still short
        chunks.append(_chunk(article, idx, " ".join(group)))
        idx += 1
    return chunks


# ── Retrieval ──────────────────────────────────────────────────────────────────
def _format_hits(scored_payloads) -> list[dict]:
    return [
        {"rank": rank + 1, "score": round(float(score), 4), **{k: p[k] for k in PAYLOAD_KEYS}}
        for rank, (score, p) in enumerate(scored_payloads)
    ]


def retrieve(query: str, model, client, k: int = 4, collection: str = COLLECTION_V1) -> list[dict]:
    """v1: return the top-k chunks by cosine similarity."""
    query_vec = model.encode(query, prompt_name="query", normalize_embeddings=True).tolist()
    response = client.query_points(collection_name=collection, query=query_vec, limit=k)
    return _format_hits((hit.score, hit.payload) for hit in response.points)


class HybridRetriever:
    """v2: dense vector search + BM25, merged with Reciprocal Rank Fusion, then reranked
    with a cross-encoder.

    The BM25 index is built from the chunk texts stored in Qdrant, so it always
    matches the vector collection (BM25 position i == Qdrant point id i).
    """

    K_RRF = 60

    def __init__(self, model, client, collection: str = COLLECTION_V2):
        from rank_bm25 import BM25Okapi
        from sentence_transformers import CrossEncoder

        self.model, self.client, self.collection = model, client, collection
        self.payloads = self._load_payloads()
        self.bm25 = BM25Okapi([p["text"].lower().split() for p in self.payloads])
        self.reranker = CrossEncoder(RERANKER_ID)

    def _load_payloads(self) -> list[dict]:
        points, offset = [], None
        while True:
            batch, offset = self.client.scroll(
                self.collection, limit=1000, offset=offset, with_payload=True, with_vectors=False
            )
            points.extend(batch)
            if offset is None:
                break
        points.sort(key=lambda p: p.id)
        return [p.payload for p in points]

    def retrieve(self, query: str, k: int = 4, fetch_k: int = 20) -> list[dict]:
        """
        1. Retrieve fetch_k candidates from Qdrant (dense vector search).
        2. Retrieve fetch_k candidates from BM25 (sparse keyword search).
        3. Merge with Reciprocal Rank Fusion (RRF, k_rrf=60).
        4. Rerank the top-fetch_k fused candidates with a cross-encoder.
        5. Return the top-k reranked results.
        """
        # 1. Dense vector search
        qvec  = self.model.encode(query, prompt_name="query", normalize_embeddings=True).tolist()
        vresp = self.client.query_points(collection_name=self.collection, query=qvec, limit=fetch_k)
        vec_rank = {hit.id: rank for rank, hit in enumerate(vresp.points)}

        # 2. Sparse BM25 search (scores over the full corpus; take top fetch_k)
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_top    = np.argsort(bm25_scores)[::-1][:fetch_k]
        bm25_rank   = {int(idx): rank for rank, idx in enumerate(bm25_top)}

        # 3. RRF fusion: score = 1/(k+rank_vec) + 1/(k+rank_bm25)
        all_ids = set(vec_rank) | set(bm25_rank)
        rrf = {
            cid: 1 / (self.K_RRF + vec_rank.get(cid, fetch_k)) + 1 / (self.K_RRF + bm25_rank.get(cid, fetch_k))
            for cid in all_ids
        }
        top_ids    = sorted(rrf, key=rrf.__getitem__, reverse=True)[:fetch_k]
        candidates = [self.payloads[int(pt_id)] for pt_id in top_ids]

        # 4. Cross-encoder reranking
        ce_scores = self.reranker.predict([(query, c["text"]) for c in candidates])
        reranked  = sorted(zip(ce_scores, candidates), reverse=True, key=lambda x: x[0])

        return _format_hits(reranked[:k])


# ── Generation ─────────────────────────────────────────────────────────────────
RAG_SYSTEM_PROMPT = """
You are a retrieval-augmented question-answering assistant.

Your task is to answer the user’s question using ONLY the information contained in the provided passages.

Instructions:
1. Base your answer strictly on the provided passages.
2. Do not use outside knowledge, assumptions, or unstated context.
3. If the passages do not contain enough information to answer the question, say so clearly.
4. Do not invent facts, sources, citations, dates, numbers, or explanations.
5. When making a claim, cite the relevant passage ID and source article title inline.
6. Use citations in this format:
   [Passage ID: <id>, Source: "<article title>"]
7. If multiple passages support the same claim, cite all relevant passages.
8. If passages conflict, explain the conflict and cite the passages on each side.
9. Keep the answer concise, accurate, and directly responsive to the user’s question.
10. If the user asks for information not covered by the passages, state:
   “The provided passages do not contain enough information to answer that.”

Input format:

Question:
{{user_question}}

Passages:
{{retrieved_passages}}

Each passage will include:
- Passage ID
- Source article title
- Passage text

Answer format:
- Direct answer first.
- Supporting details with inline citations.
- Missing information, if any.

Now answer the question using only the provided passages.
"""


def build_rag_prompt(query: str, chunks: list[dict]) -> str:
    """Assemble the user-turn message from the query and retrieved chunks."""
    passages = "\n\n".join(
        f"[Passage {c['rank']} | chunk_id={c['chunk_id']} | {c['title']!r}]\n{c['text']}"
        for c in chunks
    )
    return f"Passages:\n{passages}\n\nQuestion: {query}"


def generate(prompt: str, model: str = DEFAULT_MODEL, system: str = "") -> str:
    """Chat completion against a local Ollama server; the LLM is a parameter."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    resp = requests.post(
        OLLAMA_URL,
        json={"model": model, "messages": messages, "stream": False},
        timeout=OLLAMA_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"]


# ── Guardrails ─────────────────────────────────────────────────────────────────
CHUNK_ID_RE = re.compile(r"\b\d+_\d+\b")   # chunk_ids look like "123_45" (article_sentence)


def guardrail_1(answer: str, chunks: list[dict]) -> tuple[bool, str]:
    """Citation presence: the answer must reference at least one retrieved chunk_id."""
    chunk_ids = {c["chunk_id"] for c in chunks}
    if not set(CHUNK_ID_RE.findall(answer)) & chunk_ids:
        return False, "Answer contains no chunk_id citations from the retrieved passages."
    return True, ""


def guardrail_2(answer: str, chunks: list[dict]) -> tuple[bool, str]:
    """Citation validity: every chunk_id cited in the answer must exist in the retrieved set."""
    valid_ids = {c["chunk_id"] for c in chunks}
    phantom   = set(CHUNK_ID_RE.findall(answer)) - valid_ids
    if phantom:
        return False, f"Answer cites chunk_id(s) not in retrieved set: {phantom}"
    return True, ""


def guardrail_3(answer: str, chunks: list[dict]) -> tuple[bool, str]:
    """Unsupported answer: must not claim info is missing AND give a confident answer."""
    lower = answer.lower()
    missing_signals = [
        "not mentioned", "not provided", "not available", "no information",
        "cannot find", "cannot answer", "not in the passages", "not found",
        "insufficient information", "the passages do not", "passages don't",
        "i don't have", "not discussed",
    ]
    confident_signals = [
        "according to", "the answer is", "states that", "reports that",
        "confirms that", "reveals that", "indicates that", "it is clear",
        "the article", "the passage",
    ]
    has_missing   = any(p in lower for p in missing_signals)
    has_confident = any(p in lower for p in confident_signals)
    if has_missing and has_confident:
        return False, (
            "Answer simultaneously claims information is missing and provides "
            "a confident factual response -- contradictory."
        )
    return True, ""


GUARDRAILS = [guardrail_1, guardrail_2, guardrail_3]


def run_guardrails(answer: str, chunks: list[dict]) -> tuple[bool, list[str]]:
    """Run all guardrails; return (all_passed, list of failure reasons)."""
    failures = []
    for fn in GUARDRAILS:
        passed, reason = fn(answer, chunks)
        if not passed:
            failures.append(f"{fn.__name__}: {reason}")
    return len(failures) == 0, failures


# ── Full RAG flow ──────────────────────────────────────────────────────────────
def rag_answer(query: str, chunks: list[dict], model: str = DEFAULT_MODEL) -> dict:
    """Generate a grounded answer from already-retrieved chunks and run the guardrails."""
    answer = generate(build_rag_prompt(query, chunks), model=model, system=RAG_SYSTEM_PROMPT)
    passed, failures = run_guardrails(answer, chunks)
    return {
        "query":              query,
        "answer":             answer,
        "chunks":             chunks,
        "guardrails_passed":  passed,
        "guardrail_failures": failures,
    }
