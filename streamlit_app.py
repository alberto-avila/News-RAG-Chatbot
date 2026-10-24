"""
News RAG Chatbot — Streamlit UI
Run: streamlit run streamlit_app.py
Requires: Qdrant with the 'news_chunks' / 'news_chunks_v2' collections (python build_index.py)
          and an Ollama server with at least one of AVAILABLE_MODELS pulled.
Sources: BBC News, The Guardian, NPR News, Ars Technica
"""

import streamlit as st

from rag_pipeline import (
    DEFAULT_MODEL, HybridRetriever, get_qdrant_client, load_articles, load_embed_model,
    rag_answer, retrieve,
)

AVAILABLE_MODELS = [
    DEFAULT_MODEL,
    "qwen3:latest",
    "phi4-mini:latest",
    "granite4.1:3b",
    "gpt-oss:20b",
    "qwen2.5:0.5B",
]
PIPELINES = {
    "Hybrid (vector + BM25 + reranker)": "v2",
    "Vector only (baseline)":            "v1",
}


# ── Cached resources ───────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model...")
def cached_embed_model():
    return load_embed_model()


@st.cache_resource(show_spinner="Connecting to Qdrant...")
def cached_qdrant_client():
    return get_qdrant_client()


@st.cache_resource(show_spinner="Building BM25 index and loading reranker...")
def cached_hybrid_retriever():
    return HybridRetriever(cached_embed_model(), cached_qdrant_client())


@st.cache_data(show_spinner="Loading articles...")
def cached_articles() -> list[dict]:
    return load_articles()


def answer_query(query: str, k: int, model: str, pipeline: str) -> dict:
    if pipeline == "v2":
        chunks = cached_hybrid_retriever().retrieve(query, k=k)
    else:
        chunks = retrieve(query, cached_embed_model(), cached_qdrant_client(), k=k)
    return rag_answer(query, chunks, model=model)


def render_details(msg: dict):
    """Retrieved passages + guardrail status shown under each assistant answer."""
    with st.expander(f"Retrieved passages ({len(msg['chunks'])})", expanded=False):
        for c in msg["chunks"]:
            st.markdown(
                f"**[{c['rank']}]** `chunk_id={c['chunk_id']}` "
                f"— score `{c['score']}` "
                f"— *{c['source']}* · {c['topic']}\n\n"
                f"**[{c['title']}]({c['url']})**\n\n"
                f"> {c['text'][:250]}"
            )
            st.divider()

    if msg["guardrail_failures"]:
        st.warning(
            "**Guardrail failures**\n\n"
            + "\n\n".join(f"- {f}" for f in msg["guardrail_failures"])
        )
    else:
        st.success("All guardrails passed")


# ── Streamlit UI ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="News RAG Chatbot",
    page_icon="📰",
    layout="wide",
)

# Sidebar ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📰 News RAG")
    st.divider()

    selected_model = st.selectbox("LLM Model", AVAILABLE_MODELS, index=0)
    pipeline_label = st.radio("Retrieval pipeline", list(PIPELINES), index=0)
    pipeline       = PIPELINES[pipeline_label]
    top_k = st.slider("Retrieved passages (k)", min_value=2, max_value=8, value=4)

    st.divider()
    st.subheader("Article Browser")
    articles = cached_articles()

    topics          = sorted({a.get("topic", "general") for a in articles})
    selected_topic  = st.selectbox("Filter by topic", ["All"] + topics)
    filtered        = (
        articles if selected_topic == "All"
        else [a for a in articles if a.get("topic") == selected_topic]
    )
    st.caption(f"{len(filtered)} of {len(articles)} articles")

    for a in filtered[:12]:
        with st.expander(a["title"][:60], expanded=False):
            st.caption(f"{a['source']} · {a.get('topic','?')} · {a.get('pub_date','')[:10]}")
            st.write(a.get("description", "")[:200])
            st.markdown(f"[Read article]({a['url']})")

    if len(filtered) > 12:
        st.caption(f"...and {len(filtered) - 12} more")

    st.divider()
    if st.button("Clear chat history"):
        st.session_state.messages = []
        st.rerun()

# Main chat area ───────────────────────────────────────────────────────────────
st.title("News Chatbot")
st.caption(
    f"Corpus: **{len(articles)} articles** · "
    f"Model: **{selected_model}** · "
    f"Retrieval: **{pipeline_label}** · "
    f"k = **{top_k}**"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing conversation
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_details(msg)

# Chat input
if prompt := st.chat_input("Ask about recent news..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving passages and generating answer..."):
            result = answer_query(prompt, k=top_k, model=selected_model, pipeline=pipeline)

        msg = {
            "role":               "assistant",
            "content":            result["answer"],
            "chunks":             result["chunks"],
            "guardrails_passed":  result["guardrails_passed"],
            "guardrail_failures": result["guardrail_failures"],
        }
        st.markdown(msg["content"])
        render_details(msg)

    st.session_state.messages.append(msg)
