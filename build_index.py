"""
Chunk the scraped articles, embed every chunk and load it into Qdrant.

    python build_index.py              # build both collections (v1 + v2)
    python build_index.py --only v2    # only the improved 2-3 sentence collection

Each collection is dropped and recreated so re-runs start fresh.
"""
import argparse

import numpy as np
from qdrant_client.models import Distance, PointStruct, VectorParams
from tqdm.auto import tqdm

from rag_pipeline import (
    COLLECTION_V1, COLLECTION_V2, EMBED_DIM, PAYLOAD_KEYS,
    chunk_article, chunk_article_v2, get_qdrant_client, load_articles, load_embed_model,
)

BATCH_SIZE   = 64
UPSERT_BATCH = 256   # stay within Qdrant's request size limits

PIPELINES = {
    "v1": (COLLECTION_V1, chunk_article),
    "v2": (COLLECTION_V2, chunk_article_v2),
}


def build_collection(name, chunker, articles, model, client):
    chunks = [c for a in articles for c in chunker(a)]
    avg_words = np.mean([len(c["text"].split()) for c in chunks])
    print(f"\n[{name}] {len(chunks)} chunks, avg {avg_words:.0f} words/chunk")

    texts = [c["text"] for c in chunks]
    embeddings = np.vstack([
        model.encode(texts[i : i + BATCH_SIZE], normalize_embeddings=True, show_progress_bar=False)
        for i in tqdm(range(0, len(texts), BATCH_SIZE), desc=f"Encoding {name}")
    ])

    if client.collection_exists(name):
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
    )

    # Point id == position in the chunk list (the hybrid retriever relies on this for BM25)
    points = [
        PointStruct(id=i, vector=embeddings[i].tolist(), payload={k: c[k] for k in PAYLOAD_KEYS})
        for i, c in enumerate(chunks)
    ]
    for start in tqdm(range(0, len(points), UPSERT_BATCH), desc=f"Upserting {name}"):
        client.upsert(collection_name=name, points=points[start : start + UPSERT_BATCH])

    print(f"[{name}] {client.get_collection(name).points_count} vectors | dim={EMBED_DIM}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=list(PIPELINES), help="Build a single collection")
    args = parser.parse_args()

    articles = load_articles()
    print(f"Loaded {len(articles)} articles")
    model  = load_embed_model()
    client = get_qdrant_client()

    for key in [args.only] if args.only else PIPELINES:
        name, chunker = PIPELINES[key]
        build_collection(name, chunker, articles, model, client)
    client.close()


if __name__ == "__main__":
    main()
