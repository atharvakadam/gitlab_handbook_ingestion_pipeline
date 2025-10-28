"""
The qa_vector_search.py script is a testing tool for the GitLab handbook's vector search pipeline. It performs semantic search 
using MongoDB Atlas's vector search capabilities and includes features like optional cross-encoder re-ranking, context stitching 
for better coherence, and keyword overlap scoring. The script helps verify search quality before integration into the main agent, 
providing detailed output of search results, scores, and source information.
"""

#!/usr/bin/env python3
import argparse
import re
from typing import List, Dict, Any

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os

load_dotenv()
try:
    from sentence_transformers import CrossEncoder
    HAS_CE = True
except Exception:
    HAS_CE = False

def stitch_adjacent(results: List[Dict[str, Any]], max_per_doc=2) -> List[Dict[str, Any]]:
    by_doc = {}
    for r in results:
        by_doc.setdefault(r["doc_id"], []).append(r)
    stitched = []
    for doc_id, items in by_doc.items():
        items.sort(key=lambda x: x["chunk_index"])
        groups = []
        cur = [items[0]]
        for a, b in zip(items, items[1:]):
            if b["chunk_index"] == a["chunk_index"] + 1:
                cur.append(b)
            else:
                groups.append(cur); cur = [b]
        groups.append(cur)
        for g in groups[:max_per_doc]:
            text = "\n".join(x["chunk_text"] for x in g)
            stitched.append({
                "doc_id": doc_id,
                "start_index": g[0]["chunk_index"],
                "end_index": g[-1]["chunk_index"],
                "web_url": g[0]["web_url"],
                "title": g[0]["title"],
                "score": sum(x["score"] for x in g)/len(g),
                "context": text
            })
    return sorted(stitched, key=lambda x: x["score"], reverse=True)

def keyword_overlap_score(query: str, text: str) -> float:
    toks = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", query) if len(t) > 2]
    if not toks: return 0.0
    uniq = sorted(set(toks))
    present = sum(1 for t in uniq if re.search(rf"\b{re.escape(t)}\b", text.lower()))
    return round(present / len(uniq), 3)

def main():
    ap = argparse.ArgumentParser(description="Atlas Vector Search QA + relevance check")
    # ap.add_argument("--mongo-uri", required=True)
    ap.add_argument("--db", default="gitlab_internal_documentation")
    ap.add_argument("--collection", default="handbook")
    ap.add_argument("--index", default="handbook_knn", help="Atlas Search index name")
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    ap.add_argument("--query", required=True)
    ap.add_argument("--k", type=int, default=50, help="ANN hits to pull from Atlas")
    ap.add_argument("--top", type=int, default=6, help="How many to display after (re)ranking")
    ap.add_argument("--rerank", action="store_true", help="Apply cross-encoder re-ranking to top-k before stitching")
    args = ap.parse_args()

    client = MongoClient(os.getenv("MONGO_URI"))
    coll = client[args.db][args.collection]

    print(f"[info] loading embedder: {args.model}")
    bi = SentenceTransformer(args.model)
    qvec = bi.encode(f"query: {args.query}", normalize_embeddings=True).tolist()

    pipeline = [
        {"$vectorSearch": {
            "index": args.index,          # e.g., "vector_index"
            "path": "embedding",
            "queryVector": qvec,
            "numCandidates": max(args.k * 5, 200),  # recall knob
            "limit": args.k
        }},
        {"$project": {
            "_id": 0,
            "doc_id": 1,
            "chunk_index": 1,
            "title": 1,
            "web_url": 1,
            "chunk_text": 1,
            "score": { "$meta": "vectorSearchScore" }   # score emitted by $vectorSearch
        }}
    ]

    hits = list(coll.aggregate(pipeline))
    print(f"[info] Atlas returned {len(hits)} hits")

    if args.rerank:
        if not HAS_CE:
            print("[warn] CrossEncoder not available; install torch + sentence-transformers")
        else:
            ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
            pairs = [(args.query, h["chunk_text"]) for h in hits]
            scores = ce.predict(pairs).tolist()
            for s, h in zip(scores, hits):
                h["ce_score"] = float(s)
            hits.sort(key=lambda x: x.get("ce_score", 0), reverse=True)

    print("\n=== Top raw chunks ===")
    for i, h in enumerate(hits[:args.top], 1):
        ov = keyword_overlap_score(args.query, h["chunk_text"])
        line = f"[{i}] score={h['score']:.4f}"
        if args.rerank and 'ce_score' in h: line += f"  ce={h['ce_score']:.4f}"
        print(line, "|", h["doc_id"], "#", h["chunk_index"])
        print("    ", h["title"], "→", h["web_url"])
        print("     overlap:", ov, "|", (h["chunk_text"][:240].replace("\n"," ") + ("..." if len(h["chunk_text"])>240 else "")))

    stitched = stitch_adjacent(hits, max_per_doc=2)

    print("\n=== Stitched context blocks ===")
    for i, s in enumerate(stitched[:args.top], 1):
        ov = keyword_overlap_score(args.query, s["context"])
        print(f"[{i}] score~{s['score']:.4f} | overlap={ov} | {s['title']} [{s['doc_id']}:{s['start_index']}–{s['end_index']}]")
        print("    ", s["web_url"])
        preview = s["context"][:400].replace("\n"," ")
        print("     ", preview + ("..." if len(s['context'])>400 else ""))

if __name__ == "__main__":
    main()
