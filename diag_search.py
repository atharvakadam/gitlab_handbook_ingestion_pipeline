#!/usr/bin/env python3
"""
Diagnoses Atlas Search configuration and runs a tiny KNN probe.

Connects to MongoDB, lists Atlas Search indexes, shows a sample document,
and runs a KNN probe to verify search configuration. Useful for debugging
Atlas Search setup and ensuring proper indexing.
"""
import argparse, os, json, sys
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mongo-uri", default=os.getenv("MONGO_URI"), help="Atlas URI or env MONGO_URI")
    ap.add_argument("--db", default="gitlab_internal_documentation")
    ap.add_argument("--collection", default="handbook")
    ap.add_argument("--index", default="handbook_knn")
    ap.add_argument("--model", default="intfloat/e5-base-v2")
    args = ap.parse_args()

    if not args.mongo_uri:
        print("✗ No --mongo-uri or MONGO_URI set", file=sys.stderr)
        sys.exit(1)

    print(f"[info] connecting… db={args.db} coll={args.collection}")
    client = MongoClient(args.mongo_uri)
    coll = client[args.db][args.collection]

    print("[info] counting docs…")
    print("  estimatedDocumentCount:", coll.estimated_document_count())

    print("\n[info] listing Atlas Search indexes…")
    try:
        idx = list(coll.aggregate([{"$listSearchIndexes": {}}]))
        if not idx:
            print("  (none found)")
        for i in idx:
            print("  - name:", i.get("name"),
                  "status:", i.get("status"),
                  "queryable:", i.get("queryable"),
                  "type:", i.get("type"))
    except Exception as e:
        print("  $listSearchIndexes failed:", e)

    # Show one sample doc (sanity)
    print("\n[info] sample doc fields:")
    doc = coll.find_one({}, {"_id":0, "doc_key":1, "doc_id":1, "chunk_index":1, "embedding": {"$slice": 3}})
    print("  ", doc)

    # KNN probe
    print("\n[info] running tiny KNN probe…")
    model = SentenceTransformer(args.model)
    q = "policy benefits parental leave"
    vec = model.encode(f"query: {q}", normalize_embeddings=True).tolist()

    pipeline = [
        {"$search": {
            "index": args.index,
            "knnBeta": {
                "path": "embedding",
                "vector": vec,
                "k": 5,
                "numCandidates": 200
            }
        }},
        {"$set": {"score": {"$meta": "searchScore"}}},
        {"$limit": 3},
        {"$project": {"_id":0, "doc_id":1, "chunk_index":1, "web_url":1, "score":1}}
    ]
    try:
        res = list(coll.aggregate(pipeline))
        print("  hits:", len(res))
        for r in res:
            print("   ", round(r["score"],4), r["doc_id"], r["chunk_index"], r["web_url"])
    except Exception as e:
        print("  aggregate error:", e)

if __name__ == "__main__":
    main()
