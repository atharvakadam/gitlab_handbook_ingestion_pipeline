"""
QA tool for GitLab Handbook data in MongoDB. Validates and analyzes ingested data.

Key checks:
1. Document counts and duplicate detection
2. Required field presence (web_url, embedding, chunk_text)
3. Embedding dimension consistency
4. Random document sampling with optional URL validation
5. Token count statistics (avg, p95)
6. Collection storage metrics

Run with --help for usage options.
"""

# !/usr/bin/env python3
import argparse
import json
import sys
from pprint import pprint

from pymongo import MongoClient
from statistics import mean
import os
from dotenv import load_dotenv

load_dotenv()

def coll_stats(db, coll_name):
    return db.command("collStats", coll_name, scale=1)

def main():
    ap = argparse.ArgumentParser(description="QA checks for GitLab Handbook ingestion.")
    # ap.add_argument("--mongo-uri", required=True)
    ap.add_argument("--db", default="gitlab_internal_documentation")
    ap.add_argument("--collection", default="handbook")
    ap.add_argument("--sample", type=int, default=3, help="sample size for preview")
    ap.add_argument("--check-urls", action="store_true", help="perform HTTP 200 checks for sampled web_url (requires requests)")
    args = ap.parse_args()

    client = MongoClient(os.getenv("MONGO_URI"))
    db = client[args.db]
    coll = db[args.collection]

    print("=== A) Counts & coverage ===")
    total = coll.estimated_document_count()
    print("total_docs:", total)
    distinct_doc_ids = coll.distinct("doc_id")
    print("distinct doc_id:", len(distinct_doc_ids))

    print("\n=== B) Field completeness ===")
    missing_web = coll.count_documents({"web_url": {"$exists": False}})
    missing_emb = coll.count_documents({"embedding": {"$exists": False}})
    missing_chunk = coll.count_documents({"chunk_text": {"$exists": False}})
    empty_chunk = coll.count_documents({"$or": [{"chunk_text": {"$size": 0}}, {"chunk_text": ""}]})

    print("missing web_url:", missing_web)
    print("missing embedding:", missing_emb)
    print("missing chunk_text:", missing_chunk)
    print("empty chunk_text:", empty_chunk)

    print("\n=== C) Embedding dimensionality distribution ===")
    pipeline = [
        {"$project": {"n": {"$size": "$embedding"}}},
        {"$group": {"_id": "$n", "c": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]
    dims = list(coll.aggregate(pipeline))
    pprint(dims)

    print("\n=== D) Sample documents ===")
    sample_pipe = [
        {"$sample": {"size": args.sample}},
        {"$project": {"_id":0, "doc_key":1, "doc_id":1, "chunk_index":1, "title":1, "section":1, "web_url":1, "snippet":{"$substrCP":["$chunk_text",0,200]}}}
    ]
    sample_docs = list(coll.aggregate(sample_pipe))
    for i,d in enumerate(sample_docs,1):
        print(f"[{i}] {d['title']}  ({d['doc_id']}#{d['chunk_index']})")
        print("   web_url:", d["web_url"])
        print("   snippet:", d["snippet"].replace("\n"," ")[:200])
    if args.check_urls:
        try:
            import requests
            print("\n=== F) URL 200 checks ===")
            for d in sample_docs:
                u = d["web_url"]
                try:
                    r = requests.get(u, timeout=10)
                    print(u, "->", r.status_code)
                except Exception as e:
                    print(u, "-> ERROR:", e)
        except ImportError:
            print("[skip] requests not installed; run `pip install requests` and re-run with --check-urls")

    print("\n=== E) Token stats (avg, p95) ===")
    try:
        p = list(coll.aggregate([
            {"$group": {"_id": None, "avgTokens": {"$avg": "$token_count"}, "p95": {"$percentile": {"p":[0.95], "input":"$token_count"}}}}
        ]))
        if p:
            g = p[0]
            print("avgTokens:", round(g.get("avgTokens",0),1), "p95:", g.get("p95",[None])[0])
        else:
            print("[warn] percentile aggregation returned no result")
    except Exception as e:
        vals = [d.get("token_count",0) for d in coll.find({}, {"token_count":1})]
        if vals:
            vals_sorted = sorted(vals)
            avg = sum(vals_sorted)/len(vals_sorted)
            idx = int(0.95*len(vals_sorted))-1
            p95v = vals_sorted[max(idx,0)]
            print("avgTokens:", round(avg,1), "p95:", p95v, "(client-side)")
        else:
            print("[warn] No token_count values found")

    print("\n=== G) collStats (storage overview) ===")
    cs = coll_stats(db, args.collection)
    fields = ["count","size","storageSize","totalIndexSize","nindexes","avgObjSize"]
    for f in fields:
        print(f"{f}: {cs.get(f)}")

if __name__ == "__main__":
    main()
