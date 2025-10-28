"""
The prefilter_cleanup.py script is a data cleaning and normalization tool in your GitLab handbook ingestion pipeline. 
Here's a concise overview of its key functions:

1. Section Normalization (--normalize-section):
- Removes /index.md or /_index.md from document section paths
- Standardizes section references for consistent lookups

2. Hierarchical Prefixing (--add-prefixes):
- Adds section_prefix1: First path segment (e.g., "engineering" from "engineering/onboarding")
- Adds section_prefix2: First two path segments (e.g., "engineering/onboarding")
- Enables efficient hierarchical queries and filtering

3. Date Normalization (--backfill-dates):
- Converts updated_at string timestamps to proper MongoDB Date objects
- Stores them in a new updated_at_dt field for better querying and sorting

The script includes a --dry-run flag for previewing changes and connects to MongoDB using environment variables or command-line arguments. 
It's designed for one-time or occasional use to maintain data consistency in your handbook documentation system.
"""
#!/usr/bin/env python3
import argparse, os, sys
from pprint import pprint
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
def run_update(coll, pipeline, where=None, label="update"):
    q = where or {}
    try:
        res = coll.update_many(q, pipeline)
        print(f"[{label}] matched={res.matched_count} modified={res.modified_count}")
    except Exception as e:
        print(f"[{label}] error: {e}", file=sys.stderr)

def main():
    ap = argparse.ArgumentParser(description="One-off cleanup: normalize section, add section_prefix1/2, backfill updated_at_dt")
    ap.add_argument("--mongo-uri", default=os.getenv("MONGO_URI"), help="Atlas URI or env MONGO_URI")
    ap.add_argument("--db", default="gitlab_internal_documentation")
    ap.add_argument("--collection", default="handbook")
    ap.add_argument("--normalize-section", action="store_true", help="normalize 'section' ending with /index.md or /_index.md")
    ap.add_argument("--add-prefixes", action="store_true", help="add section_prefix1 / section_prefix2")
    ap.add_argument("--backfill-dates", action="store_true", help="add updated_at_dt from updated_at (string -> Date)")
    ap.add_argument("--dry-run", action="store_true", help="print planned actions without writing")
    args = ap.parse_args()

    if not args.mongo_uri:
        print("✗ No --mongo-uri and MONGO_URI not set", file=sys.stderr)
        sys.exit(1)

    cli = MongoClient(args.mongo_uri)
    coll = cli[args.db][args.collection]

    print("[info] collection:", f"{args.db}.{args.collection}")
    print("[info] docs:", coll.estimated_document_count())

    if args.normalize_section:
        print("\n[step] normalize section (strip /index.md & /_index.md)")
        pipeline = [
            {
                "$set": {
                    "section": {
                        "$replaceAll": {
                            "input": {
                                "$replaceAll": {"input": "$section", "find": "/index.md", "replacement": ""}
                            },
                            "find": "/_index.md",
                            "replacement": ""
                        }
                    }
                }
            }
        ]
        if not args.dry_run:
            run_update(coll, pipeline, {"section": {"$regex": r"/_?index\.md$"}}, "normalize_section")
        else:
            print("[dry-run] would run updateMany on documents where section matches /_?index\.md$")

    if args.add_prefixes:
        print("\n[step] add section_prefix1 / section_prefix2")
        pipeline = [
            {"$set": {
                "section_prefix1": {
                    "$let": {
                        "vars": {"parts": {"$split": ["$section", "/"]}},
                        "in": {"$arrayElemAt": ["$$parts", 0]}
                    }
                },
                "section_prefix2": {
                    "$let": {
                        "vars": {"parts": {"$split": ["$section", "/"]}},
                        "in": {
                            "$cond": [
                                {"$gte": [{"$size": "$$parts"}, 2]},
                                {"$concat": [
                                    {"$arrayElemAt": ["$$parts", 0]}, "/",
                                    {"$arrayElemAt": ["$$parts", 1]}
                                ]},
                                {"$arrayElemAt": ["$$parts", 0]}
                            ]
                        }
                    }
                }
            }}
        ]
        if not args.dry_run:
            run_update(coll, pipeline, None, "add_prefixes")
        else:
            print("[dry-run] would set section_prefix1/2 for all docs")

    if args.backfill_dates:
        print("\n[step] backfill updated_at_dt (Date) from updated_at (string)")
        pipeline = [
            {"$set": {
                "updated_at_dt": {
                    "$dateFromString": {"dateString": "$updated_at", "onError": None, "onNull": None}
                }
            }}
        ]
        if not args.dry_run:
            run_update(coll, pipeline, {"updated_at": {"$type": "string"}}, "backfill_dates")
        else:
            print("[dry-run] would set updated_at_dt where updated_at is a string")

    # Show top prefix counts for sanity
    print("\n[info] top section_prefix1 (post-update)")
    try:
        agg = coll.aggregate([
            {"$group": {"_id": "$section_prefix1", "n": {"$sum": 1}}},
            {"$sort": {"n": -1}},
            {"$limit": 20}
        ])
        print(list(agg))
    except Exception as e:
        print("[warn] could not aggregate section_prefix1:", e)

    print("\n[done] cleanup complete.")

if __name__ == "__main__":
    main()
