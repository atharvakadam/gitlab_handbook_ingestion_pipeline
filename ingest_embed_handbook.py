#!/usr/bin/env python3
"""
Ingests and embeds GitLab Handbook content into MongoDB for semantic search.

Processes markdown files from a local GitLab handbook repository, splitting them into
semantic chunks with configurable size/overlap. Generates vector embeddings using
Sentence Transformers (default: e5-base-v2) and stores them in MongoDB with rich metadata.

MongoDB Document Structure:
    doc_key (str): Unique identifier (format: "path/to/doc#chunk_index")
    doc_id (str): Source document path (without chunk index)
    chunk_index (int): Position of chunk in document
    title (str): Document title from filename
    section (str): Handbook section (e.g., "engineering/development")
    web_url (str): Public URL to the document
    repo_url (str): Git URL to the source file
    chunk_text (str): Text content (truncated to snippet_chars)
    token_count (int): Approximate token count
    embedding (List[float]): Vector embedding (768-d for e5-base-v2)
    embedding_model (str): Model used for embedding
    sha (str): Git commit SHA
    access_groups (List[str]): Access control (default: ["all"])
    updated_at (str): ISO timestamp of processing
    content_hash (str): SHA-256 hash of original chunk text
    source (str): Source identifier ("gitlab-handbook")
    breadcrumbs (List[str]): Document hierarchy (if available)
    tags (List[str]): Optional tags for categorization

Handles markdown elements like tables (flattened) and images (alt text preserved).
Supports resumable processing via content hashing and batch operations for efficiency.
Designed for use in CI/CD pipelines with the GitLab handbook repository.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List

from markdown_it import MarkdownIt
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient, UpdateOne, ASCENDING
from pymongo.errors import BulkWriteError
from tqdm import tqdm
from dotenv import load_dotenv
import os

load_dotenv()

BASE_WEB = "https://handbook.gitlab.com/handbook/"

def path_to_web_url(md_path: Path, hb_root: Path) -> str:
    rel = md_path.relative_to(hb_root).as_posix()
    rel = rel.replace("_index.md","").replace("index.md","")
    if rel.endswith(".md"):
        rel = rel[:-3]
    if not rel.endswith("/"):
        rel += "/"
    return BASE_WEB + rel

def md_to_text(md_str: str) -> str:
    """Render markdown to plain text; keep headings, flatten tables, add image alt text."""
    md = MarkdownIt()
    html = md.render(md_str)
    soup = BeautifulSoup(html, "html.parser")

    # images → include alt text
    for img in soup.find_all("img"):
        alt = img.get("alt")
        if alt:
            img.insert_after(soup.new_string(f" (image: {alt}) "))

    # tables → flatten
    for table in soup.find_all("table"):
        lines = []
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if headers:
            lines.append(" | ".join(headers))
        for tr in table.find_all("tr"):
            cells = [td.get_text(" ", strip=True) for td in tr.find_all("td")]
            if cells:
                lines.append(" | ".join(cells))
        table.replace_with(soup.new_string("\n".join(lines)))

    # final plain text
    return soup.get_text("\n", strip=True)

def split_into_chunks(text: str, max_tokens: int = 550, overlap_tokens: int = 80) -> List[str]:
    """A simple sentence-based splitter (token ~ word proxy)."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    chunks, cur, count = [], [], 0
    for s in sents:
        ln = len(s.split())
        if cur and count + ln > max_tokens:
            chunk = " ".join(cur).strip()
            if chunk:
                chunks.append(chunk)
            ov = " ".join(chunk.split()[-overlap_tokens:]) if overlap_tokens > 0 else ""
            cur = ([ov] if ov else []) + [s]
            count = (len(ov.split()) if ov else 0) + ln
        else:
            cur.append(s); count += ln
    if cur:
        final = " ".join(cur).strip()
        if final:
            chunks.append(final)
    return chunks

def ensure_unique_index(coll, field: str = "doc_key"):
    """Create unique index on doc_key for idempotency."""
    try:
        coll.create_index([(field, ASCENDING)], unique=True)
    except Exception:
        pass

def prefetch_existing_doc_keys(coll) -> set:
    keys = set()
    for d in coll.find({}, {"_id": 0, "doc_key": 1}):
        k = d.get("doc_key")
        if k:
            keys.add(k)
    return keys

def main():
    ap = argparse.ArgumentParser(description="Ingest + embed GitLab Handbook into MongoDB (resumable).")
    ap.add_argument("--repo-root", type=Path, required=True, help="Path to local clone of the handbook repo")
    # ap.add_argument("--mongo-uri", type=str, required=True, help="MongoDB Atlas connection URI")
    ap.add_argument("--db", type=str, default="rag")
    ap.add_argument("--collection", type=str, default="handbook_chunks")
    ap.add_argument("--commit-sha", type=str, required=True, help="Commit SHA used to build repo_url")
    ap.add_argument("--model", type=str, default="intfloat/e5-base-v2", help="Local embedding model (768-d recommended)")
    ap.add_argument("--chunk-size", type=int, default=550)
    ap.add_argument("--chunk-overlap", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=200)
    ap.add_argument("--snippet-chars", type=int, default=1000, help="Keep only first N chars of chunk_text to save space")
    ap.add_argument("--prefetch-existing", action="store_true", help="Prefetch existing doc_keys to skip re-embedding")
    ap.add_argument("--max-files", type=int, default=0, help="Debug: limit number of markdown files processed (0=all)")
    args = ap.parse_args()

    hb_root = args.repo_root / "content" / "handbook"
    if not hb_root.exists():
        print(f"[error] Could not find {hb_root}. Did you pass --repo-root correctly?", file=sys.stderr)
        sys.exit(1)

    mongo_uri = os.getenv("MONGO_URI")
    client = MongoClient(mongo_uri)
    coll = client[args.db][args.collection]
    ensure_unique_index(coll, "doc_key")

    print(f"[info] Loading embedding model: {args.model}")
    model = SentenceTransformer(args.model)

    existing_keys = set()
    if args.prefetch_existing:
        print("[info] Prefetching existing doc_keys...")
        t0 = time.time()
        existing_keys = prefetch_existing_doc_keys(coll)
        print(f"[info] Prefetched {len(existing_keys)} keys in {time.time()-t0:.1f}s")

    md_files = [p for p in hb_root.rglob("*.md") if not any(seg.startswith(".") for seg in p.parts)]
    md_files.sort()
    if args.max_files > 0:
        md_files = md_files[:args.max_files]

    total_files = 0
    total_chunks = 0
    embedded = 0
    skipped = 0
    ops = []

    now_iso = datetime.utcnow().isoformat() + "Z"
    print(f"[info] Processing {len(md_files)} markdown files")
    for p in tqdm(md_files, desc="Embedding"):
        total_files += 1
        repo_rel = p.relative_to(hb_root).as_posix()
        doc_id = repo_rel.replace(".md","/").replace("_index/","").replace("index/","")
        web_url = path_to_web_url(p, hb_root)
        section = "/".join(p.relative_to(hb_root).parts[:2]) if len(p.relative_to(hb_root).parts) >= 2 else ""
        title = p.stem.replace("_"," ").title()

        try:
            raw = p.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[warn] Failed to read {p}: {e}")
            continue

        text = md_to_text(raw)
        if not text or len(text.split()) < 10:
            continue

        chunks = split_into_chunks(text, max_tokens=args.chunk_size, overlap_tokens=args.chunk_overlap)
        total_chunks += len(chunks)

        # Batch-embed all chunks for this file for speed
        passages = [f"passage: {ch}" for ch in chunks]
        if args.prefetch_existing:
            # filter only those not present
            passages_to_embed = []
            embed_indices = []
            for i, ch in enumerate(chunks):
                doc_key = f"{doc_id}#{i}"
                if doc_key in existing_keys:
                    skipped += 1
                    continue
                passages_to_embed.append(f"passage: {ch}")
                embed_indices.append(i)
            if passages_to_embed:
                vecs = model.encode(passages_to_embed, normalize_embeddings=True)
            else:
                vecs = []
        else:
            vecs = model.encode(passages, normalize_embeddings=True)
            embed_indices = list(range(len(chunks)))

        # Build upserts
        for vec, i in zip(vecs, embed_indices):
            ch = chunks[i]
            doc_key = f"{doc_id}#{i}"
            content_hash = hashlib.sha256(ch.encode("utf-8")).hexdigest()
            token_count = len(ch.split())  # proxy
            doc = {
                "doc_key": doc_key,
                "doc_id": doc_id,
                "chunk_index": i,
                "title": title,
                "breadcrumbs": [],                 # optional; fill later if you add heading-aware splitter
                "section": section,
                "tags": [],                         # optional
                "web_url": web_url,
                "repo_url": f"https://gitlab.com/gitlab-com/content-sites/handbook/-/blob/{args.commit_sha}/content/handbook/{repo_rel}",
                "chunk_text": ch[:args.snippet_chars],
                "token_count": token_count,
                "embedding": vec.astype("float32").tolist(),
                "embedding_model": args.model,
                "sha": args.commit_sha,
                "access_groups": ["all"],
                "updated_at": now_iso,
                "content_hash": f"sha256:{content_hash}",
                "source": "gitlab-handbook"
            }
            ops.append(UpdateOne({"doc_key": doc_key}, {"$setOnInsert": doc}, upsert=True))
            embedded += 1

            if len(ops) >= args.batch_size:
                try:
                    coll.bulk_write(ops, ordered=False)
                except BulkWriteError:
                    pass
                ops = []

    if ops:
        try:
            coll.bulk_write(ops, ordered=False)
        except BulkWriteError:
            pass

    print(f"[summary] files_seen={total_files} chunks_total={total_chunks} embedded_now={embedded} skipped_existing={skipped}")
    print("[done] Ingest complete. Remember to create the Atlas Vector Search index (dimensions must match your model).")

if __name__ == "__main__":
    main()
