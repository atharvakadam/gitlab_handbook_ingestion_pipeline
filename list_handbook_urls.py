'''
This script maps local GitLab Handbook markdown files to their web and repository URLs while counting the total documents. 
It handles special cases like index pages and skips hidden files, providing a clear count of all handbook pages.
'''

from pathlib import Path
import argparse
import csv
import sys

BASE_WEB = "https://handbook.gitlab.com/handbook/"
BASE_REPO = "https://gitlab.com/gitlab-com/content-sites/handbook/-/blob/{sha}/content/handbook/"

def is_hidden(path: Path) -> bool:
    return any(part.startswith(".") for part in path.parts)

def path_to_slug(md_path: Path, handbook_root: Path) -> str:
    """
    Returns the relative slug (without .md) under content/handbook.
    Handles index.md and _index.md mapping to directory URLs.
    """
    rel = md_path.relative_to(handbook_root)
    rel_str = rel.as_posix()
    # Normalize index pages
    rel_str = rel_str.replace("_index.md", "").replace("index.md", "")
    if rel_str.endswith(".md"):
        rel_str = rel_str[:-3]  # strip .md
    # Ensure trailing slash for directory-like pages
    if not rel_str.endswith("/"):
        rel_str += "/"
    return rel_str

def path_to_web_url(md_path: Path, handbook_root: Path) -> str:
    slug = path_to_slug(md_path, handbook_root)
    return BASE_WEB + slug

def path_to_repo_url(md_path: Path, handbook_root: Path, sha: str) -> str:
    rel = md_path.relative_to(handbook_root).as_posix()
    return BASE_REPO.format(sha=sha) + rel

def iter_markdown_files(root: Path):
    handbook_root = root / "content" / "handbook"
    if not handbook_root.exists():
        print(f"[error] Could not find {handbook_root}. Did you set --root to the repo directory?", file=sys.stderr)
        sys.exit(1)
    for p in handbook_root.rglob("*.md"):
        if is_hidden(p):
            continue
        yield p, handbook_root

def main():
    ap = argparse.ArgumentParser(description="List GitLab Handbook Markdown pages as web + repo URLs.")
    ap.add_argument("--root", required=True, type=Path, help="Path to local clone of the handbook repo")
    ap.add_argument("--sha", required=True, help="Commit SHA to use for repo blob URLs")
    ap.add_argument("--csv", type=Path, default=None, help="Optional path to write CSV with repo_path,web_url,repo_url")
    ap.add_argument("--print", dest="print_which", choices=["web", "repo", "both"], default="web",
                    help="What to print to stdout (default: web)")
    args = ap.parse_args()

    rows = []
    count = 0
    for md_path, handbook_root in iter_markdown_files(args.root):
        web = path_to_web_url(md_path, handbook_root)
        repo = path_to_repo_url(md_path, handbook_root, args.sha)
        if args.print_which == "web":
            print(web)
        elif args.print_which == "repo":
            print(repo)
        else:
            print(f"{web}\t{repo}")
        rows.append({
            "repo_path": md_path.relative_to(handbook_root).as_posix(),
            "web_url": web,
            "repo_url": repo
        })
        count += 1

    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["repo_path", "web_url", "repo_url"])
            w.writeheader()
            w.writerows(rows)

    print(f"\\n[summary] files_found={count}", file=sys.stderr)

if __name__ == "__main__":
    main()
