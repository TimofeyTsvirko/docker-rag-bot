#!/usr/bin/env python3
"""
Simple script to download additional Docker-related documentation.

Examples:
  python scripts/download_docs.py --source github --repo docker/docs --path content/manuals/engine
  python scripts/download_docs.py --source url --urls https://docs.docker.com/engine/reference/run/
"""

from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "documents"


def download_github_dir(repo: str, path: str, dest: Path, branch: str = "main") -> None:
    """Clone only the needed path using sparse checkout (requires git)."""
    try:
        from git import Repo
    except ImportError:
        logger.error("GitPython not installed. pip install GitPython")
        return

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        logger.info("Cloning %s (sparse) ...", repo)
        repo_obj = Repo.clone_from(
            f"https://github.com/{repo}.git",
            tmp_path,
            branch=branch,
            depth=1,
            multi_options=["--filter=blob:none", "--sparse"],
        )
        repo_obj.git.sparse_checkout("set", path)
        src = tmp_path / path
        if not src.exists():
            logger.error("Path %s not found in repo", path)
            return
        for f in src.rglob("*"):
            if f.is_file() and f.suffix.lower() in {".md", ".txt", ".rst"}:
                rel = f.relative_to(src)
                target = dest / rel
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, target)
                logger.info("Copied %s", target)


def download_urls(urls: list[str], dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    for url in urls:
        try:
            r = httpx.get(url, follow_redirects=True, timeout=30)
            r.raise_for_status()
            soup = BeautifulSoup(r.text, "html.parser")
            # crude text extraction
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            name = url.rstrip("/").split("/")[-1] or "page"
            out = dest / f"{name}.txt"
            out.write_text(text, encoding="utf-8")
            logger.info("Saved %s", out)
        except Exception as e:
            logger.warning("Failed %s: %s", url, e)


def main():
    parser = argparse.ArgumentParser(description="Download docs into data/documents")
    parser.add_argument("--source", choices=["github", "url"], required=True)
    parser.add_argument("--repo", help="owner/repo for GitHub")
    parser.add_argument("--path", default="", help="subdirectory inside the repo")
    parser.add_argument("--branch", default="main")
    parser.add_argument("--urls", nargs="+", help="list of URLs")
    parser.add_argument("--dest", default=str(DATA_DIR))
    args = parser.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)

    if args.source == "github":
        if not args.repo:
            parser.error("--repo required for github source")
        download_github_dir(args.repo, args.path, dest, args.branch)
    else:
        if not args.urls:
            parser.error("--urls required for url source")
        download_urls(args.urls, dest)

    logger.info("Done. Files are in %s. Run the /ingest endpoint afterwards.", dest)


if __name__ == "__main__":
    main()
