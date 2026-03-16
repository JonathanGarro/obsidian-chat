import os
import re
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime

import chromadb
import requests
import yaml

from config import (
    VAULT_PATH,
    CHROMA_PATH,
    EMBED_MODEL,
    EMBED_BASE_URL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    INCLUDE_EXTENSIONS,
    EXCLUDE_FOLDERS,
)


def get_embedding(text: str) -> list[float]:
    response = requests.post(
        f"{EMBED_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]


def parse_frontmatter(content: str) -> tuple[dict, str]:
    if not content.startswith("---"):
        return {}, content
    try:
        end = content.index("---", 3)
        fm_raw = content[3:end].strip()
        metadata = yaml.safe_load(fm_raw) or {}
        body = content[end + 3:].strip()
        return metadata, body
    except (ValueError, yaml.YAMLError):
        return {}, content


def clean_markdown(text: str) -> str:
    # resolve wikilinks to just the display text
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    # remove tags
    text = re.sub(r"#\w+", "", text)
    # remove html comments
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_by_headers(text: str, max_tokens: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # split on markdown headers
    sections = re.split(r"(?m)(?=^#{1,3} )", text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        # rough token estimate 4 chars per token
        if len(section) / 4 <= max_tokens:
            chunks.append(section)
        else:
            # split long sections into overlapping windows
            words = section.split()
            window = max_tokens * 4 // 6
            step = window - overlap
            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i + window])
                if chunk:
                    chunks.append(chunk)

    return chunks


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def collect_files(vault_path: str) -> list[Path]:
    root = Path(vault_path)
    files = []
    for ext in INCLUDE_EXTENSIONS:
        for f in root.rglob(f"*{ext}"):
            # check if any part of the path is in excluded folders
            parts = f.relative_to(root).parts
            if any(part in EXCLUDE_FOLDERS for part in parts):
                continue
            files.append(f)
    return sorted(files)

def build_index(update_only: bool = False):
    print(f"vault: {VAULT_PATH}")
    print(f"chroma db: {CHROMA_PATH}")
    print(f"mode: {'incremental update' if update_only else 'full index'}")
    print()

    # check ollama
    try:
        requests.get(f"{EMBED_BASE_URL}/api/tags", timeout=5)
    except requests.ConnectionError:
        print("error: ollama is not running. start it with: ollama serve")
        return

    # set up chroma
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="obsidian_notes",
        metadata={"hnsw:space": "cosine"},
    )

    # load existing hash registry for change detection
    registry_path = Path(CHROMA_PATH) / "file_registry.json"
    registry = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())

    files = collect_files(VAULT_PATH)
    print(f"found {len(files)} markdown files")

    added = 0
    skipped = 0
    updated = 0

    for file_path in files:
        rel_path = str(file_path.relative_to(VAULT_PATH))
        current_hash = file_hash(file_path)

        if update_only and registry.get(rel_path) == current_hash:
            skipped += 1
            continue

        # if file was previously indexed remove old chunks
        if rel_path in registry:
            existing = collection.get(where={"source": rel_path})
            if existing["ids"]:
                collection.delete(ids=existing["ids"])
            updated += 1
        else:
            added += 1

        try:
            content = file_path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            print(f"  skipping (encoding error): {rel_path}")
            continue

        frontmatter, body = parse_frontmatter(content)
        body = clean_markdown(body)

        if not body.strip():
            registry[rel_path] = current_hash
            continue

        chunks = chunk_by_headers(body)

        tags = frontmatter.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]

        base_metadata = {
            "source": rel_path,
            "vault": vault_name,
            "title": frontmatter.get("title", file_path.stem),
            "tags": json.dumps(tags),  # chroma requires scalar values
            "folder": str(Path(rel_path).parent),
            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }

        # build a context prefix from title and tags to prepend to each chunk
        tag_string = " ".join(tags) if tags else ""
        context_prefix = f"Note: {file_path.stem}"
        if tag_string:
            context_prefix += f"\nTags: {tag_string}"
        context_prefix += "\n\n"

        # embed and store each chunk
        for i, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}::chunk_{i}"
            # prepend title/tags to the text before embedding so name-based
            # queries match on metadata as well as content
            enriched_chunk = context_prefix + chunk
            try:
                embedding = get_embedding(enriched_chunk)
            except Exception as e:
                print(f"  embedding error for {rel_path} chunk {i}: {e}")
                continue

            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],  # store original chunk (without prefix) for claude context
                metadatas=[{**base_metadata, "chunk_index": i}],
            )

        registry[rel_path] = current_hash
        vault_counts[vault_name] = vault_counts.get(vault_name, 0) + 1
        print(f"  [{vault_name or 'root'}] {rel_path} ({len(chunks)} chunks)")

    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))

    print()
    print(f"done. added: {added}, updated: {updated}, skipped: {skipped}")
    print(f"total documents in collection: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="index obsidian vault into chromadb")
    parser.add_argument(
        "--update",
        action="store_true",
        help="only re-index files that have changed since last run",
    )
    args = parser.parse_args()
    build_index(update_only=args.update)
