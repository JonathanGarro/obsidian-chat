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
    """get embedding from ollama."""
    response = requests.post(
        f"{EMBED_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]

def parse_frontmatter(content: str) -> tuple[dict, str]:
    """extract yaml frontmatter and return (metadata, body)."""
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
    text = re.sub(r"\[\[([^\]|]+)\|([^\]]+)\]\]", r"\2", text)
    text = re.sub(r"\[\[([^\]]+)\]\]", r"\1", text)
    text = re.sub(r"#\w+", "", text)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_by_headers(text: str, max_tokens: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    sections = re.split(r"(?m)(?=^#{1,3} )", text)
    chunks = []

    for section in sections:
        section = section.strip()
        if not section:
            continue
        if len(section) / 4 <= max_tokens:
            chunks.append(section)
        else:
            words = section.split()
            window = max_tokens * 4 // 6
            step = window - overlap
            for i in range(0, len(words), step):
                chunk = " ".join(words[i:i + window])
                if chunk:
                    chunks.append(chunk)

    return chunks


def file_hash(path: Path) -> str:
    """md5 hash of file contents for change detection."""
    return hashlib.md5(path.read_bytes()).hexdigest()

def collect_files(vault_path: str) -> list[tuple[Path, str]]:
    root = Path(vault_path)
    results = []

    for ext in INCLUDE_EXTENSIONS:
        for f in root.rglob(f"*{ext}"):
            parts = f.relative_to(root).parts
            # determine vault name from top-level directory
            vault_name = parts[0] if len(parts) > 1 else ""
            # skip excluded folders (check all path parts)
            if any(part in EXCLUDE_FOLDERS for part in parts):
                continue

            results.append((f, vault_name))

    return sorted(results, key=lambda x: x[0])

def build_index(update_only: bool = False):
    print(f"vault root: {VAULT_PATH}")
    print(f"chroma db: {CHROMA_PATH}")
    print(f"mode: {'incremental update' if update_only else 'full index'}")

    # check ollama is running
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

    # track vaults
    vault_counts: dict[str, int] = {}
    added = 0
    skipped = 0
    updated = 0
    for file_path, vault_name in files:
        rel_path = str(file_path.relative_to(VAULT_PATH))
        current_hash = file_hash(file_path)

        if update_only and registry.get(rel_path) == current_hash:
            skipped += 1
            continue

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

        rel_parts = Path(rel_path).parts
        folder_within_vault = str(Path(*rel_parts[1:-1])) if len(rel_parts) > 2 else ""

        base_metadata = {
            "source": rel_path,
            "vault": vault_name,
            "title": frontmatter.get("title", file_path.stem),
            "tags": json.dumps(tags),
            "folder": folder_within_vault,
            "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        }

        tag_string = " ".join(tags) if tags else ""
        context_prefix = f"Note: {file_path.stem}"
        if tag_string:
            context_prefix += f"\nTags: {tag_string}"
        context_prefix += "\n\n"

        for i, chunk in enumerate(chunks):
            chunk_id = f"{rel_path}::chunk_{i}"
            enriched_chunk = context_prefix + chunk
            try:
                embedding = get_embedding(enriched_chunk)
            except Exception as e:
                print(f"  embedding error for {rel_path} chunk {i}: {e}")
                continue

            collection.upsert(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[{**base_metadata, "chunk_index": i}],
            )

        registry[rel_path] = current_hash
        vault_counts[vault_name] = vault_counts.get(vault_name, 0) + 1
        print(f"  [{vault_name or 'root'}] {rel_path} ({len(chunks)} chunks)")

    Path(CHROMA_PATH).mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2))

    print(f"done. added: {added}, updated: {updated}, skipped: {skipped}")
    for vault_name, count in sorted(vault_counts.items()):
        print(f"  {vault_name or 'root'}: {count} files")
    print(f"total documents in collection: {collection.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="index obsidian vaults into chromadb")
    parser.add_argument(
        "--update",
        action="store_true",
        help="only re-index files that have changed since last run",
    )
    args = parser.parse_args()
    build_index(update_only=args.update)
