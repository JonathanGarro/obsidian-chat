import os
import re
from pathlib import Path
import anthropic
import chromadb
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from config import (
    CHROMA_PATH,
    CLAUDE_MODEL,
    EMBED_MODEL,
    EMBED_BASE_URL,
    TOP_K,
    VAULT_PATH,
)

st.set_page_config(
    page_title="Obsidian Chat",
    page_icon="🗂️",
    layout="wide",
)

@st.cache_resource
def get_chroma_collection():
    if not Path(CHROMA_PATH).exists():
        return None
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        return client.get_collection("obsidian_notes")
    except Exception:
        return None

@st.cache_resource
def get_anthropic_client():
    return anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def get_embedding(text: str) -> list[float]:
    """get embedding from ollama."""
    response = requests.post(
        f"{EMBED_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["embedding"]

def whole_word_match(word, text):
    """return true if word appears as a whole word in text (case-insensitive).
    normalizes underscores and json list syntax before matching so that
    tags like '["People/Firstname_Lastname"]' match queries for 'firstname' or 'lastname'.
    """
    # normalize: lowercase, replace underscores with spaces, strip json
    normalized = text.lower().replace("_", " ").replace("[", " ").replace("]", " ").replace('"', " ")
    pattern = re.compile(r'\b' + re.escape(word.lower()) + r'\b')
    return bool(pattern.search(normalized))

def keyword_scan(query: str, collection, folder_filter: str = None) -> list[dict]:
    stopwords = {
        "the", "was", "last", "time", "when", "did", "with", "have", "about",
        "what", "that", "this", "for", "chatted", "talked", "met", "meeting",
        "chat", "had", "you", "how", "who", "any", "notes", "and", "are",
        "has", "can", "tell", "give", "show", "find", "get", "been", "just",
        "its", "not", "from", "all", "are", "but", "your"
    }

    # extract words from query
    words = [
        w.strip("?,.'\"!").lower()
        for w in query.split()
        if len(w.strip("?,.'\"!")) >= 3 and w.strip("?,.'\"!").lower() not in stopwords
    ]

    if not words:
        return []

    all_data = collection.get(include=["documents", "metadatas"])
    source_best = {}

    for doc, meta in zip(all_data["documents"], all_data["metadatas"]):
        source = meta.get("source", "")
        title = meta.get("title", "").lower()
        tags = meta.get("tags", "").lower()
        folder = meta.get("folder", "")
        source_lower = source.lower()

        if folder_filter and folder_filter != "All folders":
            if folder != folder_filter:
                continue

        # match against tags, title, and source path
        hit = any(
            whole_word_match(w, tags) or
            whole_word_match(w, title) or
            whole_word_match(w, source_lower)
            for w in words
        )

        if hit:
            # keep longest chunk per source
            existing = source_best.get(source)
            if existing is None or len(doc) > len(existing["text"]):
                source_best[source] = {
                    "text": doc,
                    "source": source,
                    "title": meta.get("title", ""),
                    "folder": folder,
                    "modified": meta.get("modified", ""),
                    "similarity": 1.0,
                    "match_type": "keyword",
                }

    keyword_chunks = list(source_best.values())
    keyword_chunks.sort(key=lambda x: x.get("modified", ""), reverse=True)
    return keyword_chunks[:TOP_K]

def retrieve_context(query: str, collection, folder_filter: str = None) -> list[dict]:
    """keyword scan + semantic search, merged and de-duped"""
    query_embedding = get_embedding(query)

    where_clause = None
    if folder_filter and folder_filter != "All folders":
        where_clause = {"folder": {"$eq": folder_filter}}

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=TOP_K,
        where=where_clause,
        include=["documents", "metadatas", "distances"],
    )

    semantic_chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        semantic_chunks.append({
            "text": doc,
            "source": meta.get("source", "unknown"),
            "title": meta.get("title", ""),
            "folder": meta.get("folder", ""),
            "modified": meta.get("modified", ""),
            "similarity": round(1 - dist, 3),
            "match_type": "semantic",
        })

    keyword_chunks = keyword_scan(query, collection, folder_filter)
    seen_sources = set(c["source"] for c in keyword_chunks)

    merged = keyword_chunks[:]
    for chunk in semantic_chunks:
        if chunk["source"] not in seen_sources:
            merged.append(chunk)
            seen_sources.add(chunk["source"])

    return merged[:TOP_K]

def build_system_prompt() -> str:
    return """You are a knowledgeable assistant with access to the user's Obsidian notes. 
Your job is to answer questions grounded in those notes, synthesizing and connecting ideas across them.

Guidelines:
- Answer based on what's in the provided note excerpts
- When you reference a specific note, mention its title or path so the user can find it
- If the notes don't contain enough information to answer confidently, say so clearly
- Synthesize across multiple notes when relevant rather than just summarizing one
- Be direct and concise — the user knows their own notes, they want insight and synthesis
- If asked about something not covered in the retrieved notes, say the notes don't seem to cover it"""


def format_context(chunks: list[dict]) -> str:
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Note {i}: {chunk['title']} | {chunk['source']} | similarity: {chunk['similarity']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def get_folders(collection) -> list[str]:
    try:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        folders = sorted(set(m.get("folder", "") for m in all_meta if m.get("folder")))
        return ["All folders"] + folders
    except Exception:
        return ["All folders"]


def ask_claude(question: str, context_chunks: list[dict], chat_history: list) -> tuple[str, dict]:

    client = get_anthropic_client()

    context_text = format_context(context_chunks)

    messages = []
    for msg in chat_history[:-1]:
        messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({
        "role": "user",
        "content": f"""Here are relevant excerpts from my Obsidian notes:\n\n{context_text}\n\n---\n\nMy question: {question}""",
    })

    response = client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=build_system_prompt(),
        messages=messages,
    )

    # claude sonnet pricing $3 per 1M input tokens, $15 per 1M output tokens
    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    input_cost = (input_tokens / 1_000_000) * 3.00
    output_cost = (output_tokens / 1_000_000) * 15.00
    total_cost = input_cost + output_cost

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": total_cost,
    }

    return response.content[0].text, usage


st.title("Obsidian Chat")
st.caption(f"Vault: `{VAULT_PATH}`")

collection = get_chroma_collection()

if collection is None:
    st.error(
        "No index found. Run `python index.py` first to index your vault.",
        icon="⚠️",
    )
    st.stop()

doc_count = collection.count()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0

with st.sidebar:
    st.header("Settings")

    folders = get_folders(collection)
    folder_filter = st.selectbox("Scope to folder", folders)

    st.divider()
    st.metric("Indexed chunks", doc_count)

    st.divider()
    st.metric("Session cost", f"${st.session_state.session_cost:.4f}")
    st.metric("Session tokens", f"{st.session_state.session_tokens:,}")

    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.session_state.session_cost = 0.0
        st.session_state.session_tokens = 0
        st.rerun()

    st.divider()
    st.caption("To re-index your vault:")
    st.code("python index.py --update", language="bash")

# check ollama
try:
    requests.get(f"{EMBED_BASE_URL}/api/tags", timeout=2)
    ollama_ok = True
except requests.ConnectionError:
    ollama_ok = False

if not ollama_ok:
    st.warning("Ollama is not running. Start it with `ollama serve` in your terminal.")
    st.stop()

# check api key
if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY environment variable not set.")
    st.stop()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("usage"):
            usage = msg["usage"]
            st.caption(
                f"↑ {usage['input_tokens']:,} input · {usage['output_tokens']:,} output · "
                f"{usage['total_tokens']:,} total tokens · ${usage['cost_usd']:.4f}"
            )
        if msg.get("sources"):
            with st.expander("Sources", expanded=False):
                for src in msg["sources"]:
                    st.markdown(
                        f"**{src['title']}** `{src['source']}` — similarity: {src['similarity']}"
                    )

# chat input
if prompt := st.chat_input("Ask a question about your notes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching notes..."):
            try:
                chunks = retrieve_context(prompt, collection, folder_filter)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                st.stop()

        with st.spinner("Thinking..."):
            try:
                answer, usage = ask_claude(prompt, chunks, st.session_state.messages)
            except Exception as e:
                st.error(f"Claude API error: {e}")
                st.stop()

        st.markdown(answer)

        st.caption(
            f"↑ {usage['input_tokens']:,} input · {usage['output_tokens']:,} output · "
            f"{usage['total_tokens']:,} total tokens · ${usage['cost_usd']:.4f}"
        )

        if chunks:
            with st.expander("Sources", expanded=False):
                for src in chunks:
                    st.markdown(
                        f"**{src['title']}** `{src['source']}` — similarity: {src['similarity']}"
                    )

    st.session_state.session_cost += usage["cost_usd"]
    st.session_state.session_tokens += usage["total_tokens"]

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": chunks,
        "usage": usage,
    })

    st.rerun()
