import os
import re
from datetime import datetime, timedelta, date
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
    normalized = text.lower().replace("_", " ").replace("[", " ").replace("]", " ").replace('"', " ")
    pattern = re.compile(r'\b' + re.escape(word.lower()) + r'\b')
    return bool(pattern.search(normalized))

def get_vaults(collection) -> list[str]:
    try:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        vaults = sorted(set(m.get("vault", "") for m in all_meta if m.get("vault")))
        return vaults
    except Exception:
        return []


def get_folders_for_vault(collection, vault_name: str = None) -> list[str]:
    """get unique folders, optionally filtered to a specific vault."""
    try:
        all_meta = collection.get(include=["metadatas"])["metadatas"]
        folders = set()
        for m in all_meta:
            if vault_name and m.get("vault") != vault_name:
                continue
            folder = m.get("folder", "")
            if folder:
                folders.add(folder)
        return sorted(folders)
    except Exception:
        return []


def _to_int(d: date) -> int:
    return d.year * 10000 + d.month * 100 + d.day

def _fmt(yyyymmdd: int) -> str:
    s = str(yyyymmdd)
    return f"{s[:4]}-{s[4:6]}-{s[6:]}"

def parse_date_range(query: str, today: date) -> tuple[int, int, str] | None:
    """detect a relative time expression in the query and return
    (start_yyyymmdd, end_yyyymmdd, human_label), or None if none is found.
    weeks are calendar weeks (monday start); windows err slightly wide so a note on
    the boundary isn't dropped."""
    q = query.lower()

    def span(start: date, end: date, label: str):
        return (_to_int(start), _to_int(end), label)

    # "past/last/previous N days|weeks|months|years" (explicit count)
    m = re.search(r"\b(?:past|last|previous)\s+(\d+)\s+(day|week|month|year)s?\b", q)
    if m:
        n = int(m.group(1))
        per = {"day": 1, "week": 7, "month": 30, "year": 365}[m.group(2)]
        start = today - timedelta(days=per * n)
        return span(start, today, f"the past {n} {m.group(2)}{'s' if n != 1 else ''}")

    if "yesterday" in q:
        y = today - timedelta(days=1)
        return span(y, y, "yesterday")
    if "today" in q:
        return span(today, today, "today")

    this_week_start = today - timedelta(days=today.weekday())  # monday of this week
    if "past week" in q:
        return span(today - timedelta(days=7), today, "the past week")
    if "last week" in q or "previous week" in q:
        return span(this_week_start - timedelta(days=7), this_week_start - timedelta(days=1), "last week")
    if "this week" in q:
        return span(this_week_start, today, "this week")

    first_of_month = today.replace(day=1)
    if "past month" in q:
        return span(today - timedelta(days=30), today, "the past month")
    if "last month" in q or "previous month" in q:
        prev_end = first_of_month - timedelta(days=1)
        return span(prev_end.replace(day=1), prev_end, "last month")
    if "this month" in q:
        return span(first_of_month, today, "this month")

    if "last year" in q or "previous year" in q:
        return span(date(today.year - 1, 1, 1), date(today.year - 1, 12, 31), "last year")
    if "this year" in q:
        return span(date(today.year, 1, 1), today, "this year")

    if "recent" in q or "lately" in q:  # also matches "recently"
        return span(today - timedelta(days=30), today, "the last 30 days")

    return None


def build_where_clause(vault_filter: str, folder_filter: str, date_range: tuple | None = None) -> dict | None:
    """build a chroma where clause from vault, folder, and optional date filters."""
    conditions = []
    if vault_filter:
        conditions.append({"vault": {"$eq": vault_filter}})
    if folder_filter:
        conditions.append({"folder": {"$eq": folder_filter}})
    if date_range:
        start, end, _ = date_range
        conditions.append({"date_int": {"$gte": start}})
        conditions.append({"date_int": {"$lte": end}})

    if len(conditions) == 0:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}


def keyword_scan(query: str, collection, vault_filter: str = None, folder_filter: str = None, date_range: tuple | None = None) -> list[dict]:
    stopwords = {
        "the", "was", "last", "time", "when", "did", "with", "have", "about",
        "what", "that", "this", "for", "chatted", "talked", "met", "meeting",
        "chat", "had", "you", "how", "who", "any", "notes", "and", "are",
        "has", "can", "tell", "give", "show", "find", "get", "been", "just",
        "its", "not", "from", "all", "are", "but", "your"
    }

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
        links = meta.get("links", "").lower()
        folder = meta.get("folder", "")
        vault = meta.get("vault", "")
        source_lower = source.lower()

        if vault_filter and vault != vault_filter:
            continue
        if folder_filter and folder != folder_filter:
            continue
        if date_range:
            start, end, _ = date_range
            note_date = meta.get("date_int")
            # only filter out notes that have a date and fall outside the window;
            # undated chunks are kept rather than silently dropped
            if note_date is not None and not (start <= note_date <= end):
                continue

        # links is the backlink signal: a note that links [[Hannah Garcia]] in a
        # people/teams property (or inline) matches a query naming her, the same way
        # obsidian's backlinks panel would surface it. count how many *distinct* query
        # words a note matches so that "hannah garcia" (2 words) outranks the many
        # notes that only match "hannah" (e.g. every [[Hannah Kahn]] note).
        matched = {
            w for w in words
            if whole_word_match(w, tags)
            or whole_word_match(w, links)
            or whole_word_match(w, title)
            or whole_word_match(w, source_lower)
        }

        if matched:
            existing = source_best.get(source)
            if existing is None:
                source_best[source] = {
                    "text": doc,
                    "source": source,
                    "vault": vault,
                    "title": meta.get("title", ""),
                    "folder": folder,
                    "modified": meta.get("modified", ""),
                    "date_int": meta.get("date_int"),
                    "match_count": len(matched),
                    "similarity": 1.0,
                    "match_type": "keyword",
                }
            else:
                # keep the longest chunk's text, but track the best match count
                # seen across all chunks of this note
                if len(doc) > len(existing["text"]):
                    existing["text"] = doc
                existing["match_count"] = max(existing["match_count"], len(matched))

    keyword_chunks = list(source_best.values())
    # stable sorts, applied least-significant first, so the final priority is:
    # real notes before entity stubs -> more query words matched -> more recent.
    # the match-count rank is what separates "hannah garcia" from "hannah" alone.
    keyword_chunks.sort(key=lambda x: x.get("modified", ""), reverse=True)
    keyword_chunks.sort(key=lambda x: x.get("match_count", 0), reverse=True)
    keyword_chunks.sort(key=lambda x: x["source"].startswith("Directory/"))
    return keyword_chunks[:TOP_K]


def _retrieve(query: str, collection, query_embedding, vault_filter, folder_filter, date_range) -> list[dict]:
    """one hybrid retrieval pass for a given (possibly None) date_range."""
    where_clause = build_where_clause(vault_filter, folder_filter, date_range)

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
            "vault": meta.get("vault", ""),
            "title": meta.get("title", ""),
            "folder": meta.get("folder", ""),
            "modified": meta.get("modified", ""),
            "date_int": meta.get("date_int"),
            "similarity": round(1 - dist, 3),
            "match_type": "semantic",
        })

    keyword_chunks = keyword_scan(query, collection, vault_filter, folder_filter, date_range)
    seen_sources = set(c["source"] for c in keyword_chunks)

    merged = keyword_chunks[:]
    for chunk in semantic_chunks:
        if chunk["source"] not in seen_sources:
            merged.append(chunk)
            seen_sources.add(chunk["source"])

    return merged[:TOP_K]


def retrieve_context(query: str, collection, vault_filter: str = None, folder_filter: str = None,
                     today: date = None) -> tuple[list[dict], str | None]:
    """hybrid retrieval (keyword/backlink scan + semantic), optionally scoped to a date
    window parsed from the query. returns (chunks, scope_label). if a window is detected
    but no notes fall inside it, retries unscoped so the user still gets an answer."""
    today = today or datetime.now().date()
    query_embedding = get_embedding(query)
    date_range = parse_date_range(query, today)

    chunks = _retrieve(query, collection, query_embedding, vault_filter, folder_filter, date_range)

    if date_range is None:
        return chunks, None

    start, end, label = date_range
    if chunks:
        return chunks, f"{label} ({_fmt(start)} to {_fmt(end)})"

    # nothing dated inside the window — fall back to an unscoped search
    chunks = _retrieve(query, collection, query_embedding, vault_filter, folder_filter, None)
    return chunks, f"{label}: no notes dated in that window, showing all matches"


def build_system_prompt() -> str:
    today = datetime.now().strftime("%A, %B %d, %Y")
    return f"""You are a knowledgeable assistant with access to the user's Obsidian notes.
Your job is to answer questions grounded in those notes, synthesizing and connecting ideas across them.

Today's date is {today}. Each note excerpt is labeled with its date; use it to reason about
relative time references like "last week" or "recently".

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
        vault_label = f" [{chunk['vault']}]" if chunk.get("vault") else ""
        date_label = f" | date: {_fmt(chunk['date_int'])}" if chunk.get("date_int") else ""
        parts.append(
            f"[Note {i}: {chunk['title']}{vault_label} | {chunk['source']}{date_label} | similarity: {chunk['similarity']}]\n{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


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

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    input_cost = (input_tokens / 1_000_000) * 3.00
    output_cost = (output_tokens / 1_000_000) * 15.00

    usage = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "cost_usd": input_cost + output_cost,
    }

    return response.content[0].text, usage

st.title("Obsidian Chat")
st.caption(f"Vault root: `{VAULT_PATH}`")

collection = get_chroma_collection()

if collection is None:
    st.error("No index found. Run `python index.py` first to index your vaults.")
    st.stop()

doc_count = collection.count()

# init session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_cost" not in st.session_state:
    st.session_state.session_cost = 0.0
if "session_tokens" not in st.session_state:
    st.session_state.session_tokens = 0

with st.sidebar:
    st.header("Settings")

    # vault selector
    vaults = get_vaults(collection)
    vault_options = ["All vaults"] + vaults
    selected_vault_label = st.selectbox("Vault", vault_options)
    vault_filter = None if selected_vault_label == "All vaults" else selected_vault_label

    # folder selector dynamic based on vault selection
    folders = get_folders_for_vault(collection, vault_filter)
    if folders:
        folder_options = ["All folders"] + folders
        selected_folder_label = st.selectbox("Folder", folder_options)
        folder_filter = None if selected_folder_label == "All folders" else selected_folder_label
    else:
        folder_filter = None
        if vault_filter:
            st.caption("No subfolders found in this vault.")

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
    st.caption("To re-index your vaults:")
    st.code("python index.py --update", language="bash")

# check ollama is running
try:
    requests.get(f"{EMBED_BASE_URL}/api/tags", timeout=2)
    ollama_ok = True
except requests.ConnectionError:
    ollama_ok = False

if not ollama_ok:
    st.warning("Ollama is not running. Start it with `ollama serve` in your terminal.")
    st.stop()

if not os.environ.get("ANTHROPIC_API_KEY"):
    st.error("ANTHROPIC_API_KEY environment variable not set.")
    st.stop()

# render chat history
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
                    vault_label = f" [{src['vault']}]" if src.get("vault") else ""
                    st.markdown(
                        f"**{src['title']}**{vault_label} `{src['source']}` — similarity: {src['similarity']}"
                    )

# chat input
if prompt := st.chat_input("Ask a question about your notes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching notes..."):
            try:
                chunks, scope = retrieve_context(prompt, collection, vault_filter, folder_filter)
            except Exception as e:
                st.error(f"Retrieval error: {e}")
                st.stop()

        if scope:
            st.caption(f"Scoped to {scope}")

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
                    vault_label = f" [{src['vault']}]" if src.get("vault") else ""
                    st.markdown(
                        f"**{src['title']}**{vault_label} `{src['source']}` — similarity: {src['similarity']}"
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