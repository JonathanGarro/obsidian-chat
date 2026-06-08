# obsidian-chat

I love using [Obsidian](https://obsidian.md) for my notes, but as the vault grows it becomes tough to sift through everything. This project uses Streamlit as an interface to chat with the vault via Claude's API.

## how it works

Notes are indexed into a local ChromaDB vector store using Ollama embeddings. When you ask a question, retrieval runs in two passes that are then merged and deduplicated:

1. A **backlink / keyword pass** that matches your query against each note's title, file path, remaining tags, and — most importantly — the entities it links to.
2. A **semantic pass** over the embedded note bodies.

### links-first entity lookup

My vault is links-first: people, teams, and organizations live as `[[wikilinks]]` inside `people:`, `teams:`, and `organizations:` frontmatter properties (plus inline `[[links]]` in note bodies), rather than as `#People/Joe_Smith` tags. The indexer reads those properties and all inline links into a single `links` field on every chunk.

That makes entity questions behave like Obsidian's backlinks panel: asking "what did I discuss with Jane Doe last week" surfaces every note that links `[[Jane Doe]]`, even when her name never appears in the note's title or body. Entity stub notes under `Directory/` (which are mostly empty) are pushed to the bottom of results so a real meeting note always wins.

Tags are still indexed and matched — they're just reserved for note types, status, and cross-cutting themes now, rather than for naming people. If your vault still uses people tags, the keyword pass will continue to match them; the entity behavior above is additive.

### date scoping

Relative time expressions in your query narrow retrieval to a date window: `today`, `yesterday`, `this week`, `last week`, `this/last month`, `this/last year`, `past N days/weeks/months`, and `recently`. Each note is dated by its `date:` frontmatter property when present, falling back to the file's modified time. If nothing falls inside the detected window, the search quietly retries unscoped so you still get an answer, and the app shows you which window it applied. Today's date is also passed to Claude so it can reason about the labels itself.

## setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com/download/mac) running locally

```bash
# pull the embedding model
ollama pull nomic-embed-text

# install dependencies
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# add your API key
echo "ANTHROPIC_API_KEY=your_key_here" > .env
```

Edit `config.py` and set `VAULT_PATH`. Mine points at the parent folder that holds my separate areas — `"~/Documents/Cloud Vault"` — and each top-level subdirectory (Hewlett, Personal, Red Cross) becomes a selectable vault in the sidebar.

## usage

```bash
# index your vault (first run)
python index.py

# incremental update (run this every time you add/remove/rename files)
python index.py --update

# start the app
streamlit run app.py
```

## notes

- `outputs/chroma_db` is local only and not tracked in git — re-index on a new machine
- Ollama must be running before you start the app.
- Vault and folder scoping are available in the sidebar to limit retrieval to a section of your vault; date scoping is inferred from the wording of your question.
- I estimate the cost per run, but this will vary depending on your model and the size of your vault.
- **Schema change:** the `people`/`teams`/`links`/`date_int` fields were added to the index. If you indexed before this change, `--update` won't backfill them (the files themselves haven't changed, so the hash registry skips them). Do one full re-index: delete `outputs/chroma_db` or run `python index.py` without `--update`.
