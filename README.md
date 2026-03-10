# obsidian-chat

I love using [Obsidian](https://obsidian.md) for my notes, but as the vault grows it becomes tough to sift through everything. This project uses Streamlit as an interface to chat with the vault via Claude's API. 

## how it works

Notes are indexed into a local ChromaDB vector store using Ollama embeddings. When you ask a question, retrieval runs in two passes. First, keyword matching against note titles, tags, and file paths, then semantic search and then the merged results are passed to Claude as context.

Tag-based lookup handles people and proper nouns (e.g. querying "joe smith" will surface notes tagged `People/Joe_Smith` even if the name doesn't appear in the note title). Your tag usage may vary, so tweaks to the code will be necessary for other users.

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

Edit `config.py` and set `VAULT_PATH` to your vault. Mine is `"~/Documents/Cloud Vault/Hewlett"`

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
- Folder scoping is available in the sidebar to limit retrieval to a specific section of your vault
- I estimate the cost per run, but this will vary depending on your model and the size of your vault.