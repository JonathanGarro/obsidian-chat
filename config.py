import os

VAULT_PATH = os.path.expanduser("~/Documents/Cloud Vault/Hewlett")

CHROMA_PATH = "outputs/chroma_db"

# ollama embedding model (run: ollama pull nomic-embed-text)
EMBED_MODEL = "nomic-embed-text"
EMBED_BASE_URL = "http://localhost:11434"

# claude model to use for answering
# remember if i change this, need to update pricing estimate in app.py
CLAUDE_MODEL = "claude-sonnet-4-6"

# chunking settings
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# retrieval settings
TOP_K = 6

# file extensions to index
INCLUDE_EXTENSIONS = [".md"]

# folders to exclud
EXCLUDE_FOLDERS = [
    ".obsidian",
    ".trash",
    "templates",
    "Templates",
]
