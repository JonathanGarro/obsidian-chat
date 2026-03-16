import os

VAULT_PATH = os.path.expanduser("~/Documents/Cloud Vault")
CHROMA_PATH = "outputs/chroma_db"
EMBED_MODEL = "nomic-embed-text"
EMBED_BASE_URL = "http://localhost:11434"
CLAUDE_MODEL = "claude-sonnet-4-6"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K = 6
INCLUDE_EXTENSIONS = [".md"]
EXCLUDE_FOLDERS = [
    ".obsidian",
    ".trash",
    "templates",
    "Templates",
]
