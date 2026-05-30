# RAG Chatbot (Learning Project)

A command-line Retrieval Augmented Generation (RAG) chatbot in Python. Ask questions about your own documents — it retrieves relevant passages and uses an LLM to answer grounded in that content.

## Run & Operate

```bash
# First time — builds the vector database and starts chat:
bash rag-chatbot/run.sh

# Or step by step:
/home/runner/workspace/.pythonlibs/bin/python rag-chatbot/ingest.py   # build the DB (once)
/home/runner/workspace/.pythonlibs/bin/python rag-chatbot/chat.py     # start chatting
```

**Python runtime:** `/home/runner/workspace/.pythonlibs/bin/python` (Python 3.11)

## Stack

- Python 3.11
- `sentence-transformers` + `all-MiniLM-L6-v2` — local embeddings (no API key needed)
- `chromadb` — local vector database stored in `rag-chatbot/chroma_db/`
- `groq` — LLM API client (model: `llama-3.1-8b-instant`, free tier)
- `python-dotenv` — loads `.env` secrets locally

## Where things live

```
rag-chatbot/
├── ingest.py          # Ingestion pipeline: read → chunk → embed → store in ChromaDB
├── query.py           # RAG engine: retrieve context → build prompt → call Groq LLM
├── chat.py            # CLI conversation loop with history
├── run.sh             # Convenience launcher (ingest if needed, then chat)
├── requirements.txt   # All dependencies with explanations
├── README.md          # Full learning guide: what RAG is, architecture, setup, interview prep
├── .env.example       # Template for local API key configuration
├── documents/         # Put your .txt files here — the chatbot's knowledge base
│   ├── machine_learning_basics.txt
│   ├── python_programming.txt
│   └── vector_databases.txt
└── chroma_db/         # Auto-generated vector database (created by ingest.py)
```

## Architecture decisions

- **Character-based chunking with overlap** — simple, predictable, easy to understand for a learning project. Overlap (50 chars) prevents key sentences from being split at chunk boundaries.
- **`all-MiniLM-L6-v2` for embeddings** — runs locally (no API cost), 384-dim vectors, fast on CPU, trained specifically for semantic similarity.
- **Cosine distance in ChromaDB** — scale-invariant; better than Euclidean for text where vector magnitude doesn't reflect meaning.
- **Rolling conversation history** — capped at 10 turns to stay within LLaMA's 8192 token context window.
- **Grounded system prompt** — explicitly instructs the LLM to answer only from retrieved context and say "I don't know" otherwise, minimizing hallucination.
- **`llama-3.1-8b-instant` via Groq** — `llama3-8b-8192` was decommissioned; this is the successor on Groq's free tier.

## Product

A CLI chatbot that answers questions about documents you provide. Add `.txt` files to `documents/`, run `ingest.py` to re-index, then ask questions in `chat.py`. Shows which source documents were used for each answer (with relevance %).

## User preferences

_Populate as you build — explicit user instructions worth remembering across sessions._

## Gotchas

- Always use the full Python path: `/home/runner/workspace/.pythonlibs/bin/python` — `python` and `pip` are not on PATH in this Nix environment.
- Re-run `ingest.py` whenever you add or change files in `documents/` — it wipes and rebuilds the database.
- `sentence-transformers` must be installed via `pip` (not via uv/`installLanguagePackages`) due to a platform constraint resolver conflict with the workspace's `uv` config. Use: `pip install sentence-transformers --extra-index-url https://download.pytorch.org/whl/cpu`
- The Groq model `llama3-8b-8192` is decommissioned — use `llama-3.1-8b-instant` instead.
- The `all-MiniLM-L6-v2` model downloads from HuggingFace on first run (~90MB). Subsequent runs use the cached version.

## Pointers

- See `rag-chatbot/README.md` for the full learning guide
- See the `pnpm-workspace` skill for workspace structure, TypeScript setup, and package details
