[README.md](https://github.com/user-attachments/files/28419829/README.md)
# RAG Chatbot — Learning Project

A command-line Retrieval Augmented Generation (RAG) chatbot built with Python, ChromaDB, sentence-transformers, and the Groq API. Every file, function, and design decision is documented in detail.

---

## What is RAG and Why Does It Exist?

**The Problem with Pure LLMs**

Large language models like LLaMA or GPT are trained on data up to a certain date. They can't know about:
- Your private documents
- Information after their training cutoff
- Proprietary company data
- Niche or highly specific knowledge

If you ask a plain LLM a question it doesn't know, it will often *hallucinate* — confidently stating a plausible-sounding but false answer. There's no mechanism to say "I don't have information about that."

**The RAG Solution**

RAG (Retrieval Augmented Generation) solves this by adding a retrieval step *before* the LLM generates a response:

1. Before answering, search your own documents for relevant passages
2. Inject those passages directly into the prompt
3. Instruct the LLM to answer *only* from those passages
4. The LLM generates a grounded answer it can attribute to sources

This combines the reasoning fluency of LLMs with the factual precision of a search system. The result: answers that are both natural-sounding *and* grounded in real documents.

---

## Architecture — Data Flow

```
INGESTION (run once, offline)
─────────────────────────────
documents/*.txt
       │
       ▼
  [ingest.py]
  Load .txt files
       │
       ▼
  Chunk text into
  500-char overlapping
  segments
       │
       ▼
  sentence-transformers
  (all-MiniLM-L6-v2)
  converts each chunk
  to 384-number vector
  (embedding)
       │
       ▼
  ChromaDB (on disk)
  stores vectors +
  original text +
  metadata


QUERY (runs on every user question)
────────────────────────────────────
  User types question
       │
       ▼
  [query.py]
  Convert question to
  embedding (same model!)
       │
       ▼
  ChromaDB cosine
  similarity search →
  top 3 most similar
  chunks retrieved
       │
       ▼
  Build grounded prompt:
  system: "only use this
  context: [3 chunks]"
  history: [past turns]
  user: [question]
       │
       ▼
  Groq API
  (llama3-8b-8192)
  generates answer
       │
       ▼
  [chat.py]
  Display answer +
  sources used
  Save turn to history
       │
       ▼
  Loop for next question
```

---

## File-by-File Breakdown

### `ingest.py` — The Indexer
**What it does:** Reads every `.txt` file from `documents/`, splits each into overlapping 500-character chunks, converts each chunk to a 384-dimensional embedding vector using `sentence-transformers`, and stores everything in ChromaDB on disk.

**Run it:** `python ingest.py`

**When to re-run:** Whenever you add, remove, or change files in `documents/`. It wipes and rebuilds the database from scratch.

**Key functions:**
- `load_documents(directory)` — Scans a folder and returns all `.txt` file contents
- `chunk_text(text, chunk_size, overlap)` — Splits text into overlapping segments
- `ingest_documents()` — Orchestrates the full ingestion pipeline

---

### `query.py` — The RAG Engine
**What it does:** Takes a user question and conversation history, retrieves the 3 most relevant document chunks from ChromaDB, constructs a carefully grounded prompt, and calls the Groq LLM to generate an answer.

**Not run directly** — imported and called by `chat.py`.

**Key functions:**
- `load_retriever()` — Connects to ChromaDB and returns the collection object
- `retrieve_context(collection, question)` — Embeds the question and runs similarity search
- `build_prompt(question, context_chunks, history)` — Assembles the LLM message array
- `query_rag(collection, question, history)` — Full pipeline: retrieve → prompt → generate → return

---

### `chat.py` — The Conversation Loop
**What it does:** Manages the command-line interface. Handles user input, calls `query_rag()` on each turn, displays responses and sources, maintains rolling conversation history, and handles special commands (`history`, `clear`, `quit`).

**Run it:** `python chat.py`

**Key functions:**
- `run_chat()` — The main conversation loop
- `display_sources(context_chunks)` — Shows which documents were consulted
- `format_history_for_display(history)` — Summarizes past conversation turns

---

### `documents/` — Your Knowledge Base
Put any `.txt` files here. The chatbot will only answer from these documents. Included samples:
- `machine_learning_basics.txt`
- `python_programming.txt`
- `vector_databases.txt`

---

### `chroma_db/` — The Vector Database (auto-generated)
Created by `ingest.py`. Do not edit manually. Safe to delete and re-run `ingest.py` to rebuild.

---

### `requirements.txt` — Dependencies
All Python packages needed, each annotated with its purpose.

---

## Step-by-Step Setup

### 1. Prerequisites
- Python 3.9 or higher
- A free Groq API key from [console.groq.com](https://console.groq.com)
  - Sign in → API Keys → Create API Key

### 2. Navigate to the project folder
```bash
cd rag-chatbot
```

### 3. Create a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate    # On Windows: venv\Scripts\activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```
This may take a few minutes — PyTorch is large (~700MB).

### 5. Set your API key

**On Replit:** The `GROQ_API_KEY` secret is already configured — skip this step.

**Locally:** Create a `.env` file in the `rag-chatbot/` folder:
```bash
cp .env.example .env
# Edit .env and paste your Groq API key
```

### 6. Add your documents
Place `.txt` files in the `documents/` folder. Three sample files are already included. You can add your own — study notes, articles, documentation, anything you want the chatbot to know about.

### 7. Build the vector database
```bash
python ingest.py
```
You'll see it load documents, chunk them, generate embeddings, and save to ChromaDB. This takes 1–2 minutes on first run (downloading the model), then ~10 seconds on subsequent runs.

### 8. Start chatting

**On Replit** (easiest):
```bash
bash rag-chatbot/run.sh
```

**Locally:**
```bash
python chat.py
```
Ask questions about your documents. Try:
- "What is supervised learning?"
- "How do virtual environments work in Python?"
- "What is cosine similarity and why is it used?"
- "What's the difference between a vector database and a regular database?"

---

## Concepts Explained

### What are Embeddings?
An embedding is a way of representing text as a list of numbers that captures its meaning. The key insight: texts with similar meanings end up with similar numbers, even if they use completely different words.

Example: "The cat sat on the mat" and "A feline rested on the rug" will have very similar embeddings because they describe the same thing. "The stock market crashed" will have a very different embedding.

The `all-MiniLM-L6-v2` model was specifically trained on millions of sentence pairs to produce embeddings where semantic similarity correlates with vector proximity. It produces 384 numbers per sentence — that's the "dimension" of the embedding space.

### Why ChromaDB Instead of a Regular Database?
A regular database (PostgreSQL, SQLite) is optimized for exact matches and range queries. "Find all rows where name='Alice'" or "Find products where price < 50."

ChromaDB is optimized for *approximate nearest neighbor* search: "Find the 3 vectors most similar to this query vector." This requires specialized indexing algorithms (HNSW — Hierarchical Navigable Small World graphs) that wouldn't make sense in a relational database.

ChromaDB also handles the embedding storage, indexing, and retrieval in one package — no need to manage a separate search index.

### Why Overlap in Chunking?
If you split a document at hard boundaries, important sentences that span a boundary get cut in half. The beginning of the idea is in chunk N, the conclusion is in chunk N+1. When chunk N is retrieved, the answer is incomplete.

Overlap ensures every sentence appears whole in at least one chunk. With 500-char chunks and 50-char overlap, a sentence at position 480 appears at the end of chunk 1 and the beginning of chunk 2 — it's fully preserved in chunk 2 even if chunk 1 gets retrieved.

---

## Three Ways to Extend This Project

### 1. Add PDF and Web Ingestion
The current ingestion only handles `.txt` files. Real-world RAG systems need to handle PDFs, Word documents, HTML pages, etc. You could extend `ingest.py` using:
- `pypdf` or `pdfplumber` for PDF files
- `beautifulsoup4` for web scraping
- `python-docx` for Word documents

The `chunk_text()` and ChromaDB storage logic would remain unchanged — only the loading step changes.

### 2. Smarter Chunking Strategies
The current character-based chunking is simple but imprecise — it can split mid-sentence. Better strategies:
- **Sentence-based chunking**: Use `nltk` or `spacy` to split at sentence boundaries
- **Semantic chunking**: Use an LLM to identify natural topic breaks in a document
- **Recursive character splitting** (LangChain's approach): Try to split at paragraph → sentence → word boundaries in order of preference

### 3. Add a Web Interface with Flask
Turn the CLI chatbot into a web app. The `query_rag()` function in `query.py` is already pure Python with no CLI coupling — you could wrap it in a Flask route:

```python
@app.route("/chat", methods=["POST"])
def chat():
    question = request.json["question"]
    history = request.json.get("history", [])
    answer, sources = query_rag(collection, question, history)
    return {"answer": answer, "sources": [s["filename"] for s in sources]}
```

Then build a simple HTML/JavaScript frontend that calls this endpoint.

---

## Interview Questions This Project Prepares You For

**"What is RAG and why is it used?"**
RAG (Retrieval Augmented Generation) grounds LLM responses in external documents by retrieving relevant passages before generation. It solves LLM hallucination by restricting the model to only answer from provided context, and it lets you use private or recent data that wasn't in the model's training set.

**"What is a vector embedding?"**
A numerical representation of text that captures semantic meaning as a high-dimensional vector. Semantically similar texts produce vectors that are geometrically close in the embedding space. Models like `sentence-transformers` are trained specifically to produce embeddings where proximity correlates with semantic similarity.

**"How does cosine similarity work?"**
It measures the angle between two vectors rather than the distance between their endpoints. This makes it scale-invariant — a long document and a short document about the same topic will be more similar than a short document about a different topic, even if the word counts differ enormously. Scores range from -1 (opposite) to 1 (identical).

**"Why use a vector database instead of regular search?"**
Traditional keyword search (like SQL `LIKE` or ElasticSearch) matches exact or fuzzy strings. Vector search matches by semantic meaning, so "automobile" finds "car" results even with no string overlap. Vector databases use approximate nearest neighbor algorithms (like HNSW) that make this search fast even with millions of vectors.

**"What is the difference between the embedding model and the LLM?"**
The embedding model (sentence-transformers) converts text to fixed-size numerical vectors for similarity search — it's not generating text, just representing meaning mathematically. The LLM (LLaMA via Groq) generates human-readable answers from a prompt — it's a language model, not a search tool. RAG pipelines use both: embeddings for retrieval, LLM for generation.

**"How do you prevent the LLM from hallucinating?"**
By grounding it with a system prompt that explicitly says "only answer from the provided context, say you don't know if the answer isn't there." The retrieved chunks are injected directly into the prompt, and the instruction prevents the model from going beyond that context. This doesn't eliminate hallucination entirely but dramatically reduces it for in-domain questions.

**"What are the limitations of this approach?"**
- Chunk quality is critical: bad chunking leads to bad retrieval
- Embedding model quality affects retrieval accuracy
- The retrieval step only finds text-level similarity, not logical relationships
- Long documents with context spread across many chunks are hard to retrieve completely
- The system can only answer what's in the documents — it can't reason beyond them
