"""
=============================================================================
ingest.py — Document Ingestion Pipeline
=============================================================================
ROLE IN THE SYSTEM:
    This is the "preparation" step of the RAG pipeline. It runs ONCE (or
    whenever your documents change). It reads raw .txt files, breaks them into
    overlapping chunks, converts each chunk into a numerical embedding, and
    stores everything in ChromaDB so that query.py can search it later.

    Think of this as building the index for a book. You do the hard work once
    upfront so that every future lookup is fast.

WHY A SEPARATE INGESTION STEP?
    Embedding generation is computationally expensive. If you re-embedded all
    your documents on every user question, the system would be unbearably slow.
    By pre-computing embeddings and storing them, each query only needs to embed
    a single sentence (fast) and then search the pre-built index (also fast).
=============================================================================
"""

import os
import glob
from tqdm import tqdm  # Progress bar library — gives visual feedback during ingestion

import chromadb
from chromadb.utils import embedding_functions

# SentenceTransformer converts text into numerical vectors (embeddings)
from sentence_transformers import SentenceTransformer


# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================
# Putting these at the top makes them easy to find and change without hunting
# through the code. This is a common Python best practice.

DOCUMENTS_DIR = "documents"       # Folder where your .txt files live
CHROMA_DB_DIR = "chroma_db"       # Folder where ChromaDB will persist data to disk
COLLECTION_NAME = "documents"     # The name for our collection inside ChromaDB

# The embedding model we use. 'all-MiniLM-L6-v2' is a popular choice because:
#   - It's small (22M parameters) so it runs fast even on CPU
#   - It produces 384-dimensional vectors — compact but expressive
#   - It was specifically trained for semantic similarity tasks
#   - It's completely free and runs locally (no API calls, no rate limits)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Chunk size: how many characters per chunk.
# WHY 500? It's a balance:
#   - Too small: each chunk lacks enough context for the LLM to give a useful answer
#   - Too large: the LLM gets overwhelmed with irrelevant text; also costs more tokens
CHUNK_SIZE = 500

# Overlap: how many characters the end of one chunk shares with the start of the next.
# WHY OVERLAP? Imagine a key sentence sits at the boundary between two chunks.
# Without overlap, that sentence gets split in half and neither chunk contains the
# full thought. With overlap, the sentence appears complete in at least one chunk.
# This ensures no important idea falls "between the cracks."
CHUNK_OVERLAP = 50


def load_documents(directory: str) -> list[dict]:
    """
    Read all .txt files from a directory and return them as a list of dicts.

    WHY THIS FUNCTION EXISTS:
        We need a clean way to load raw text files into memory before processing.
        Separating file I/O into its own function makes the code easier to test
        and extend (e.g., later you could add PDF support here without touching
        the chunking logic).

    PARAMETERS:
        directory (str): Path to the folder containing .txt files.
                         Can be relative (e.g., "documents") or absolute.

    RETURNS:
        list[dict]: Each dict has two keys:
            - "filename": just the file's name (e.g., "python_basics.txt")
            - "content": the full text content of the file as a string
    """
    documents = []

    # glob.glob finds all files matching a pattern.
    # "**/*.txt" means: in any subdirectory, any file ending in .txt
    # recursive=True enables the ** wildcard to match subdirectories
    pattern = os.path.join(directory, "**", "*.txt")
    file_paths = glob.glob(pattern, recursive=True)

    if not file_paths:
        print(f"Warning: No .txt files found in '{directory}'")
        return documents

    print(f"Found {len(file_paths)} document(s) in '{directory}'")

    for file_path in file_paths:
        # 'encoding="utf-8"' ensures we handle special characters correctly.
        # Without specifying encoding, Python uses your system default which
        # may fail on files with accented characters or em-dashes.
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        documents.append({
            "filename": os.path.basename(file_path),  # Just "foo.txt", not "documents/foo.txt"
            "content": content
        })
        print(f"  Loaded: {os.path.basename(file_path)} ({len(content)} chars)")

    return documents


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split a long string into smaller overlapping chunks.

    WHY THIS FUNCTION EXISTS:
        LLMs have a context window limit — they can only process a certain number
        of tokens (roughly words) at once. If you pass an entire 10-page document,
        it won't fit. Chunking breaks documents into pieces that fit within limits.

        More importantly, when the user asks a question, we want to retrieve only
        the RELEVANT piece of a document, not the whole thing. Smaller, focused
        chunks retrieve more precisely than large blobs.

    HOW OVERLAP WORKS:
        If chunk_size=500 and overlap=50, the chunks look like this:
          Chunk 1: characters 0   → 500
          Chunk 2: characters 450 → 950    (starts 50 chars before chunk 1 ends)
          Chunk 3: characters 900 → 1400   (starts 50 chars before chunk 2 ends)
        The 50-character overlap ensures a sentence near a boundary appears whole
        in at least one of the adjacent chunks.

    PARAMETERS:
        text (str): The full text of a document to be split.
        chunk_size (int): Maximum number of characters per chunk.
        overlap (int): Number of characters to repeat between adjacent chunks.

    RETURNS:
        list[str]: A list of text chunks. Each is a substring of the original text.
    """
    chunks = []
    start = 0  # Index of the first character of the current chunk

    while start < len(text):
        end = start + chunk_size  # Index of the last character of the current chunk

        # Extract the chunk. Python slicing is safe even if end > len(text)
        # — it just returns up to the end of the string.
        chunk = text[start:end]

        # Only save non-empty chunks (guards against edge cases at end of file)
        if chunk.strip():
            chunks.append(chunk)

        # Move the start pointer forward, but subtract overlap so the next chunk
        # "backs up" and re-reads the last `overlap` characters of this chunk.
        start += chunk_size - overlap

    return chunks


def ingest_documents():
    """
    Main ingestion pipeline: load documents, chunk them, embed them, store in ChromaDB.

    WHY THIS FUNCTION EXISTS:
        This is the orchestrator — it calls all the other functions in the right
        order and handles the ChromaDB setup. Keeping orchestration logic separate
        from individual steps (loading, chunking, embedding) makes each step
        independently testable.

    RETURNS:
        None. The side effect is that ChromaDB on disk is populated with embeddings.
    """

    # -------------------------------------------------------------------------
    # STEP 1: Load raw documents from disk
    # -------------------------------------------------------------------------
    documents = load_documents(DOCUMENTS_DIR)

    if not documents:
        print("No documents to ingest. Add .txt files to the documents/ folder.")
        return

    # -------------------------------------------------------------------------
    # STEP 2: Set up ChromaDB
    # -------------------------------------------------------------------------
    # PersistentClient stores the database on disk so embeddings survive
    # between runs. Without persistence, you'd have to re-ingest every time.
    print(f"\nConnecting to ChromaDB at '{CHROMA_DB_DIR}'...")
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # delete_collection is called if it exists so we can re-run ingest cleanly.
    # WHY: If you add new documents and re-run ingest.py, you want a fresh start
    # rather than accumulating duplicates. In production you'd want smarter
    # incremental updates, but for learning this is clearer.
    try:
        client.delete_collection(name=COLLECTION_NAME)
        print(f"Deleted existing collection '{COLLECTION_NAME}' for fresh ingestion.")
    except Exception:
        pass  # Collection didn't exist yet — that's fine

    # SentenceTransformerEmbeddingFunction tells ChromaDB to use our local model
    # to generate embeddings automatically when we add documents.
    # WHY USE CHROMADB'S BUILT-IN EMBEDDING FUNCTION?
    # It integrates cleanly so ChromaDB can also embed queries at search time,
    # ensuring the query and document embeddings use the exact same model.
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    # Create a new collection. Think of a collection like a table in SQL —
    # it holds a related set of embeddings with their metadata.
    collection = client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
        # cosine distance measures the angle between vectors.
        # "l2" (Euclidean) measures straight-line distance. Cosine is generally
        # better for text similarity because it's not affected by vector magnitude.
        metadata={"hnsw:space": "cosine"}
    )

    # -------------------------------------------------------------------------
    # STEP 3: Chunk each document and store in ChromaDB
    # -------------------------------------------------------------------------
    print("\nIngesting documents...")

    all_chunks = []    # The text of each chunk
    all_ids = []       # A unique ID for each chunk (ChromaDB requires unique IDs)
    all_metadata = []  # Extra info stored alongside each chunk (for provenance)

    for doc in documents:
        chunks = chunk_text(doc["content"])
        print(f"  {doc['filename']}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            # Create a unique ID. Using filename + index ensures no collisions.
            # IDs must be strings in ChromaDB.
            chunk_id = f"{doc['filename']}_chunk_{i}"

            all_chunks.append(chunk)
            all_ids.append(chunk_id)
            all_metadata.append({
                "filename": doc["filename"],
                "chunk_index": i,
                # Store total chunks so we can understand context later
                "total_chunks": len(chunks)
            })

    # Add all chunks to ChromaDB in one batch call.
    # ChromaDB will call our embedding_fn on each chunk automatically.
    # tqdm wraps the list to display a progress bar in the terminal.
    # WHY BATCH? It's much more efficient than inserting one at a time —
    # the embedding model processes chunks in parallel internally.
    print(f"\nGenerating embeddings for {len(all_chunks)} chunks (this may take a minute)...")

    # We add in batches to show progress; ChromaDB supports large batches fine
    batch_size = 50  # Process 50 chunks at a time for the progress bar
    for i in tqdm(range(0, len(all_chunks), batch_size), desc="Embedding batches"):
        batch_end = min(i + batch_size, len(all_chunks))
        collection.add(
            documents=all_chunks[i:batch_end],
            ids=all_ids[i:batch_end],
            metadatas=all_metadata[i:batch_end]
        )

    print(f"\n✓ Ingestion complete! {len(all_chunks)} chunks stored in ChromaDB.")
    print(f"  Database saved to '{CHROMA_DB_DIR}/'")
    print(f"  You can now run: python chat.py")


# =============================================================================
# ENTRY POINT
# =============================================================================
# This pattern (if __name__ == "__main__") means this block only runs when
# the file is executed directly (python ingest.py), NOT when it's imported
# by another module. This lets other files import our functions without
# triggering ingestion as a side effect.
if __name__ == "__main__":
    ingest_documents()
