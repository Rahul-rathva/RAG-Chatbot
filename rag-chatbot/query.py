"""
=============================================================================
query.py — Query Engine: Retrieval + LLM Generation
=============================================================================
ROLE IN THE SYSTEM:
    This module handles ONE question-answer cycle. Given a user's question and
    conversation history, it:
      1. Converts the question to an embedding
      2. Searches ChromaDB for the most relevant document chunks
      3. Builds a carefully crafted prompt that "grounds" the LLM in that context
      4. Calls the Groq API to generate an answer
      5. Returns the answer as a string

    chat.py calls this module in a loop to create a conversation. query.py
    doesn't know or care about the loop — it just handles one turn.

WHY SEPARATE QUERY FROM CHAT?
    Single Responsibility Principle: query.py is purely about "given a question,
    get an answer." chat.py is purely about "manage a conversation loop."
    This separation makes each piece easier to test, modify, and understand.
=============================================================================
"""

import os
import chromadb
from chromadb.utils import embedding_functions
from groq import Groq


# =============================================================================
# CONFIGURATION
# =============================================================================

CHROMA_DB_DIR = "chroma_db"        # Must match what ingest.py used
COLLECTION_NAME = "documents"      # Must match what ingest.py used
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # Must match what ingest.py used — critical!

# How many document chunks to retrieve per query.
# WHY 3? It's a sweet spot:
#   - Too few (1): might miss context from different parts of a document
#   - Too many (10): the LLM prompt gets huge, expensive, and diluted with
#     less-relevant text that can confuse the answer
NUM_RESULTS = 3

# The LLM model to use via Groq.
# llama-3.1-8b-instant means: LLaMA 3.1, 8 billion parameters, optimized for speed.
# WHY THIS MODEL? It's free on Groq, fast, and capable enough for Q&A tasks.
# "8b" is the smaller LLaMA 3.1 — the 70b version is smarter but slower and
# has stricter rate limits on the free tier.
# Note: llama3-8b-8192 was decommissioned; llama-3.1-8b-instant is the successor.
LLM_MODEL = "llama-3.1-8b-instant"

# Maximum tokens the LLM can generate in its response.
# A token is roughly 0.75 words. 512 tokens ≈ 380 words — enough for
# a thorough answer without hitting rate limits.
MAX_TOKENS = 512


def load_retriever():
    """
    Connect to ChromaDB and return the collection object ready for querying.

    WHY THIS FUNCTION EXISTS:
        Creating the ChromaDB client and loading the collection involves a few
        steps. Extracting them into a function keeps query_rag() clean and also
        makes it easy to call load_retriever() once at startup rather than on
        every single query (which would be slow).

    RETURNS:
        chromadb.Collection: The collection object we can call .query() on.

    RAISES:
        SystemExit: If the ChromaDB directory doesn't exist (meaning ingest.py
                    hasn't been run yet), we print a helpful message and exit.
    """
    if not os.path.exists(CHROMA_DB_DIR):
        print(f"Error: ChromaDB database not found at '{CHROMA_DB_DIR}'.")
        print("Please run:  python ingest.py  first to process your documents.")
        raise SystemExit(1)

    # PersistentClient reads from disk — no re-embedding needed
    client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

    # We MUST use the same embedding function as ingest.py.
    # WHY? When you search, the query gets embedded using this model. If the
    # query embedding and the stored document embeddings come from different
    # models, the similarity scores are meaningless — like comparing distances
    # measured in miles to distances measured in kilometers.
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    collection = client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn
    )

    return collection


def retrieve_context(collection, question: str, n_results: int = NUM_RESULTS) -> list[dict]:
    """
    Search ChromaDB for the document chunks most semantically similar to the question.

    HOW SEMANTIC SEARCH WORKS:
        1. The question is converted to an embedding vector (384 numbers)
        2. ChromaDB computes cosine similarity between the question vector and
           every stored chunk vector
        3. The top n_results chunks with the highest similarity scores are returned

    COSINE SIMILARITY EXPLAINED:
        Imagine each embedding as an arrow pointing from the origin in 384-dimensional
        space. Two semantically similar sentences point in roughly the same direction.
        Cosine similarity measures the cosine of the angle between two arrows.
        - cos(0°) = 1.0  → identical direction → very similar meaning
        - cos(90°) = 0.0 → perpendicular → unrelated meaning
        - cos(180°) = -1.0 → opposite direction → opposite meaning

        The beauty of cosine similarity is it finds semantic relationships, not just
        keyword matches. "automobile" and "car" will be close even though the words
        share no characters, because they appear in similar contexts in training data.

    PARAMETERS:
        collection: The ChromaDB collection returned by load_retriever()
        question (str): The user's natural language question
        n_results (int): How many chunks to retrieve

    RETURNS:
        list[dict]: Each dict contains:
            - "text": the chunk's text content
            - "filename": which document it came from
            - "distance": how different it is (lower = more similar for cosine distance)
    """
    # ChromaDB automatically embeds the question using our embedding function
    results = collection.query(
        query_texts=[question],  # List because ChromaDB supports batch queries
        n_results=n_results,
        include=["documents", "metadatas", "distances"]  # What fields to return
    )

    # results is a dict with lists-of-lists because ChromaDB supports batch queries.
    # results["documents"][0] is the list of chunks for our first (only) query.
    chunks = []
    for i in range(len(results["documents"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "filename": results["metadatas"][0][i]["filename"],
            # Distance (not similarity): 0 = identical, 2 = completely different
            "distance": results["distances"][0][i]
        })

    return chunks


def build_prompt(question: str, context_chunks: list[dict], conversation_history: list[dict]) -> list[dict]:
    """
    Construct the message list to send to the LLM.

    WHY CAREFUL PROMPT ENGINEERING?
        Without a carefully designed prompt, an LLM will answer from its training
        data regardless of your documents. It might give plausible-sounding but
        wrong answers, or answer questions your documents don't cover.

        By explicitly telling the model "only use the provided context", we:
          1. Ground the model in your actual documents (RAG's core value)
          2. Make the model say "I don't know" when the answer isn't in your docs
             rather than hallucinating
          3. Make answers attributable — we know exactly which documents were used

    THE MESSAGE FORMAT:
        Groq (and most LLM APIs) use a "chat completion" format:
        - "system" message: sets the AI's role, behavior, and constraints
        - "user" messages: what the human said
        - "assistant" messages: what the AI previously responded

        Including conversation history lets the model understand follow-up questions.
        "What else does it cover?" only makes sense if the model knows what "it" is.

    PARAMETERS:
        question (str): The user's current question
        context_chunks (list[dict]): Retrieved chunks from retrieve_context()
        conversation_history (list[dict]): Previous turns in the format
                                           [{"role": "user"|"assistant", "content": "..."}]

    RETURNS:
        list[dict]: The messages array ready to pass to the Groq API
    """

    # Build a formatted string of all retrieved context chunks
    # We include the source filename so the LLM can reference it and the user
    # can know where information came from.
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"\n--- Context {i} (from: {chunk['filename']}) ---\n"
        context_text += chunk["text"]
        context_text += "\n"

    # The system prompt is the most important part of prompt engineering.
    # We explicitly constrain the model to only use our context.
    # WHY "If the answer is not in the context, say so"?
    # This prevents hallucination — making up plausible-sounding but false answers.
    system_prompt = f"""You are a helpful assistant that answers questions based strictly on the provided context documents.

CONTEXT DOCUMENTS:
{context_text}

INSTRUCTIONS:
- Answer the user's question using ONLY the information in the context documents above
- If the answer is not contained in the context, say "I don't have information about that in the provided documents"
- Be concise but thorough
- When possible, indicate which document your answer comes from
- Do not make up information or use knowledge outside the provided context"""

    # Start with the system message, then append conversation history,
    # then add the new user question at the end.
    messages = [{"role": "system", "content": system_prompt}]

    # Append previous turns so the model has context for follow-up questions
    # WHY INCLUDE HISTORY? Without it, "Tell me more" has no referent.
    messages.extend(conversation_history)

    # Finally, add the current user question
    messages.append({"role": "user", "content": question})

    return messages


def query_rag(collection, question: str, conversation_history: list[dict]) -> tuple[str, list[dict]]:
    """
    Full RAG pipeline: retrieve relevant chunks, build prompt, call LLM, return answer.

    This is the main function that chat.py calls. It orchestrates the retrieval
    and generation steps.

    PARAMETERS:
        collection: ChromaDB collection from load_retriever()
        question (str): The user's question for this turn
        conversation_history (list[dict]): All previous messages in this session

    RETURNS:
        tuple[str, list[dict]]:
            - str: The LLM's answer as a plain string
            - list[dict]: The context chunks that were retrieved (for transparency)
    """

    # -------------------------------------------------------------------------
    # STEP 1: Retrieve semantically relevant chunks
    # -------------------------------------------------------------------------
    context_chunks = retrieve_context(collection, question)

    # -------------------------------------------------------------------------
    # STEP 2: Build the prompt with context and history
    # -------------------------------------------------------------------------
    messages = build_prompt(question, context_chunks, conversation_history)

    # -------------------------------------------------------------------------
    # STEP 3: Call the Groq API
    # -------------------------------------------------------------------------
    # Groq() reads GROQ_API_KEY from the environment automatically.
    # WHY GROQ? It uses custom LPU (Language Processing Unit) hardware that
    # runs LLaMA models 10-20x faster than typical GPU setups, at no cost on
    # the free tier. This makes the chat feel responsive.
    client = Groq()

    # chat.completions.create is the standard OpenAI-compatible endpoint.
    # Groq's API is intentionally API-compatible with OpenAI, so code written
    # for one often works with the other by swapping the client.
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        max_tokens=MAX_TOKENS,
        # temperature controls randomness:
        # 0.0 = deterministic/focused, 1.0 = creative/unpredictable
        # 0.3 is slightly creative but still mostly factual — good for Q&A
        temperature=0.3,
    )

    # Extract the text content from the API response object.
    # The response has nested structure: choices is a list (for multiple completions),
    # [0] gets the first (and only, since we didn't set n>1) completion,
    # .message.content is the actual text.
    answer = response.choices[0].message.content

    return answer, context_chunks
