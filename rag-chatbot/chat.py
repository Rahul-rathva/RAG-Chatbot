"""
=============================================================================
chat.py — Command-Line Chat Interface
=============================================================================
ROLE IN THE SYSTEM:
    This is the entry point — the file you run to actually use the chatbot.
    It manages the conversation loop: displaying the prompt, reading user
    input, calling the RAG query engine, showing the response, and maintaining
    conversation history across turns.

    chat.py knows nothing about embeddings, ChromaDB, or LLM APIs. It just
    manages the user interaction and delegates the hard work to query.py.
    This clean separation of concerns is a core software design principle.

HOW TO RUN:
    1. First, run: python ingest.py   (builds the vector database)
    2. Then, run: python chat.py      (starts the conversation)
=============================================================================
"""

import os
import sys
from dotenv import load_dotenv  # Loads environment variables from .env file

# Import our query engine functions
from query import load_retriever, query_rag

# load_dotenv() reads the .env file (if it exists) and sets environment variables.
# This is how GROQ_API_KEY gets into os.environ so the Groq client can find it.
# On Replit, secrets are already in the environment, so this is mainly useful
# for local development.
load_dotenv()


def print_separator(char: str = "─", width: int = 60) -> None:
    """
    Print a horizontal divider line for visual formatting.

    WHY THIS FUNCTION EXISTS:
        The terminal output needs visual structure to be readable. Instead of
        repeating print("─" * 60) everywhere, we have one reusable function.
        Even small utilities like this are worth extracting — they signal intent
        and make the code self-documenting.

    PARAMETERS:
        char (str): The character to repeat. Default "─" (Unicode box-drawing char).
        width (int): How many characters wide the line should be.

    RETURNS:
        None (prints to stdout as a side effect)
    """
    print(char * width)


def display_sources(context_chunks: list[dict]) -> None:
    """
    Print the source documents that were retrieved for a given query.

    WHY SHOW SOURCES?
        Transparency is a core feature of RAG. By showing which documents the
        answer came from, the user can:
          1. Verify the answer against the source material
          2. Know when the answer might be incomplete (if sources seem off-topic)
          3. Understand the system's reasoning

        This is sometimes called "grounding" — connecting AI outputs to real sources.
        It's a key difference between RAG and a pure LLM: you can trace every claim.

    PARAMETERS:
        context_chunks (list[dict]): The chunks returned by retrieve_context(),
                                     each with "filename" and "distance" keys.

    RETURNS:
        None
    """
    print("\n📚 Sources consulted:")
    for i, chunk in enumerate(context_chunks, 1):
        # Similarity = 1 - distance for cosine distance. This converts the raw
        # distance metric into a percentage that's easier to interpret.
        # distance=0 means identical → similarity=100%
        # distance=1 means very different → similarity=0%
        similarity = (1 - chunk["distance"]) * 100
        print(f"  {i}. {chunk['filename']}  (relevance: {similarity:.0f}%)")


def format_history_for_display(conversation_history: list[dict]) -> None:
    """
    Print a summary of the current conversation history to show how many turns
    have occurred. Called if the user types 'history'.

    PARAMETERS:
        conversation_history (list[dict]): The accumulated message history

    RETURNS:
        None
    """
    if not conversation_history:
        print("No conversation history yet.")
        return

    # Each turn = 1 user message + 1 assistant message = 2 entries in history
    turns = len(conversation_history) // 2
    print(f"\nConversation has {turns} turn(s) so far.")
    for i, msg in enumerate(conversation_history):
        role_label = "You" if msg["role"] == "user" else "Assistant"
        # Show only first 80 chars of each message to avoid flooding the terminal
        preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
        print(f"  [{role_label}]: {preview}")


def run_chat():
    """
    Main conversation loop — the heart of the CLI chatbot.

    WHY A LOOP?
        A chatbot needs to process multiple questions in sequence, maintaining
        state (the conversation history) between turns. A while loop is the
        natural structure for "keep doing this until the user quits."

    HOW CONVERSATION HISTORY WORKS:
        conversation_history is a list of dicts with "role" and "content" keys.
        After each turn we append:
          {"role": "user", "content": question}
          {"role": "assistant", "content": answer}

        On the next turn, this history is passed to build_prompt() in query.py,
        which includes it in the LLM's messages. The LLM can then understand
        follow-up questions like "what else?" or "can you elaborate on that?"

        WHY NOT INCLUDE UNLIMITED HISTORY?
            LLMs have context window limits (e.g., 8192 tokens for llama3-8b).
            We keep only the last MAX_HISTORY_TURNS turns to avoid exceeding this.
            Older turns matter less for the current conversation anyway.

    RETURNS:
        None
    """

    # Keep only the last N conversation turns to stay within token limits.
    # 10 turns = 20 messages (10 user + 10 assistant). At ~50 tokens each,
    # that's ~1000 tokens — well within the 8192 token limit.
    MAX_HISTORY_TURNS = 10

    # -------------------------------------------------------------------------
    # STARTUP
    # -------------------------------------------------------------------------
    print_separator("═")
    print("  RAG Chatbot — Powered by ChromaDB + Groq (LLaMA 3)")
    print_separator("═")
    print("\nConnecting to document database...")

    # Load ChromaDB once at startup. We don't reload it on every question
    # because loading from disk has latency. Once loaded, queries are fast.
    try:
        collection = load_retriever()
    except SystemExit:
        # load_retriever() already printed a helpful error message
        sys.exit(1)

    # Count how many chunks are stored so the user knows the database is ready
    doc_count = collection.count()
    print(f"✓ Connected! Database contains {doc_count} document chunks.")
    print("\nCommands:")
    print("  Type your question and press Enter")
    print("  'history' — show conversation history")
    print("  'clear'   — clear conversation history")
    print("  'quit' or 'exit' — end the session")
    print_separator()

    # conversation_history stores all previous turns as a list of message dicts.
    # Starts empty — no prior context on first turn.
    conversation_history = []

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    while True:
        print()  # Blank line for readability

        # Get user input. strip() removes leading/trailing whitespace.
        # The try/except catches Ctrl+C and Ctrl+D so the user can exit cleanly.
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            # KeyboardInterrupt = Ctrl+C, EOFError = Ctrl+D (EOF on stdin)
            print("\n\nGoodbye!")
            break

        # Skip empty inputs — user just pressed Enter
        if not user_input:
            continue

        # Handle special commands
        # .lower() makes commands case-insensitive ("QUIT" = "quit" = "Quit")
        command = user_input.lower()

        if command in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        elif command == "history":
            format_history_for_display(conversation_history)
            continue

        elif command == "clear":
            conversation_history = []
            print("✓ Conversation history cleared.")
            continue

        # -------------------------------------------------------------------------
        # RAG QUERY
        # -------------------------------------------------------------------------
        print("\nSearching documents and generating response...")
        print_separator("·")

        try:
            # query_rag does the heavy lifting: retrieve → prompt → LLM → answer
            answer, context_chunks = query_rag(
                collection=collection,
                question=user_input,
                conversation_history=conversation_history
            )
        except Exception as e:
            # Catching broad Exception here because many things could go wrong:
            # network issues with Groq API, rate limits, malformed responses, etc.
            # We print the error but don't crash the whole program — the user can
            # try again or ask a different question.
            print(f"\n⚠ Error generating response: {e}")
            print("Please try again. If the error persists, check your GROQ_API_KEY.")
            continue

        # Display the answer
        print(f"\nAssistant: {answer}")

        # Show which documents were used (transparency feature)
        display_sources(context_chunks)

        print_separator()

        # -------------------------------------------------------------------------
        # UPDATE CONVERSATION HISTORY
        # -------------------------------------------------------------------------
        # Append this turn to history so future questions can reference it.
        conversation_history.append({"role": "user", "content": user_input})
        conversation_history.append({"role": "assistant", "content": answer})

        # Trim history to the last MAX_HISTORY_TURNS turns.
        # Each turn = 2 messages, so we keep the last MAX_HISTORY_TURNS * 2 messages.
        # List slicing [-n:] returns the last n elements — negative index counts from end.
        if len(conversation_history) > MAX_HISTORY_TURNS * 2:
            conversation_history = conversation_history[-(MAX_HISTORY_TURNS * 2):]


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    run_chat()
