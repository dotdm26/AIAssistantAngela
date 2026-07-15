"""One-time migration: re-embed existing conversations with Gemini's embedding
model and resize the conversations.embedding column to match its dimension.

The previous embedding column was sized for Ollama's mxbai-embed-large model
(1024 dimensions). Google's gemini-embedding-001 returns 3072-dimension
vectors, so old rows cannot simply be reused - they must be re-embedded.

Usage:
    python migrate_gemini_embeddings.py [--batch-size 50] [--dry-run]
"""
import argparse
import os
import sys
import time

import psycopg2
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-embed existing conversations with Gemini and resize the embedding column."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection URL. Defaults to DATABASE_URL from environment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Number of rows to process per batch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show how many rows would be updated without modifying the database.",
    )
    return parser.parse_args()


def format_vector_literal(embedding):
    return "[" + ",".join(str(float(x)) for x in embedding) + "]"


def fetch_pending_rows(cursor):
    """Fetch rows that still need to be re-embedded (embedding_new is NULL)."""
    cursor.execute(
        "SELECT id, user_message, agent_response FROM conversations WHERE embedding_new IS NULL ORDER BY id"
    )
    return cursor.fetchall()


def embed_with_retry(embeddings_client, text: str, max_retries: int = 6):
    """Call embed_query, retrying with backoff if the API is rate-limited (429)."""
    for attempt in range(max_retries):
        try:
            return embeddings_client.embed_query(text)
        except Exception as exc:
            message = str(exc)
            if "RESOURCE_EXHAUSTED" in message or "429" in message:
                wait_seconds = 30
                print(f"Rate limited, waiting {wait_seconds}s before retrying (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_seconds)
                continue
            raise
    raise RuntimeError("Exceeded max retries due to persistent rate limiting.")


def migrate(db_url: str, batch_size: int, dry_run: bool):
    if not db_url:
        raise ValueError("Database URL is required. Supply --db-url or set DATABASE_URL.")

    api_key = os.getenv("FALLBACK_GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("Google API key required. Set FALLBACK_GOOGLE_API_KEY in environment.")

    embeddings_client = GoogleGenerativeAIEmbeddings(api_key=api_key, model="gemini-embedding-001")

    dummy_embedding = embeddings_client.embed_query("Initialize embedding dimension check.")
    new_dim = len(dummy_embedding)
    print(f"Detected Gemini embedding dimension: {new_dim}")

    connection = psycopg2.connect(db_url)
    try:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'conversations' AND column_name = 'embedding_new'
                """
            )
            if cursor.fetchone() is None:
                print(f"Adding temporary column embedding_new vector({new_dim})...")
                cursor.execute(f"ALTER TABLE conversations ADD COLUMN embedding_new vector({new_dim});")
                connection.commit()

            rows = fetch_pending_rows(cursor)
            total = len(rows)
            print(f"Found {total} conversation rows still needing embeddings.")

            if dry_run:
                print(f"Dry run: {total} rows would be re-embedded and the column would be resized.")
                return

            processed = 0
            for i in range(0, total, batch_size):
                batch = rows[i:i + batch_size]
                for row_id, user_message, agent_response in batch:
                    text = f"{user_message or ''}\n\n{agent_response or ''}".strip()
                    if not text:
                        print(f"Skipping row {row_id} because it contains no text.")
                        continue
                    embedding = embed_with_retry(embeddings_client, text)
                    value = format_vector_literal(embedding)
                    cursor.execute(
                        "UPDATE conversations SET embedding_new = %s::vector WHERE id = %s",
                        (value, row_id),
                    )
                    time.sleep(0.7)
                connection.commit()
                processed += len(batch)
                print(f"Re-embedded {processed}/{total} rows...")

            cursor.execute("SELECT COUNT(*) FROM conversations WHERE embedding_new IS NULL")
            remaining = cursor.fetchone()[0]
            if remaining > 0:
                print(f"{remaining} row(s) still missing embeddings (likely empty text). Skipping column swap.")
                return

            print("Swapping columns: dropping old embedding column and renaming embedding_new...")
            cursor.execute("ALTER TABLE conversations DROP COLUMN embedding;")
            cursor.execute("ALTER TABLE conversations RENAME COLUMN embedding_new TO embedding;")
            connection.commit()
            print(f"Migration complete. conversations.embedding is now vector({new_dim}).")
    finally:
        connection.close()


def main():
    args = parse_args()
    try:
        migrate(args.db_url, args.batch_size, args.dry_run)
    except Exception as exc:
        print(f"Migration failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
