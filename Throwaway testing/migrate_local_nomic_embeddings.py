"""One-time migration: re-embed existing conversations with a local Nomic model.

This script re-embeds rows into a temporary `embedding_new` column and then swaps
it into place as `embedding` once complete.

Usage:
    python migrate_local_nomic_embeddings.py [--batch-size 50] [--dry-run]
"""

import argparse
import os
import sys

# Avoid HF Xet/CAS path issues in some environments (401 Unauthorized on public repos).
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import psycopg2
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Re-embed existing conversations with local Nomic embeddings."
    )
    parser.add_argument(
        "--db-url",
        default=os.getenv("DATABASE_URL"),
        help="PostgreSQL connection URL. Defaults to DATABASE_URL from environment.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5"),
        help="Local embedding model name.",
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


def embed_document(model, text: str):
    vector = model.encode(
        f"search_document: {text}",
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return vector.tolist()


def fetch_pending_rows(cursor):
    cursor.execute(
        "SELECT id, user_message, agent_response FROM conversations WHERE embedding_new IS NULL ORDER BY id"
    )
    return cursor.fetchall()


def migrate(db_url: str, model_name: str, batch_size: int, dry_run: bool):
    if not db_url:
        raise ValueError("Database URL is required. Supply --db-url or set DATABASE_URL.")

    model = SentenceTransformer(model_name, trust_remote_code=True)
    sample = embed_document(model, "Initialize embedding dimension check.")
    dim = len(sample)
    print(f"Using local model: {model_name}")
    print(f"Detected embedding dimension: {dim}")

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
                print(f"Adding temporary column embedding_new vector({dim})...")
                cursor.execute(f"ALTER TABLE conversations ADD COLUMN embedding_new vector({dim});")
                connection.commit()

            rows = fetch_pending_rows(cursor)
            total = len(rows)
            print(f"Found {total} conversation rows still needing embeddings.")

            if dry_run:
                print(f"Dry run: {total} rows would be re-embedded and the column would be resized.")
                return

            processed = 0
            for i in range(0, total, batch_size):
                batch = rows[i : i + batch_size]
                for row_id, user_message, agent_response in batch:
                    text = f"{user_message or ''}\n\n{agent_response or ''}".strip()
                    if not text:
                        print(f"Skipping row {row_id} because it contains no text.")
                        continue
                    embedding = embed_document(model, text)
                    value = format_vector_literal(embedding)
                    cursor.execute(
                        "UPDATE conversations SET embedding_new = %s::vector WHERE id = %s",
                        (value, row_id),
                    )

                connection.commit()
                processed += len(batch)
                print(f"Re-embedded {processed}/{total} rows...")

            cursor.execute("SELECT COUNT(*) FROM conversations WHERE embedding_new IS NULL")
            remaining = cursor.fetchone()[0]
            if remaining > 0:
                print(f"{remaining} row(s) still missing embeddings. Skipping column swap.")
                return

            print("Swapping columns: dropping old embedding column and renaming embedding_new...")
            cursor.execute("ALTER TABLE conversations DROP COLUMN embedding;")
            cursor.execute("ALTER TABLE conversations RENAME COLUMN embedding_new TO embedding;")
            connection.commit()
            print(f"Migration complete. conversations.embedding is now vector({dim}).")
    finally:
        connection.close()


def main():
    args = parse_args()
    try:
        migrate(args.db_url, args.model, args.batch_size, args.dry_run)
    except Exception as exc:
        print(f"Migration failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
