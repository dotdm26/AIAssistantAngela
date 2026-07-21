# This file handles semantic command registration and retrieval for LLM workflows.

import json
import os
import re

import dotenv
import psycopg2
from langchain_core.tools import tool

dotenv.load_dotenv()

COMMAND_TOKEN_RE = re.compile(r"^[a-z0-9_-]{2,64}$")


def _normalize_command_key(text: str) -> str:
    return (text or "").strip().lower()


def _looks_like_command_key(text: str) -> bool:
    return bool(COMMAND_TOKEN_RE.fullmatch(_normalize_command_key(text)))


def is_command_candidate(text: str, max_len: int = 64) -> bool:
    """Return True when text is a single command-like token under max_len."""
    normalized = _normalize_command_key(text)
    return bool(normalized and len(normalized) <= max_len and " " not in normalized and _looks_like_command_key(normalized))


def _ensure_command_memory_table(conn):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS command_memory (
                id SERIAL PRIMARY KEY,
                session_id VARCHAR(255) NOT NULL,
                trigger_text TEXT NOT NULL,
                normalized_trigger TEXT NOT NULL,
                response_text TEXT NOT NULL,
                first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE (session_id, normalized_trigger, response_text)
            )
            """
        )
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS command_memory_lookup_idx
            ON command_memory (session_id, normalized_trigger, last_seen_at DESC)
            """
        )
    conn.commit()


def detect_command_registration(message: str):
    """Return (command_key, response_text) when user message defines a new command."""
    text = (message or "").strip()
    if not text:
        return None

    patterns = [
        re.compile(
            r"^(?:save|register|add|create)\s+command\s+([a-z0-9_-]{2,64})\s*(?:=>|->|:|=)\s*(.+)$",
            re.IGNORECASE,
        ),
        re.compile(
            r"^(?:when i say|if i say)\s+([a-z0-9_-]{2,64})\s*,?\s*(?:respond|reply|do|run|execute)\s+(.+)$",
            re.IGNORECASE,
        ),
    ]

    for pattern in patterns:
        match = pattern.match(text)
        if not match:
            continue

        key = _normalize_command_key(match.group(1))
        value = match.group(2).strip()
        if _looks_like_command_key(key) and value:
            return key, value

    return None


def detect_command_lookup(message: str):
    """Return a command key when user message asks to run/use a command."""
    text = (message or "").strip()
    if not text:
        return None

    slash = re.match(r"^/([a-z0-9_-]{2,64})$", text, re.IGNORECASE)
    if slash:
        return _normalize_command_key(slash.group(1))

    explicit = re.match(
        r"^(?:run|use|execute|do)\s+(?:command\s+)?([a-z0-9_-]{2,64})$",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return _normalize_command_key(explicit.group(1))

    if is_command_candidate(text):
        return _normalize_command_key(text)

    return None


def _save_command_row(conn, session_id: str, command_key: str, response_text: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO command_memory (
                session_id,
                trigger_text,
                normalized_trigger,
                response_text,
                first_seen_at,
                last_seen_at
            )
            VALUES (%s, %s, %s, %s, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
            ON CONFLICT (session_id, normalized_trigger, response_text)
            DO UPDATE SET
                trigger_text = EXCLUDED.trigger_text,
                last_seen_at = CURRENT_TIMESTAMP
            """,
            (session_id, command_key, command_key, response_text),
        )
    conn.commit()


def _lookup_command_row(conn, session_id: str, command_key: str):
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT response_text
            FROM command_memory
            WHERE session_id = %s
              AND normalized_trigger = %s
            ORDER BY last_seen_at DESC, id DESC
            LIMIT 1
            """,
            (session_id, command_key),
        )
        return cur.fetchone()


@tool
def save_command(command: str, response_text: str, session_id: str = "global") -> str:
    """Save/register a command and its associated functionality/response."""
    try:
        normalized = _normalize_command_key(command)
        if not _looks_like_command_key(normalized):
            return "Invalid command key. Use 2-64 chars: a-z, 0-9, _ or -."
        if not (response_text or "").strip():
            return "Command response/functionality cannot be empty."

        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        try:
            _ensure_command_memory_table(conn)
            _save_command_row(conn, session_id, normalized, response_text.strip())
            return "Command saved successfully."
        finally:
            conn.close()
    except Exception as e:
        return f"An error occurred: {e}"


@tool
def use_command(command: str, session_id: str = "global") -> str:
    """Retrieve the most likely functionality/response associated with a command."""
    try:
        normalized = _normalize_command_key(command)
        if not _looks_like_command_key(normalized):
            return "Invalid command key."

        conn = psycopg2.connect(os.getenv("DATABASE_URL"))
        try:
            _ensure_command_memory_table(conn)
            result = _lookup_command_row(conn, session_id, normalized)
            if result:
                response_text = result[0]
                return f"Command found: {normalized} | response={response_text}"
            return "Command not found."
        finally:
            conn.close()
    except Exception as e:
        return f"An error occurred: {e}"


@tool
def process_command(message: str, session_id: str = "global") -> str:
    """
    Detect whether the message defines a new command or invokes an existing one,
    then return a machine-readable decision/result.
    """
    try:
        registration = detect_command_registration(message)
        if registration:
            command_key, response_text = registration
            save_result = save_command.invoke(
                {
                    "command": command_key,
                    "response_text": response_text,
                    "session_id": session_id,
                }
            )
            return json.dumps(
                {
                    "intent": "register",
                    "command": command_key,
                    "status": "saved" if "successfully" in save_result.lower() else "failed",
                    "result": save_result,
                }
            )

        lookup_key = detect_command_lookup(message)
        if lookup_key:
            use_result = use_command.invoke({"command": lookup_key, "session_id": session_id})
            found = use_result.lower().startswith("command found")
            return json.dumps(
                {
                    "intent": "lookup",
                    "command": lookup_key,
                    "status": "found" if found else "not_found",
                    "result": use_result,
                }
            )

        return json.dumps(
            {
                "intent": "none",
                "status": "ignored",
                "result": "No command registration or command lookup intent detected.",
            }
        )
    except Exception as e:
        return json.dumps({"intent": "error", "status": "failed", "result": str(e)})