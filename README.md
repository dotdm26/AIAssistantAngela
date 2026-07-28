# AIAssistantAngela

A Discord-based AI assistant that combines:
- Gemini chat responses via LangChain
- PostgreSQL conversation memory
- Hybrid retrieval (text + semantic)
- Tool calling (custom command memory and Google Calendar tools)
- Automated RSS monitoring with article summaries

## 1. Prerequisites

- Python 3.11+ (3.12 recommended)
- PostgreSQL 14+ (with `pgvector` extension available)
- A Discord bot token
- A Google API key for Gemini

Optional:
- Google Calendar OAuth client secrets if you want calendar tools

## 2. Clone and install

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Create a PostgreSQL database

Create a database (example name: `angela`) and ensure your app user can connect.

Set `DATABASE_URL` in this format:

```text
postgresql://USERNAME:PASSWORD@HOST:5432/DB_NAME
```

Example:

```text
postgresql://dotdm26:your_password@localhost:5432/angela
```

### pgvector requirement

This project stores embeddings in a `vector` column.

Run once as a privileged PostgreSQL user:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

If `vector` is not available, the app still runs but semantic embedding storage/queries may fail.

## 4. Configure environment variables

Create a `.env` file in the project root.

Minimum required:

```dotenv
DISCORD_TOKEN="your_discord_bot_token"
DATABASE_URL="postgresql://USERNAME:PASSWORD@localhost:5432/angela"
TEST_KEY2="your_google_gemini_api_key"
```

Recommended:

```dotenv
DISCORD_CHANNEL_ID="optional_channel_id_for_startup_message"
USER_ID="optional_user_id_for_startup_greeting_context"
NEWS_FEED_URL="http://feeds.bbci.co.uk/news/rss.xml"

# Optional allowlist (bot accepts all users if unset)
user1="your_discord_username_or_id"
user2="another_username_or_id"

# Optional retrieval tuning
HISTORY_LIMIT=10
HYBRID_MIN_PROMPT_CHARS=24
HYBRID_TOP_K=5
HYBRID_EXCLUDE_RECENT_COUNT=4
HYBRID_MIN_SCORE=0.25
HYBRID_MIN_SCORE_MEMORY=0.15
COMMAND_MEMORY_MAX_PROMPT_LEN=16
COMMAND_MEMORY_MIN_USAGE=2
COMMAND_MEMORY_MIN_SHARE=0.6

# Optional local embedding model
LOCAL_EMBEDDING_MODEL="nomic-ai/nomic-embed-text-v1.5"

# Optional extra system behavior appended to prompt
EXTRA_INSTRUCTIONS=""
```

Notes:
- `TEST_KEY2` is what `src/config.py` currently reads as the Gemini key.
- If you prefer a different name like `GOOGLE_API_KEY`, update `src/config.py` accordingly.
- `DISCORD_CHANNEL_ID` is required for RSS posting.
- `NEWS_FEED_URL` defaults to BBC RSS if not set.

## 5. RSS feed summarization

The bot includes a background RSS monitor that runs every 30 minutes.

Behavior:
- Reads the configured RSS feed.
- Stores seen article IDs in PostgreSQL table `seen_articles` to avoid reposting duplicates.
- Fetches article page content from each new entry.
- Uses Gemini to summarize the article into concise bullet points.
- Posts title, link, and summary to the configured Discord channel.

Database note:
- `seen_articles` is created automatically at runtime if it does not exist.

## 6. Optional: Google Calendar tools setup

Calendar tools live in `src/tools/calendar_tools.py`.

To enable them:

1. Create OAuth client credentials in Google Cloud for Calendar API.
2. Download the OAuth client JSON and place it at:

```text
src/tools/client_secrets.json
```

3. First calendar tool call will open browser auth and create:

```text
src/tools/token.json
```

## 7. Run the bot

From project root with venv active:

```bash
python main.py
```

If successful, you should see a login message in the console.

## 8. Project structure (high level)

- `main.py`: Discord bot runtime and message handling
- `src/agent.py`: LLM orchestration, retrieval, and tool execution loop
- `src/rss.py`: RSS polling loop and Discord posting
- `src/conversation_store.py`: PostgreSQL storage, schema setup, hybrid search
- `src/tools/commands.py`: command registration/lookup tools
- `src/tools/news_feed_tools.py`: feed parsing, dedupe, and article text extraction
- `src/tools/calendar_tools.py`: Google Calendar tools
- `src/utils/prompts.py`: system prompt and formatting instructions

## 9. Troubleshooting

### `ModuleNotFoundError` for local packages
Run from project root and use:

```bash
python main.py
```

### `Function must have a docstring if description not provided`
Every LangChain `@tool` function must have a function-level docstring directly under `def`.

### `FileNotFoundError: client_secrets.json`
Place your OAuth client file at:

```text
src/tools/client_secrets.json
```

### PostgreSQL authentication failed
Verify credentials with:

```bash
psql -h localhost -U USERNAME -d DB_NAME
```

Then correct `DATABASE_URL` in `.env`.

### Discord rate-limit or quota errors
If you see `RESOURCE_EXHAUSTED` / `429`, wait and retry after a short period.

### RSS monitor not starting
Check these values in `.env`:
- `DATABASE_URL`
- `DISCORD_CHANNEL_ID`

If either is missing, RSS startup is skipped and a message is logged.

### RSS posts show no summary
Some sites block scraping or return unreadable HTML. In those cases, the bot falls back to feed-level content when available.

## 10. Development notes

- Command memory resolution is session-aware.
- Tool calling supports multiple tools in one conversation turn.
- Calendar tools currently provide read-only access (no event creation/edit yet).
