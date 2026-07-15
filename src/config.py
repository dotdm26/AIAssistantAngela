import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("TEST_KEY2")
LOCAL_EMBEDDING_MODEL = os.getenv("LOCAL_EMBEDDING_MODEL", "nomic-ai/nomic-embed-text-v1.5")
HYBRID_TOP_K = max(1, int(os.getenv("HYBRID_TOP_K", "3")))
HYBRID_MIN_PROMPT_CHARS = max(1, int(os.getenv("HYBRID_MIN_PROMPT_CHARS", "24")))
HYBRID_EXCLUDE_RECENT_COUNT = max(0, int(os.getenv("HISTORY_LIMIT", "10")))
