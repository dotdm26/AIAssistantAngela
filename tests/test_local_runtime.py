import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.runtime.config import load_config, save_config
from src.runtime.observability import HealthRegistry, redact
from src.storage.sqlite_store import SQLiteConversationStore


class LocalRuntimeTests(unittest.TestCase):
    def test_sqlite_history_and_hybrid_search(self):
        async def exercise():
            with tempfile.TemporaryDirectory() as directory:
                store = SQLiteConversationStore(
                    Path(directory) / "angela.db",
                    embedding_dim=3,
                    local_embedding_model="test-model",
                    hybrid_exclude_recent_count=0,
                )
                store.save_conversation(
                    "user-1",
                    "What is the weather?",
                    "It is sunny.",
                    [1.0, 0.0, 0.0],
                )
                history = await store.get_conversation_history("user-1")
                matches = await store.hybrid_search_conversations(
                    "weather",
                    "user-1",
                    lambda _: asyncio.sleep(0, result=[1.0, 0.0, 0.0]),
                )
                store.close()

                self.assertEqual(len(history), 2)
                self.assertEqual(matches[0][0], "What is the weather?")

        asyncio.run(exercise())

    def test_config_round_trip_with_sqlite(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict(
                os.environ,
                {
                    "ANGELA_DATA_DIR": directory,
                    "ANGELA_STORAGE_BACKEND": "sqlite",
                },
            ):
                config = load_config()
                self.assertEqual(config.storage.backend, "sqlite")
                path = save_config(config)
                self.assertTrue(path.exists())
                self.assertEqual(load_config(path).storage.backend, "sqlite")

    def test_health_and_redaction(self):
        health = HealthRegistry()
        health.update("database", "ready", "sqlite")
        self.assertEqual(health.snapshot()[0]["state"], "ready")
        self.assertNotIn("not-secret", redact("api_key=not-secret"))


if __name__ == "__main__":
    unittest.main()

