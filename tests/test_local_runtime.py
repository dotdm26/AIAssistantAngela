import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.desktop.app import system_tray_supported
from src.runtime.config import AppConfig, load_config, save_config
from src.runtime.discord_adapter import DiscordAdapter
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

    def test_system_tray_is_disabled_in_wsl(self):
        with (
            patch.dict(os.environ, {"WSL_DISTRO_NAME": "Ubuntu"}, clear=True),
            patch("src.desktop.app.sys.platform", "linux"),
            patch("src.desktop.app.platform.release", return_value="6.6.0-microsoft-standard-WSL2"),
        ):
            self.assertFalse(system_tray_supported())

    def test_system_tray_can_be_disabled_explicitly(self):
        with patch.dict(os.environ, {"ANGELA_DISABLE_TRAY": "true"}, clear=True):
            self.assertFalse(system_tray_supported())

    def test_discord_adapter_registers_dispatchable_event_names(self):
        adapter = DiscordAdapter(
            config=AppConfig(),
            secrets=object(),
            agent=object(),
        )

        self.assertEqual(adapter.client.on_ready, adapter.on_ready)
        self.assertEqual(adapter.client.on_message, adapter.on_message)


if __name__ == "__main__":
    unittest.main()

