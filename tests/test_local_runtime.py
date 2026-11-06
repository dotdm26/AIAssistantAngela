import asyncio
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.agent import ToolFilter
from src.desktop.app import system_tray_supported
from src.runtime.backup import backup_data, restore_data
from src.runtime.config import AppConfig, default_data_dir, load_config, save_config
from src.runtime.discord_adapter import DiscordAdapter
from src.runtime.observability import HealthRegistry, redact
from src.storage.sqlite_store import SQLiteConversationStore
from src.utils.embeddings import cosine_similarity


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

    def test_embedding_math_and_backend_defaults(self):
        self.assertAlmostEqual(cosine_similarity([1.0, 0.0], [1.0, 0.0]), 1.0)
        self.assertEqual(AppConfig().memory.backend, "sentence-transformers")

    def test_tool_filter_can_use_shared_embedding_provider(self):
        class FakeEmbeddingProvider:
            def embed_document(self, text):
                return [1.0, 0.0]

            def embed_query(self, text):
                return [1.0, 0.0]

        tool_filter = ToolFilter(FakeEmbeddingProvider())
        selected = tool_filter.get_relevant_tools("search the web", max_rounds=2)
        self.assertEqual(len(selected), 2)

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

    def test_legacy_config_is_migrated_and_persisted(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "channel_id": "123",
                        "database_url": "postgresql://example.invalid/angela",
                    }
                ),
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"ANGELA_STORAGE_BACKEND": "sqlite"}, clear=False):
                config = load_config(config_path)

            self.assertEqual(config.schema_version, 1)
            self.assertEqual(config.discord.channel_id, "123")
            self.assertEqual(config.storage.backend, "postgres")
            persisted = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["schema_version"], 1)
            self.assertNotIn("database_url", persisted)

    def test_windows_default_data_dir_uses_localappdata(self):
        with (
            patch("src.runtime.config.sys.platform", "win32"),
            patch.dict(os.environ, {"LOCALAPPDATA": "C:\\Users\\Angela"}, clear=True),
        ):
            self.assertEqual(str(default_data_dir()), "C:\\Users\\Angela/Angela")

    def test_backup_and_restore_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            restored = root / "restored"
            config = AppConfig(data_dir=source)
            save_config(config)
            store = SQLiteConversationStore(
                config.sqlite_path,
                embedding_dim=3,
                local_embedding_model="test-model",
                hybrid_exclude_recent_count=0,
            )
            store.save_conversation("user-1", "Remember purple", "I will remember purple", [1.0, 0.0, 0.0])
            store.close()

            archive = backup_data(config, root / "angela-backup.zip")
            restore_data(archive, restored)
            restored_config = load_config(restored / "config.json")
            restored_store = SQLiteConversationStore(
                restored_config.sqlite_path,
                embedding_dim=3,
                local_embedding_model="test-model",
                hybrid_exclude_recent_count=0,
            )
            history = asyncio.run(restored_store.get_conversation_history("user-1"))
            restored_store.close()

            self.assertEqual(len(history), 2)
            self.assertEqual(history[0].content, "Remember purple")
            self.assertNotIn(b"DISCORD_TOKEN", archive.read_bytes())


if __name__ == "__main__":
    unittest.main()

