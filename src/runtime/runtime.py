"""Lifecycle coordinator for the Angela worker."""

from __future__ import annotations

import logging
from collections.abc import Callable

from src.agent import AIAgent
from src.runtime.config import AppConfig
from src.runtime.discord_adapter import DiscordAdapter
from src.runtime.observability import HealthRegistry
from src.runtime.secrets import SecretStore


class AngelaRuntime:
    def __init__(
        self,
        config: AppConfig,
        secrets: SecretStore,
        health_callback: Callable[[dict], None] | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.secrets = secrets
        self.logger = logger or logging.getLogger("angela.runtime")
        self.health = HealthRegistry(on_change=self._health_changed)
        self._health_callback = health_callback
        self.agent: AIAgent | None = None
        self.discord: DiscordAdapter | None = None
        self._stopped = False

    def _health_changed(self, status) -> None:
        if self._health_callback:
            self._health_callback(status.as_dict())

    async def run(self) -> None:
        self._stopped = False
        discord_token = self.secrets.get("discord_token")
        google_api_key = self.secrets.get("google_api_key")
        problems = self.config.validate_runtime(
            discord_token=discord_token,
            google_api_key=google_api_key,
        )
        if problems:
            self.health.update("runtime", "failed", "; ".join(problems))
            raise RuntimeError(" ".join(problems))

        self.health.update("runtime", "starting")
        self.health.update("database", "starting")
        try:
            self.agent = AIAgent(config=self.config, secret_store=self.secrets)
            self.health.update("database", "ready", self.config.storage.backend)
            self.health.update("memory", "ready", self.config.memory.embedding_model)
        except Exception as exc:
            self.health.update("database", "failed", str(exc))
            self.logger.exception("Failed to initialize Angela")
            raise

        self.discord = DiscordAdapter(
            config=self.config,
            secrets=self.secrets,
            agent=self.agent,
            health=self.health,
            logger=logging.getLogger("angela.discord"),
        )
        self.health.update("runtime", "ready")
        try:
            await self.discord.run()
        finally:
            await self.stop()

    async def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        if self.discord:
            await self.discord.stop()
            self.discord = None
        if self.agent:
            self.agent.close()
            self.agent = None
        self.health.update("runtime", "disabled", "Stopped")

