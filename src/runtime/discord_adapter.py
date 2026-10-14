"""Discord transport for the interface-independent Angela agent."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import asyncpg
import discord
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent import AIAgent
from src.rss import RSSMonitor
from src.runtime.config import AppConfig
from src.runtime.observability import HealthRegistry
from src.runtime.secrets import SecretStore
from src.utils.prompts import (
    configure_formatting,
    rss_article_summary_instructions,
    rss_article_summary_prompt,
)


def trim_for_discord(text: str, limit: int = 1990) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def extract_model_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        value = content.get("text")
        return value if isinstance(value, str) else ""
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str) and part.strip():
                chunks.append(part.strip())
            elif isinstance(part, dict) and isinstance(part.get("text"), str) and part["text"].strip():
                chunks.append(part["text"].strip())
        return "\n".join(chunks)
    return ""


class DiscordAdapter:
    """Own the Discord client and translate Discord events into agent calls."""

    def __init__(
        self,
        config: AppConfig,
        secrets: SecretStore,
        agent: AIAgent,
        health: HealthRegistry | None = None,
        logger: logging.Logger | None = None,
    ):
        self.config = config
        self.secrets = secrets
        self.agent = agent
        self.health = health or HealthRegistry()
        self.logger = logger or logging.getLogger("angela.discord")
        self.rss_monitor: Optional[RSSMonitor] = None
        self.rss_db_pool: Optional[asyncpg.Pool] = None
        self._startup_handled = False

        intents = discord.Intents.default()
        intents.message_content = True
        self.client = discord.Client(intents=intents)
        self.client.event(self._on_ready)
        self.client.event(self._on_message)

    def _allowed_values(self) -> set[str]:
        return {value for value in self.config.discord.allowed_users if value}

    def _is_allowed_user(self, author) -> bool:
        allowed_values = self._allowed_values()
        if not allowed_values:
            return True
        return (
            str(author.id) in allowed_values
            or author.name in allowed_values
            or author.global_name in allowed_values
        )

    async def _prepare_history(self, user_key: str | None):
        history = await self.agent.get_conversation_history(
            user_key,
            limit=self.config.memory.history_limit,
        )
        if not history:
            return [SystemMessage(content=self.agent.system_prompt)]
        history.insert(0, SystemMessage(content=self.agent.system_prompt))
        return history

    def _schedule_store(self, user_key: str, prompt: str, response: str) -> None:
        task = asyncio.create_task(self.agent.store_conversation(user_key, prompt, response))

        def report_failure(completed: asyncio.Task) -> None:
            try:
                exception = completed.exception()
            except asyncio.CancelledError:
                return
            if exception:
                self.logger.error("Failed to store conversation: %s", exception)

        task.add_done_callback(report_failure)

    async def _send_startup_greeting(self, channel, user_key: str | None) -> None:
        history = await self._prepare_history(user_key)
        session_key = user_key or "discord-startup"
        self.agent.conversation_history[session_key] = history
        prompt = "Greet the user, and if useful, briefly mention a topic in one previous conversation."
        greeting = await self.agent.generate_reply(history, prompt, session_id=session_key)
        if not greeting:
            await channel.send("Sorry, I didn't get a usable reply from the model.")
            return
        await channel.send(trim_for_discord(greeting))
        history.append(AIMessage(content=greeting))
        self._schedule_store(session_key, prompt, greeting)

    async def _handle_user_message(self, message) -> None:
        prompt = message.content
        if not prompt or not prompt.strip():
            return

        user_key = str(message.author.id)
        history = await self._prepare_history(user_key)
        self.agent.conversation_history[user_key] = history
        response = await self.agent.generate_reply(history, prompt, session_id=user_key)
        if not response:
            await message.channel.send("Sorry, I didn't get a usable reply from the model.")
            return
        history.extend([HumanMessage(content=prompt), AIMessage(content=response)])
        await message.channel.send(trim_for_discord(response))
        self._schedule_store(user_key, prompt, response)

    async def _summarize_article(self, article: dict) -> str:
        title = (article.get("title") or "Untitled").strip()
        source_text = (article.get("content") or article.get("summary") or "").strip()[:9000]
        if not source_text:
            return "No readable content was found for this article."

        messages = [
            SystemMessage(
                content=(
                    f"{self.agent.system_prompt}\n\n"
                    f"{configure_formatting(is_tool_task=True)}\n\n"
                    f"{rss_article_summary_instructions()}"
                )
            ),
            HumanMessage(content=rss_article_summary_prompt(title, source_text)),
        ]
        try:
            if hasattr(self.agent.llm, "ainvoke"):
                response = await self.agent.llm.ainvoke(messages)
            else:
                response = await asyncio.to_thread(self.agent.llm.invoke, messages)
            summary = extract_model_text(getattr(response, "content", "")).strip()
            return trim_for_discord(summary or "Summary unavailable.", limit=1200)
        except Exception as exc:
            self.logger.exception("Article summarization failed")
            return "Summary unavailable due to a model error."

    async def _ensure_rss_monitor(self) -> None:
        if self.rss_monitor is not None:
            return
        if not self.config.discord.channel_id:
            self.health.update("rss", "disabled", "No Discord channel is configured.")
            return
        if self.config.storage.backend != "postgres" or not self.config.storage.database_url:
            self.health.update("rss", "disabled", "RSS currently requires PostgreSQL storage.")
            return

        try:
            self.rss_db_pool = await asyncpg.create_pool(
                dsn=self.config.storage.database_url,
                min_size=1,
                max_size=3,
            )
            self.rss_monitor = RSSMonitor(
                bot=self.client,
                db_pool=self.rss_db_pool,
                channel_id=int(self.config.discord.channel_id),
                feed_url=self.config.news_feed_url,
                article_summarizer=self._summarize_article,
            )
            self.health.update("rss", "ready", f"Monitoring {self.config.news_feed_url}")
        except Exception as exc:
            self.health.update("rss", "failed", str(exc))
            self.logger.exception("Failed to start RSS monitor")

    async def _on_ready(self):
        self.health.update("discord", "ready", f"Connected as {self.client.user}")
        self.logger.info("Connected to Discord as %s", self.client.user)
        await self._ensure_rss_monitor()

        if self._startup_handled:
            return
        self._startup_handled = True
        channel_id = self.config.discord.channel_id
        if not channel_id:
            self.logger.info("No Discord channel configured; startup greeting skipped")
            return
        channel = self.client.get_channel(int(channel_id))
        if channel is None:
            self.logger.warning("Configured Discord channel %s was not found", channel_id)
            return
        await channel.send("```STARTING UP...```")
        await channel.send(
            "**I am Angela, an AI. I am your assistant, your secretary, and someone to whom you can talk. "
            "I hope I can help make your time here a little more comfortable.**"
        )
        await channel.send("**...**")
        await self._send_startup_greeting(channel, self.config.discord.startup_user_id)

    async def _on_message(self, message):
        if message.author.bot or not self._is_allowed_user(message.author):
            return
        try:
            await self._handle_user_message(message)
        except Exception as exc:
            self.logger.exception("LLM message handling failed")
            if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
                await message.channel.send("The AI service is currently rate-limited or out of quota.")
            else:
                await message.channel.send("Sorry, I hit an error while responding.")

    async def run(self) -> None:
        token = self.secrets.get("discord_token")
        if not token:
            raise RuntimeError("Discord bot token is not configured.")
        self.health.update("discord", "starting")
        await self.client.start(token)

    async def stop(self) -> None:
        if self.rss_monitor is not None:
            self.rss_monitor.rss_ticker.cancel()
            self.rss_monitor = None
        if self.rss_db_pool is not None:
            await self.rss_db_pool.close()
            self.rss_db_pool = None
        if not self.client.is_closed():
            await self.client.close()
        self.health.update("discord", "disabled", "Stopped by user")

