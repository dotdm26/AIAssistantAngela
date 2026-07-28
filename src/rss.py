from discord.ext import tasks, commands
from typing import Awaitable, Callable, Optional
from src.tools.news_feed_tools import get_news_feed as fetch_new_articles


def _trim_for_discord(text: str, limit: int = 1990) -> str:
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."

class RSSMonitor(commands.Cog):
    def __init__(
        self,
        bot,
        db_pool,
        channel_id: int,
        feed_url: str = "http://feeds.bbci.co.uk/news/rss.xml",
        article_summarizer: Optional[Callable[[dict], Awaitable[str]]] = None,
    ):
        self.bot = bot
        self.db_pool = db_pool
        self.channel_id = channel_id
        self.feed_url = feed_url
        self.article_summarizer = article_summarizer
        if not self.rss_ticker.is_running():
            self.rss_ticker.start()

    @tasks.loop(minutes=30)
    async def rss_ticker(self):
        fresh_news = await fetch_new_articles(self.feed_url, self.db_pool)
        if not fresh_news:
            return

        channel = self.bot.get_channel(self.channel_id)
        if channel is None:
            print(f"RSSMonitor: channel {self.channel_id} not found.")
            return

        for item in fresh_news:
            summary_text = ""
            if self.article_summarizer is not None:
                try:
                    summary_text = await self.article_summarizer(item)
                except Exception as error:
                    print(f"RSSMonitor: summary generation failed: {error}")

            summary_block = f"\n\n**Summary:**\n{summary_text}" if summary_text else ""
            payload = f"📰 **New Article:** {item['title']}\n{item['link']}{summary_block}"
            await channel.send(_trim_for_discord(payload))

    @rss_ticker.before_loop
    async def before_ticker(self):
        await self.bot.wait_until_ready()