import feedparser
import asyncpg
import re
from html import unescape

from typing import Any
import httpx

#CREATE TABLE IF NOT EXISTS seen_articles (
#    article_guid TEXT PRIMARY KEY,
#    title TEXT NOT NULL,
#    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
#);

async def ensure_seen_articles_table(db_pool: asyncpg.Pool) -> None:
    async with db_pool.acquire() as connection:
        await connection.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_articles (
                article_guid TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )
            """
        )


def _extract_article_text_from_html(html: str, max_chars: int = 12000) -> str:
    text = re.sub(r"<script\b[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style\b[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = unescape(text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        return text[:max_chars]
    return text


async def fetch_article_text(url: str, timeout_seconds: float = 12.0) -> str:
    if not url:
        return ""

    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; AngelaBot/1.0; +https://github.com/)"
    }
    try:
        async with httpx.AsyncClient(timeout=timeout_seconds, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if "html" not in content_type.lower():
                return ""
            return _extract_article_text_from_html(response.text)
    except Exception as error:
        print(f"Error fetching article content from {url}: {error}")
        return ""


async def get_news_feed(url: str, db_pool: asyncpg.Pool) -> list[dict[str, Any]]:
    """Fetch latest feed entries that have not been posted before."""
    await ensure_seen_articles_table(db_pool)

    try:
        feed = feedparser.parse(url)
        if feed.bozo:
            raise ValueError(f"Failed to parse feed: {feed.bozo_exception}")

        if getattr(feed, "status", None) == 304:
            return []

        entries = feed.entries[:3]
        news_feed: list[dict[str, Any]] = []

        async with db_pool.acquire() as connection:
            for entry in entries:
                guid = entry.get("id") or entry.get("link")
                if not guid:
                    continue

                existing_article = await connection.fetchval(
                    "SELECT 1 FROM seen_articles WHERE article_guid = $1",
                    guid,
                )
                if existing_article:
                    continue

                title = entry.get("title", "Untitled")
                await connection.execute(
                    "INSERT INTO seen_articles (article_guid, title) VALUES ($1, $2)",
                    guid,
                    title,
                )

                news_feed.append(
                    {
                        "title": title,
                        "link": entry.get("link", ""),
                        "published": entry.get("published", ""),
                        "summary": entry.get("summary", ""),
                        "content": await fetch_article_text(entry.get("link", "")),
                    }
                )

        return news_feed
    except Exception as error:
        print(f"Error fetching news feed: {error}")
        return []
