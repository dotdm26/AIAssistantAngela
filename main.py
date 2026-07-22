import os
import asyncio
import discord
from dotenv import load_dotenv
from pydantic_settings import BaseSettings
from typing import Optional, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.agent import AIAgent

load_dotenv()

class Settings(BaseSettings):
    DISCORD_TOKEN: str
    DISCORD_CHANNEL_ID: Optional[str] = None
    USER_ID: Optional[str] = None

settings = Settings()
agent = AIAgent()
HISTORY_LIMIT = max(1, int(os.getenv("HISTORY_LIMIT", "10")))

intents = discord.Intents.default()
intents.message_content = True
client = discord.Client(intents=intents)

def get_allowed_values() -> set[str]:
    allowed_values = {os.getenv("user1"), os.getenv("user2")}
    return {value for value in allowed_values if value}

def is_allowed_user(author) -> bool:
    allowed_values = get_allowed_values()
    if not allowed_values:
        return True

    return (
        str(author.id) in allowed_values
        or author.name in allowed_values
        or author.global_name in allowed_values
    )

def trim_for_discord(text: str, limit: int = 1990) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."

def _log_store_conversation_error(task: asyncio.Task):
    exc = task.exception()
    if exc:
        print(f"Failed to store conversation in background: {exc}")

def schedule_store_conversation(user_key: str, user_message: str, agent_response: str):
    task = asyncio.create_task(agent.store_conversation(user_key, user_message, agent_response))
    task.add_done_callback(_log_store_conversation_error)

def prepare_history(
    user_key: str,
    limit: Optional[int] = HISTORY_LIMIT,
) -> list[Union[HumanMessage, AIMessage, SystemMessage]]:
    history = agent.get_conversation_history(user_key, limit=limit)
    if not history:
        return [SystemMessage(content=agent.system_prompt)]

    history.insert(0, SystemMessage(content=agent.system_prompt))
    return history

async def send_startup_greeting(channel, user_key: str):
    history = prepare_history(user_key)
    agent.conversation_history[user_key] = history

    new_session_prompt = (
        "Greet the user, and if useful, briefly mention a topic in one of your previous conversations."
    )
    greeting_text = await agent.generate_reply(history, new_session_prompt, session_id=user_key)
    if not greeting_text:
        await channel.send("Sorry, I didn't get a usable reply from the model.")
        return

    safe_reply = trim_for_discord(greeting_text)
    await channel.send(safe_reply)
    history.extend([AIMessage(content=greeting_text)])
    schedule_store_conversation(user_key, new_session_prompt, greeting_text)

async def handle_user_message(message):
    prompt = message.content
    if not prompt or not prompt.strip():
        return

    user_key = str(message.author.id)
    history = prepare_history(user_key)
    agent.conversation_history[user_key] = history

    reply_text = await agent.generate_reply(history, prompt, session_id=user_key)
    if not reply_text:
        await message.channel.send("Sorry, I didn't get a usable reply from the model.")
        return

    safe_reply = trim_for_discord(reply_text)
    history.extend([HumanMessage(content=prompt), AIMessage(content=reply_text)])
    await message.channel.send(safe_reply)
    schedule_store_conversation(user_key, prompt, reply_text)


@client.event
async def on_ready():
    print(f"We have logged in as {client.user}")

    if not settings.DISCORD_CHANNEL_ID:
        print("DISCORD_CHANNEL_ID is not set. Startup greeting skipped.")
        return

    channel = client.get_channel(int(settings.DISCORD_CHANNEL_ID))
    if channel:
        await channel.send("```STARTING UP...```")
        await channel.send(
            "**I am Angela, an AI. I am your assistant, your secretary, and someone to whom you can talk. "
            "I hope I can help make your time here a little more comfortable.**"
        )
        await channel.send("**...**")
        await send_startup_greeting(channel, settings.USER_ID)

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if not is_allowed_user(message.author):
        return

    try:
        await handle_user_message(message)
    except Exception as exc:
        print(f"LLM error: {exc}")
        if "RESOURCE_EXHAUSTED" in str(exc) or "429" in str(exc):
            await message.channel.send(
                "The AI service is currently rate-limited or out of quota. Please try again shortly."
            )
        else:
            await message.channel.send("Sorry, I hit an error while responding. ERROR: " + str(exc))

client.run(settings.DISCORD_TOKEN)
