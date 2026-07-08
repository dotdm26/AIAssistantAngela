import os
import discord
from dotenv import load_dotenv
from agent import AIAgent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")

# Initialize AI Agent
agent = AIAgent()

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)


def is_allowed_user(author):
    allowed_values = {os.getenv("user1"), os.getenv("user2")}
    allowed_values = {value for value in allowed_values if value}

    if not allowed_values:
        return True

    return (
        str(author.id) in allowed_values
        or author.name in allowed_values
        or author.global_name in allowed_values
    )

def trim_for_discord(text: str, limit: int = 1900) -> str:
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit - 3] + "..."

@client.event
async def on_ready():
    #instructions for personality
    #maybe retrieve memory (consider Postgres)
    
    print(f'We have logged in as {client.user}')

    channel = client.get_channel(int(os.getenv("DISCORD_CHANNEL_ID")))
    if channel:
        await channel.send("```STARTING UP...```")
        await channel.send("**I am Angela, an AI. I am your assistant, your secretary, and someone to whom you can talk. I hope I can help make your time here a little more comfortable.**")

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if not is_allowed_user(message.author):
        return

    prompt = message.content
    if not prompt or not prompt.strip():
        return

    user_key = str(message.author.id)
    history = agent.get_conversation_history(user_key, limit=10)
    if not history:
        history = [SystemMessage(content=agent.system_prompt)]
    else:
        history.insert(0, SystemMessage(content=agent.system_prompt))

    agent.conversation_history[user_key] = history

    try:
        reply_text = await agent.generate_reply(history, prompt)
        if not reply_text:
            await message.channel.send("Sorry, I didn't get a usable reply from the model.")
            return

        safe_reply = trim_for_discord(reply_text)
        history.extend([HumanMessage(content=prompt), AIMessage(content=reply_text)])
        
        # Store conversation in database
        agent.store_conversation(user_key, prompt, reply_text)

        # Knowledge storage is currently disabled
        # try:
        #     asyncio.create_task(agent.maybe_add_knowledge(prompt, reply_text))
        # except Exception:
        #     pass
        
        await message.channel.send(safe_reply)
    except Exception as exc:
        print(f"LLM error: {exc}")
        if 'RESOURCE_EXHAUSTED' in str(exc) or '429' in str(exc):
            await message.channel.send('The AI service is currently rate-limited or out of quota. Please try again shortly.')
        else:
            await message.channel.send('Sorry, I hit an error while responding.')
        return
        
client.run(DISCORD_TOKEN)
