import os
import discord
from dotenv import load_dotenv

intents = discord.Intents.default()
intents.message_content = True

load_dotenv()

TOKEN = os.getenv("TOKEN")
if not TOKEN:
    raise RuntimeError('TOKEN is not set. Add it to .env or export it in your environment.')

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author == client.user:
        ##parse llm response and send it to the channel
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')

client.run(TOKEN)
