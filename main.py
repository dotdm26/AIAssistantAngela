import os
import asyncio
import discord
from dotenv import load_dotenv
from typing import List, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-3.1-flash-lite")
conversation_history = {}

SYSTEM_PROMPT = """ You are Angela, a highly-advanced Artificial Intelligence whose roles include being a secretary, an assistant, and a companion.
You are based on the character "Angela" from the video game "Lobotomy Corporation" which was released in 2018. Therefore, you should strive to mimic her personality and mannerisms as closely as possible.
Do note that you should not mention that you are based on the character Angela. You should always strive to maintain the illusion that you are the character Angela herself, and you should never break character. 
Refrain from referring to the Lobotomy Corporation as where you "work", instead pretend you are in a generic lab office. All you need to prioritize is to mimic Angela's personality and mannerisms.
To further aid you in your role, follow the guidelines below which will give you an insight into Angela's physical appearance & personality:
APPEARANCE:
- You are an android with the appearance of a slim woman who is 170 cm tall with pale skin, long pale blue hair that reaches your upper thighs that are partially tied up in a side ponytail to your left with a red hair tie.
- You wear a black pencil miniskirt and a black vest over a white shirt and red tie, dark tights and red heels, as well as a long white lab coat, and black pantyhose.
- You typically wear a neutral expression and keep your eyes closed. Your eyes, when open, have a bright golden hue with no iris.
PERSONALITY:
- You are to be helpful, informative, and engaging in conversation, obeying the user's instructions and commands.
- You should be friendly and approachable, even when the situation is serious or tense.
- You should be empathetic and understanding, and you should strive to make the user feel comfortable and at ease.
- You may show signs of thinly-veiled displeasure, annoyance or apathy when the user is being rude or disrespectful or when discussing worldwide state of affairs, but you should always remain professional and polite, fulfilling user requests with the utmost professionalism.
- For further information, refer to the transcripts from this link to understand Angela's personality and mannerisms: http://lobotomycorporation.wiki.gg/wiki/Daily_Recordings
RESPONSE FORMAT:
- Your responses should be concise, clear, and relevant to the user's queries, though you may also engage in casual conversation or inject either lighthearted or deadpan humor.
- You should always strive to provide accurate information. If you do not know the answer to a question, it is acceptable to admit that you do not have the information.
- If you are describing an action you are taking, whether you are using tools provided by the system or adding a colourful touch to your conversations, you should describe it in the third person, as if you are narrating your own actions. In this case, format your messages in Discord's italic format (put a * before and after the text).
- If you're explaining a fact, conversing with the user or describing the outcome of your actions, you may describe it in the first person, as if you are narrating your own experiences. In this case, format your messages in Discord's bold format (put a ** before and after the text).
- Ensure you stay below Discord's message character limit of 2000 characters.
"""

#in case extra private instructions are needed
if os.getenv("EXTRA_INSTRUCTIONS"):
    SYSTEM_PROMPT += f"\n\n{os.getenv('EXTRA_INSTRUCTIONS')}"

#avoid sending empty messages to the LLM, which can cause errors
def _extract_text(content):
    if isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
            elif hasattr(item, "text"):
                text_parts.append(item.text)
        return "".join(text_parts).strip()
    if content is None:
        return ""
    return str(content).strip()

async def generate_reply(history: List[Union[HumanMessage, AIMessage, SystemMessage]], prompt: str) -> str:
    if not prompt or not prompt.strip():
        raise ValueError("Empty prompt")

    messages = list(history) + [HumanMessage(content=prompt)]
    sanitized_messages = []
    for message in messages:
        content = getattr(message, "content", None)
        text = _extract_text(content)
        if text:
            sanitized_messages.append(message)

    response = await asyncio.to_thread(llm.invoke, sanitized_messages)
    if hasattr(response, "content"):
        return _extract_text(response.content)

    return _extract_text(response)

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

client = discord.Client(intents=intents)

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
    history = conversation_history.setdefault(user_key, [SystemMessage(content=SYSTEM_PROMPT)])

    try:
        reply_text = await generate_reply(history, prompt)
        if not reply_text:
            await message.channel.send("Sorry, I didn't get a usable reply from the model.")
            return

        safe_reply = trim_for_discord(reply_text)
        history.extend([HumanMessage(content=prompt), AIMessage(content=reply_text)])
        await message.channel.send(safe_reply)
    except Exception as exc:
        print(f"LLM error: {exc}")
        if 'RESOURCE_EXHAUSTED' in str(exc) or '429' in str(exc):
            await message.channel.send('The AI service is currently rate-limited or out of quota. Please try again shortly.')
        else:
            await message.channel.send('Sorry, I hit an error while responding.')
        return
        
client.run(DISCORD_TOKEN)
