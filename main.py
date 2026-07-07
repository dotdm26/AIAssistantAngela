import os
import discord
from dotenv import load_dotenv
from typing import TypedDict, List
from langchain_core.messages import HumanMessage, AIMessage
from langchain_google_genai import GoogleGenerativeAI
from langgraph.graph import StateGraph, START, END

load_dotenv()

intents = discord.Intents.default()
intents.message_content = True

DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = GoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-2.5-flash-lite")

def process(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]
    response = llm.invoke([user_message])
    response_text = response.content if hasattr(response, "content") else str(response)
    state["messages"].append(AIMessage(content=response_text))
    return state

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

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'We have logged in as {client.user}')

@client.event
async def on_message(message):
    if message.author.bot:
        return

    if not is_allowed_user(message.author):
        return

    if message.content.startswith('$hello'):
        await message.channel.send('Hello!')
        return

    if message.content.startswith('$ask'):
        prompt = message.content[len('$ask'):].strip()
        if not prompt:
            await message.channel.send('Please provide a question after `$ask`.')
            return

        try:
            result = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            reply = result["messages"][-1].content
            safe_reply = trim_for_discord(reply)
            await message.channel.send(safe_reply)
        except Exception as exc:
            print(f"LLM error: {exc}")
            if 'RESOURCE_EXHAUSTED' in str(exc) or '429' in str(exc):
                await message.channel.send('The AI service is currently rate-limited or out of quota. Please try again shortly.')
            else:
                await message.channel.send('Sorry, I hit an error while responding.')
        return

client.run(DISCORD_TOKEN)
