import hashlib
import os
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from typing import List, Union, Dict, Optional, Tuple
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import psycopg2
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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


class AIAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(api_key=GOOGLE_API_KEY, model="gemini-3.1-flash-lite")
        self.embeddings = GoogleGenerativeAIEmbeddings(api_key=GOOGLE_API_KEY, model="gemini-embedding-001")
        self.conversation_history = {}

        # Database connection
        self.db_connection = psycopg2.connect(os.getenv("DATABASE_URL"))
        self._enable_pgvector()
        self.create_tables()
        
        # Agent configuration
        self.system_prompt = """ You are Angela, a highly-advanced Artificial Intelligence whose roles include being a secretary, an assistant, and a companion.
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
            self.system_prompt += f"\n\n{os.getenv('EXTRA_INSTRUCTIONS')}"
    
    #MANUALLY CREATE THE EXTENSION AS SUPERUSER IN POSTGRESQL BEFORE RUNNING THE BOT, OTHERWISE EMBEDDINGS WILL BE STORED AS JSON
    def _enable_pgvector(self):
        """Enable pgvector extension if not already enabled"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.db_connection.commit()
        except Exception as e:
            print(f"Note: pgvector extension not available: {e}. Embeddings will be stored as JSON.")
        
    def create_tables(self):
        """Ensure required tables exist"""
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    user_message TEXT,
                    agent_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB
                )
            """)
            # agent_knowledge table is currently disabled
            # cursor.execute("""
            #     CREATE TABLE IF NOT EXISTS agent_knowledge (
            #         id SERIAL PRIMARY KEY,
            #         key VARCHAR(255) NOT NULL UNIQUE,
            #         value TEXT,
            #         embedding vector(768),
            #         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            #         updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            #     )
            # """)
            self.db_connection.commit()

    def get_conversation_history(self, session_id: str, limit: int = 5) -> List[Union[HumanMessage, AIMessage, SystemMessage]]:
        """Retrieve conversation history for context"""
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                SELECT user_message, agent_response 
                FROM conversations 
                WHERE session_id = %s 
                ORDER BY created_at DESC 
                LIMIT %s
            """, (session_id, limit))
            rows = cursor.fetchall()
            history = []
            for user_msg, agent_resp in reversed(rows):
                history.append(HumanMessage(content=user_msg))
                history.append(AIMessage(content=agent_resp))
            return history

    def get_latest_conversation(self, limit: int = 1) -> List[Tuple[str, str, str]]:
        """Retrieve the most recent conversations for a startup summary."""
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                SELECT session_id, user_message, agent_response
                FROM conversations
                ORDER BY created_at DESC
                LIMIT %s
            """, (limit,))
            return cursor.fetchall()
        
    async def generate_reply(self, history: List[Union[HumanMessage, AIMessage, SystemMessage]], prompt: str) -> str:
        if not prompt or not prompt.strip():
            raise ValueError("Empty prompt")

        # Knowledge lookup is disabled for now
        messages = list(history) + [HumanMessage(content=prompt)]
        sanitized_messages = []
        for message in messages:
            content = getattr(message, "content", None)
            text = _extract_text(content)
            if text:
                sanitized_messages.append(message)

        if hasattr(self.llm, "ainvoke"):
            response = await self.llm.ainvoke(sanitized_messages)
        else:
            response = await asyncio.to_thread(self.llm.invoke, sanitized_messages)

        if hasattr(response, "content"):
            return _extract_text(response.content)

        return _extract_text(response)
    
    def store_conversation(self, session_id: str, user_message: str, agent_response: str):
        """Store a conversation in the database"""
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations (session_id, user_message, agent_response)
                VALUES (%s, %s, %s)
            """, (session_id, user_message, agent_response))
            self.db_connection.commit()

    # Knowledge storage and retrieval is currently disabled.
    # async def decide_knowledge_to_store(
    #     self, prompt: str, response: str
    # ) -> Optional[Tuple[str, str]]:
    #     """Ask the LLM whether this conversation should be stored as knowledge."""
    #     decision_instructions = (
    #         "You are a knowledge curator. Review the user prompt and the assistant reply below. "
    #         "If this exchange contains information about the user, any useful fact, preference, habit, or long-term context that should be stored in the agent's knowledge base, "
    #         "respond with exactly two lines in this format:\n"
    #         "KEY: <short unique identifier>\n"
    #         "VALUE: <concise summary of the knowledge>\n"
    #     )

    #     messages = [
    #         SystemMessage(content=decision_instructions),
    #         HumanMessage(content=f"User: {prompt}\nAssistant: {response}"),
    #     ]

    #     if hasattr(self.llm, "ainvoke"):
    #         decision_result = await self.llm.ainvoke(messages)
    #     else:
    #         decision_result = await asyncio.to_thread(self.llm.invoke, messages)

    #     content = _extract_text(getattr(decision_result, "content", decision_result))
    #     if not content:
    #         return None

    #     if content.strip().upper().startswith("NO"):
    #         return None

    #     lines = [line.strip() for line in content.splitlines() if line.strip()]
    #     key = None
    #     value = None
    #     for line in lines:
    #         if line.upper().startswith("KEY:"):
    #             key = line.split(":", 1)[1].strip()
    #         elif line.upper().startswith("VALUE:"):
    #             value = line.split(":", 1)[1].strip()

    #     if not key and value:
    #         digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]
    #         key = f"knowledge_{digest}"

    #     if not key or not value:
    #         return None

    #     return key, value

    # async def maybe_add_knowledge(self, prompt: str, response: str) -> bool:
    #     """Optionally add knowledge if the LLM decides it is worth storing."""
    #     decision = await self.decide_knowledge_to_store(prompt, response)
    #     if not decision:
    #         return False

    #     key, value = decision
    #     await asyncio.to_thread(self.add_knowledge, key, value)
    #     return True

    # def add_knowledge(self, key: str, value: str):
    #     """Add knowledge to the agent's database with embeddings"""
    #     try:
    #         # Generate embedding for the value
    #         embedding = self.embeddings.embed_query(value)
    #     except Exception as e:
    #         print(f"Error generating embedding: {e}")
    #         embedding = None
        
    #     with self.db_connection.cursor() as cursor:
    #         cursor.execute("""
    #             SELECT id FROM agent_knowledge WHERE key = %s
    #         """, (key,))
    #         exists = cursor.fetchone()
            
    #         if exists:
    #             # Update existing knowledge
    #             if embedding:
    #                 cursor.execute("""
    #                     UPDATE agent_knowledge 
    #                     SET value = %s, embedding = %s::vector, updated_at = CURRENT_TIMESTAMP
    #                     WHERE key = %s
    #                 """, (value, embedding, key))
    #             else:
    #                 cursor.execute("""
    #                     UPDATE agent_knowledge 
    #                     SET value = %s, updated_at = CURRENT_TIMESTAMP
    #                     WHERE key = %s
    #                 """, (value, key))
    #         else:
    #             # Insert new knowledge
    #             if embedding:
    #                 cursor.execute("""
    #                     INSERT INTO agent_knowledge (key, value, embedding)
    #                     VALUES (%s, %s, %s::vector)
    #                 """, (key, value, embedding))
    #             else:
    #                 cursor.execute("""
    #                     INSERT INTO agent_knowledge (key, value)
    #                     VALUES (%s, %s)
    #                 """, (key, value))
    #         self.db_connection.commit()
    # 
    # async def query_knowledge(self, query: str, similarity_threshold: float = 0.5, limit: int = 3) -> Optional[str]:
    #     """Query the agent's knowledge base using semantic similarity"""
    #     try:
    #         # Generate embedding for the query
    #         if hasattr(self.embeddings, "aembed_query"):
    #             query_embedding = await self.embeddings.aembed_query(query)
    #         else:
    #             query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
            
    #         with self.db_connection.cursor() as cursor:
    #             # Try vector similarity search with pgvector
    #             cursor.execute("""
    #                 SELECT value, 1 - (embedding <=> %s::vector) as similarity
    #                 FROM agent_knowledge
    #                 WHERE embedding IS NOT NULL
    #                 ORDER BY similarity DESC
    #                 LIMIT %s
    #             """, (query_embedding, limit))
    #             results = cursor.fetchall()
                
    #             if results and results[0][1] >= similarity_threshold:
    #                 # Format results as context for the LLM
    #                 context = "\n\n".join([result[0] for result in results])
    #                 return context
                
    #             # Fallback to keyword search if no results
    #             cursor.execute("""
    #                 SELECT value
    #                 FROM agent_knowledge
    #                 WHERE key ILIKE %s OR value ILIKE %s
    #                 LIMIT 1
    #             """, (f"%{query}%", f"%{query}%"))
    #             result = cursor.fetchone()
                
    #             if result:
    #                 return result[0]
                
    #             return None
    #     except Exception as e:
    #         print(f"Error querying knowledge: {e}")
    #         return None
    
    def close(self):
        """Clean up resources"""
        self.db_connection.close()
