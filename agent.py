import hashlib
import os
import asyncio
import subprocess

#from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from typing import List, Union, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import OllamaLLM, OllamaEmbeddings
import psycopg2
from dotenv import load_dotenv

load_dotenv()

ollama_url = os.getenv("OLLAMA_URL")
if not ollama_url:
    try:
        host_ip = subprocess.check_output(
            "ip route | grep default | awk '{print $3}'",
            shell=True,
            text=True,
        ).strip()
        if host_ip:
            ollama_url = f"http://{host_ip}:11434"
    except Exception as e:
        print(f"Error determining host IP: {e}")

if not ollama_url and os.path.exists("/etc/resolv.conf"):
    try:
        with open("/etc/resolv.conf", "r") as f:
            for line in f:
                if line.startswith("nameserver"):
                    parts = line.split()
                    if len(parts) >= 2:
                        host_ip = parts[1].strip()
                        if host_ip:
                            ollama_url = f"http://{host_ip}:11434"
                            break
    except Exception as e:
        print(f"Error reading /etc/resolv.conf: {e}")

if not ollama_url:
    ollama_url = "http://localhost:11434"

print(f"Using Ollama URL: {ollama_url}")

#GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

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
        self.llm = OllamaLLM(model="llama3.2", base_url=ollama_url, keep_alive="30m")
        self.embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=ollama_url)
        self.embedding_dim = self._detect_embedding_dimension()
        self.conversation_history = {}

        # Database connection
        self.db_connection = psycopg2.connect(os.getenv("DATABASE_URL"))
        self._enable_pgvector()
        self.create_tables()
        self._ensure_embedding_column()
        
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
    
    #MANUALLY CREATE THE EXTENSION AS SUPERUSER IN POSTGRESQL BEFORE RUNNING THE BOT
    def _enable_pgvector(self):
        """Enable pgvector extension if not already enabled"""
        try:
            with self.db_connection.cursor() as cursor:
                cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
                self.db_connection.commit()
        except Exception as e:
            print(f"Note: pgvector extension not available: {e}")
        
    def create_tables(self):
        """Ensure required tables exist"""
        with self.db_connection.cursor() as cursor:
            cursor.execute(f"""
                CREATE TABLE IF NOT EXISTS conversations (
                    id SERIAL PRIMARY KEY,
                    session_id VARCHAR(255) NOT NULL,
                    user_message TEXT,
                    agent_response TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    metadata JSONB,
                    embedding vector({self.embedding_dim})
                )
            """)
            self.db_connection.commit()

    def _detect_embedding_dimension(self) -> int:
        """Query the embeddings model once to learn its vector size."""
        sample_text = "Detect embedding dimension."
        if hasattr(self.embeddings, "embed_query"):
            embedding = self.embeddings.embed_query(sample_text)
        else:
            embedding = asyncio.run(self.embeddings.aembed_query(sample_text))
        return len(embedding)

    def _get_embedding_column_info(self, cursor) -> Optional[str]:
        cursor.execute(
            """
            SELECT pg_catalog.format_type(att.atttypid, att.atttypmod) AS type_name
            FROM pg_catalog.pg_attribute att
            JOIN pg_catalog.pg_class cls ON att.attrelid = cls.oid
            JOIN pg_catalog.pg_namespace ns ON cls.relnamespace = ns.oid
            WHERE ns.nspname = 'public'
              AND cls.relname = 'conversations'
              AND att.attname = 'embedding'
              AND att.attnum > 0
            """
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def _ensure_embedding_column(self):
        """Verify the conversations.embedding column matches the active model's dimension.

        Does not modify or delete existing data automatically. If there is a
        mismatch and rows already have embeddings stored, this raises so the
        mismatch can be resolved deliberately (e.g. via a migration script).
        """
        with self.db_connection.cursor() as cursor:
            column_info = self._get_embedding_column_info(cursor)
            if not column_info:
                cursor.execute(f"ALTER TABLE conversations ADD COLUMN embedding vector({self.embedding_dim});")
                self.db_connection.commit()
                return

            import re
            match = re.search(r"vector\((\d+)\)", column_info)
            if not match:
                print(f"Warning: embedding column has unexpected type '{column_info}'; skipping dimension check.")
                return

            current_dim = int(match.group(1))
            if current_dim == self.embedding_dim:
                return

            cursor.execute("SELECT COUNT(*) FROM conversations WHERE embedding IS NOT NULL;")
            non_null_count = cursor.fetchone()[0]
            print(
                f"WARNING: conversations.embedding is vector({current_dim}), but the active "
                f"embeddings model ('mxbai-embed-large') returns vector({self.embedding_dim}). "
                f"{non_null_count} existing row(s) already have embeddings stored in the old "
                "dimension. New inserts will fail until this is resolved. Run a migration to "
                "either re-embed existing rows or recreate the embedding column."
            )

    def _format_messages_for_llm(
        self,
        messages: List[Union[HumanMessage, AIMessage, SystemMessage]],
    ) -> str:
        formatted_lines = []
        for message in messages:
            if isinstance(message, SystemMessage):
                role = "System"
            elif isinstance(message, HumanMessage):
                role = "User"
            elif isinstance(message, AIMessage):
                role = "Assistant"
            else:
                role = "Message"
            content = _extract_text(getattr(message, "content", message))
            if content:
                formatted_lines.append(f"{role}: {content}")
        return "\n".join(formatted_lines).strip()

    def get_conversation_history(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[Union[HumanMessage, AIMessage, SystemMessage]]:
        """Retrieve conversation history for context"""
        with self.db_connection.cursor() as cursor:
            query = """
                SELECT user_message, agent_response
                FROM conversations
                WHERE session_id = %s
                ORDER BY created_at DESC
            """
            params = [session_id]
            if limit is not None:
                query += " LIMIT %s"
                params.append(limit)

            cursor.execute(query, params)
            rows = cursor.fetchall()
            history = []
            for user_msg, agent_resp in reversed(rows):
                history.append(HumanMessage(content=user_msg))
                history.append(AIMessage(content=agent_resp))
            return history
        
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

        llm_input = self._format_messages_for_llm(sanitized_messages)

        if hasattr(self.llm, "agenerate"):
            response = await self.llm.agenerate([llm_input])
            output = _extract_text(response.generations[0][0].text)
        elif hasattr(self.llm, "generate"):
            response = await asyncio.to_thread(self.llm.generate, [llm_input])
            output = _extract_text(response.generations[0][0].text)
        else:
            response = await asyncio.to_thread(self.llm, llm_input)
            output = _extract_text(response)

        return output
    
    async def generate_embedding(self, text: str):
        """Generate an embedding for the given text."""
        if hasattr(self.embeddings, "aembed_query"):
            return await self.embeddings.aembed_query(text)
        return await asyncio.to_thread(self.embeddings.embed_query, text)

    def _save_conversation(self, session_id: str, user_message: str, agent_response: str, embedding):
        with self.db_connection.cursor() as cursor:
            cursor.execute("""
                INSERT INTO conversations (session_id, user_message, agent_response, embedding)
                VALUES (%s, %s, %s, %s)
            """, (session_id, user_message, agent_response, embedding))
            self.db_connection.commit()

    async def store_conversation(self, session_id: str, user_message: str, agent_response: str):
        combined_text = f"{user_message}\n\n{agent_response}"
        embedding = await self.generate_embedding(combined_text)
        await asyncio.to_thread(self._save_conversation, session_id, user_message, agent_response, embedding)

    def close(self):
        """Clean up resources"""
        self.db_connection.close()
