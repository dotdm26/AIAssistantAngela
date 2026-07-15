import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("FALLBACK_GOOGLE_API_KEY")
print("has key:", bool(api_key))

from langchain_google_genai import GoogleGenerativeAIEmbeddings

emb = GoogleGenerativeAIEmbeddings(api_key=api_key, model="gemini-embedding-001")
vec = emb.embed_query("Detect embedding dimension.")
print("dimension:", len(vec))
