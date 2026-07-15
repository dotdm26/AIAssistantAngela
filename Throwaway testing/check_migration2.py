import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()
cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name = 'conversations'")
print("columns:", cur.fetchall())
cur.execute("SELECT COUNT(*) FROM conversations WHERE embedding_new IS NOT NULL")
print("embedding_new non-null:", cur.fetchone()[0])
conn.close()
