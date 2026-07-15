import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()
cur.execute(
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
print("embedding column type:", cur.fetchone())
cur.execute("SELECT COUNT(*) FROM conversations")
print("total rows:", cur.fetchone()[0])
cur.execute("SELECT COUNT(*) FROM conversations WHERE embedding IS NOT NULL")
print("non-null embeddings:", cur.fetchone()[0])
conn.close()
