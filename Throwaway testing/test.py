import psycopg2

conn = psycopg2.connect("postgresql://dotdm26:01072019@localhost:5432/angela")
cursor = conn.cursor()

# Check if tables exist
cursor.execute("""
    SELECT table_name FROM information_schema.tables 
    WHERE table_schema = 'public'
""")
tables = cursor.fetchall()
print("Tables in database:", tables)

conn.close()