from langchain_ollama import OllamaEmbeddings
from pgvector.psycopg2 import register_vector
import psycopg2
import os
from datasets import load_dataset

# Connexion
conn = psycopg2.connect(
    dbname="rag_db", user="postgres", password="1234", host="postgres", port=5432
)
register_vector(conn)  # pour pgvector

# Préparer embedding
emb = OllamaEmbeddings(model="nomic-embed-text",
                       base_url="http://host.docker.internal:11434")

# Charger le dataset
ds = load_dataset("neural-bridge/rag-dataset-12000", split="train")

# Vérifier la dimension de l'embedding et donc changer si nécessaire dans la table
sample_embedding = emb.embed_documents(["test"])[0]
vector_dim = len(sample_embedding)
print("Embedding dimension :", vector_dim)

# Créer la table (si elle n'existe pas)
with conn.cursor() as cur:
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS documents (
            id SERIAL PRIMARY KEY,
            content TEXT,
            embedding vector({vector_dim})
        );
    """)
    # Créer l'index vectoriel pour accélérer les requêtes en utilisant la similarité cosinus (souvent la meilleure pour du texte)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_embedding
        ON documents
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    """)
    conn.commit()

# Ingestion des documents
for i, item in enumerate(ds):
    text = item["context"]
    embedding = emb.embed_documents([text])[0]

    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s);",
            (text, embedding)
        )
    if i % 100 == 0:
        print(f"{i} documents ingérés...")

conn.commit()
print("Ingestion terminée.")
