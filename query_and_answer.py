from langchain_ollama import OllamaEmbeddings, OllamaLLM
from pgvector.psycopg2 import register_vector
import psycopg2

# Configuration
DB_CONFIG = {
    "dbname": "rag_db",
    "user": "postgres", 
    "password": "1234",
    "host": "postgres",
    "port": 5432
}

def search_documents(question, top_k=10):
    """Recherche les documents les plus similaires"""
    conn = psycopg2.connect(**DB_CONFIG)
    register_vector(conn)
    embeddings = OllamaEmbeddings(model="nomic-embed-text",
                                  base_url="http://host.docker.internal:11434")
    
    # Crée embedding de la question
    query_embedding = embeddings.embed_documents([question])[0]
    
    # Recherche avec une approche hybride : d'abord par mots-clés, puis par similarité
    with conn.cursor() as cur:
        # Extraire des mots-clés de la question
        keywords = question.lower().split()
        
        # Recherche combinée : similarité vectorielle + mots-clés
        cur.execute("""
            SELECT content, 
                   1 - (embedding <=> %s::vector) as similarity,
                   CASE 
                       WHEN lower(content) LIKE ANY(%s) THEN 1.0
                       ELSE 0.0
                   END as keyword_match
            FROM documents 
            WHERE length(content) > 100
            ORDER BY 
                keyword_match DESC,
                embedding <=> %s::vector
            LIMIT %s;
        """, (query_embedding, [f'%{kw}%' for kw in keywords], query_embedding, top_k))
        results = cur.fetchall()
    
    conn.close()
    return results

def ask_question(question):
    """Questionne le RAG"""
    print(f"Question posée : {question}")
    
    # Récupére documents pertinents
    doc_results = search_documents(question, top_k=10)
    
    if not doc_results:
        return "Aucun document trouvé."
    
    # Affiche les scores
    print(f"Scores de similarité: {[f'{doc[1]:.3f}' for doc in doc_results[:5]]}")
    print(f"Matches mots-clés: {[f'{doc[2]:.0f}' for doc in doc_results[:5]]}")
    
    # Prends les 3 meilleurs documents
    docs = [doc[0] for doc in doc_results[:3]]
    
    # Affiche les documents sources
    print(f"Premier document : {docs[0][:100]}")
    print(f"Deuxième document : {docs[1][:100]}")
    print(f"Troisième document : {docs[2][:100]}")
    print("="*50)
    
    # Crée le contexte avec les documents complets
    context = "\n\n--- Document ---\n\n".join(docs)
    
    # Génére réponse avec un prompt plus efficace
    llm = OllamaLLM(model="gemma",
                    base_url="http://host.docker.internal:11434")
    prompt = f"""You are a knowledgeable assistant. Answer the question using the information from the provided documents.

Documents:
{context}

Question: {question}

Instructions:
- Read through ALL the text in ALL document very carefully
- Look for the information requested in the question
- If you find the answer anywhere in the documents, provide it clearly
- The answer can be anywhere in the document, not just at the beginning

Answer:"""
    
    print("Génération de la réponse...")
    return llm.invoke(prompt)

# Utilisation simple
if __name__ == "__main__":
    print("RAG démarré!")
    
    while True:
        question = input("\n Question (ou 'quit'): ").strip()
        
        if question.lower() in ['quit', 'q', 'exit']:
            print("Au revoir!")
            break
            
        if question:
            try:
                response = ask_question(question)
                print(f"\n {response}\n" + "="*60)
            except Exception as e:
                print(f"Erreur: {e}")