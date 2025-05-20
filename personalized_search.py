import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)

import numpy as np
import psycopg2
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from sentence_transformers import SentenceTransformer

# ----- DB CONNECTION -----
conn = psycopg2.connect(
    dbname='eventdb',
    user='postgres',
    password='Ahmet1212.',
    host='localhost',
    port='5432'
)
cur = conn.cursor()

# ----- LOAD EMBEDDING MODEL -----
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# ----- LOAD FAISS INDEX FROM LANGCHAIN -----
faiss_index = FAISS.load_local("faiss_index_dir", embedding_model, allow_dangerous_deserialization=True)

# ----- EXTRACT EVENT METADATA -----
event_metadatas = [doc.metadata for doc in faiss_index.docstore._dict.values()]
event_ids = [meta["event_id"] for meta in event_metadatas]

# ----- FETCH EVENT TITLES FROM DB -----
placeholders = ','.join(['%s'] * len(event_ids))
cur.execute(f"SELECT id, title FROM events WHERE id IN ({placeholders})", event_ids)
event_id_title_map = dict(cur.fetchall())

# ----- LOAD USER PROFILE VECTORS -----
user_profile_vectors = np.load("user_profile_vectors.npy")
user_profile_ids = np.load("user_profile_ids.npy")

# ----- NORMALIZE FUNCTION -----
def normalize(vectors):
    from faiss import normalize_L2
    normalize_L2(vectors)
    return vectors

# ----- GET SPECIFIC USER VECTOR -----
def get_user_vector(user_id):
    if user_id in user_profile_ids:
        idx = np.where(user_profile_ids == user_id)[0][0]
        return user_profile_vectors[idx:idx+1].astype('float32')
    return None

# ----- MAIN SEARCH FUNCTION -----
def recommend_events(user_id, query_text, k=5, alpha=0.7):
    print(f'🔍 Query: "{query_text}" | 👤 User ID: {user_id}')

    user_vec = get_user_vector(user_id)
    query_vec = sentence_model.encode([query_text], convert_to_numpy=True).astype('float32')
    normalize(query_vec)

    if user_vec is not None:
        combined_vec = alpha * user_vec + (1 - alpha) * query_vec
        normalize(combined_vec)
    else:
        print("⚠️ No user profile found. Using query only.")
        combined_vec = query_vec

    distances, indices = faiss_index.index.search(combined_vec, k)

    print(f"\n Top {k} Recommendations:")
    retrieved = []

    for rank, idx in enumerate(indices[0]):
        meta = event_metadatas[idx]
        event_id = meta["event_id"]
        title = event_id_title_map.get(event_id, "Unknown Title")
        score = distances[0][rank]

        cur.execute("SELECT description, category, tags FROM events WHERE id = %s", (event_id,))
        desc, cat, tags = cur.fetchone()

        retrieved.append({
            "event_id": event_id,
            "title": title,
            "description": desc,
            "category": cat,
            "tags": tags
        })

        print(f"{rank+1}. Event ID {event_id} | Title: {title} | Similarity: {score:.4f}")

    return retrieved

# ----- BUILD RAG PROMPT -----
def build_rag_prompt(query, retrieved_events):
    context = ""
    for i, event in enumerate(retrieved_events):
        title = event.get("title", "Unknown Title")
        description = event.get("description", "No description available")
        category = event.get("category", "N/A")
        tags = event.get("tags", "N/A")
        context += f"{i+1}. {title}: {description}. Category: {category}. Tags: {tags}\n"

    prompt = f"""
You are a helpful assistant that analyzes event data to give personalized insights or summaries based on a user's query.

Below are some relevant events retrieved for the query: "{query}"

{context}

Please provide a brief insight or recommendation to the user based on these events. 
Do **not** repeat the list. Focus on what trends, patterns, or suggestions might be helpful.

[INSIGHT]
""".strip()
    return prompt


# ----- EXAMPLE USAGE -----
if __name__ == "__main__":
    user_id = 90
    query_text = "rock concert"

    retrieved_events = recommend_events(user_id, query_text)
    prompt = build_rag_prompt(query_text, retrieved_events)

    print("\n RAG Prompt to feed into LLM:\n")
    print(prompt)
