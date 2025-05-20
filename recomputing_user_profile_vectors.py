import numpy as np
import psycopg2
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
import warnings
from langchain_core._api.deprecation import LangChainDeprecationWarning
warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)


def normalize(vectors):
    faiss.normalize_L2(vectors)
    return vectors

# Connect to DB
conn = psycopg2.connect(
    dbname="eventdb",
    user="postgres",
    password="Ahmet1212.",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# Load saved FAISS index and embedding model
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.load_local("faiss_index_dir", embedding_model, allow_dangerous_deserialization=True)

# Extract event metadata and embeddings from FAISS index
event_metadatas = faiss_index.docstore._dict  # Correct way to access metadata
event_embeddings = faiss_index.index.reconstruct_n(0, faiss_index.index.ntotal)

# Build event_id to embedding vector map
event_id_to_embedding = {}
for i, (key, doc) in enumerate(event_metadatas.items()):
    event_id = doc.metadata['event_id']
    event_id_to_embedding[event_id] = event_embeddings[i]


# Fetch user-event interaction history from DB
cur.execute("""
    SELECT user_id, event_id
    FROM user_event_history
""")
rows = cur.fetchall()

from collections import defaultdict
user_event_embeddings = defaultdict(list)

for user_id, event_id in rows:
    embedding = event_id_to_embedding.get(event_id)
    if embedding is not None:
        user_event_embeddings[user_id].append(embedding)

user_ids = []
user_profile_vectors = []

for user_id, embeddings in user_event_embeddings.items():
    if embeddings:
        vectors = np.vstack(embeddings)
        profile_vector = np.mean(vectors, axis=0, keepdims=True)
        profile_vector = normalize(profile_vector)[0]
        user_ids.append(user_id)
        user_profile_vectors.append(profile_vector)

user_profile_vectors = np.vstack(user_profile_vectors)
user_ids = np.array(user_ids)

# Save user profile vectors and user ids for later use
np.save("user_profile_vectors.npy", user_profile_vectors)
np.save("user_profile_ids.npy", user_ids)

print(f"Recomputed and saved profile vectors for {len(user_ids)} users.")
