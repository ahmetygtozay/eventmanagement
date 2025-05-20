import numpy as np
import faiss
import psycopg2
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_community.embeddings import SentenceTransformerEmbeddings

# --- DB CONNECTION ---
conn = psycopg2.connect(
    dbname="eventdb",
    user="postgres",
    password="Ahmet1212.",
    host="localhost",
    port="5432"
)
cur = conn.cursor()

# --- FETCH EVENTS ---
cur.execute("""
    SELECT id, title, category, tags, location 
    FROM events
""")
rows = cur.fetchall()

event_ids = []
texts = []

for row in rows:
    event_id, title,  category, tags, location = row
    event_ids.append(event_id)

    text_parts = [
        title or "",
        
        f"Category: {category}" if category else "",
        f"Tags: {', '.join(tags) if isinstance(tags, list) else tags}" if tags else "",
        f"Location: {location}"if location else ""
    ]
    full_text = ". ".join(part.strip() for part in text_parts if part).strip()
    texts.append(full_text)

# --- EMBEDDING ---
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
embeddings = embeddings.astype('float32')

# --- NORMALIZE FOR COSINE SIMILARITY ---
faiss.normalize_L2(embeddings)

# --- BUILD RAW FAISS INDEX ---
dimension = embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)  # Inner product used with normalized vectors = cosine
index.add(embeddings)

# --- BUILD LangChain INDEX MANUALLY ---
documents = [
    Document(page_content=text, metadata={"event_id": eid})
    for text, eid in zip(texts, event_ids)
]
docstore = InMemoryDocstore({str(i): doc for i, doc in enumerate(documents)})
index_to_docstore_id = {i: str(i) for i in range(len(documents))}

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

faiss_index = FAISS(
    embedding_function=embedding_model,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id
)

# --- SAVE LangChain-Compatible INDEX ---
faiss_index.save_local("faiss_index_dir")

print(f" Saved {len(texts)} normalized event embeddings to 'faiss_index_dir'.")
