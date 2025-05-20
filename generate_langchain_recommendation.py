from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

faiss_index = FAISS.load_local(
    "faiss_index_dir",
    embedding_model,
    allow_dangerous_deserialization=True
)

llm = ChatOpenAI(model_name="gpt-4", temperature=0.7)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=faiss_index.as_retriever(search_kwargs={"k": 5}),
)

query = "Find me some rock concerts happening this weekend in Berlin."

response = qa_chain.run(query)
print(response)
