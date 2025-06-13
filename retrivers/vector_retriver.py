from pymongo import MongoClient
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
# from langchain_core.retrievers import VectorStoreRetriever
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI,GoogleGenerativeAIEmbeddings
import shutil
import os

# Clean old store
if os.path.exists("mongo_chroma_db_google"):
    shutil.rmtree("mongo_chroma_db_google")
load_dotenv()

# Step 1: Connect to MongoDB and load data
client = MongoClient("mongodb://localhost:27017")
collection = client["syncrous"]["users"]

documents = []
for record in collection.find():
    if "email" in record:
        documents.append(Document(page_content=record["email"], metadata={"_id": str(record["_id"])}))

print(documents)
# Step 2: Generate embeddings
embedding = OpenAIEmbeddings()  # Or use HuggingFaceEmbeddings for free

embedding1 = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# Step 3: Create Chroma vector store
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embedding1,
    persist_directory="mongo_chroma_db_google",  # use a new directory
    collection_name="emails"
)

# Step 4: Get retriever from vector store
retriever = vectorstore.as_retriever()

# Step 5: Use retriever
query = "all gmail stored"
# results = retriever.get_relevant_documents(query)
results = retriever.invoke(query)
# Print results
for doc in results:
    print(f"📧 {doc.page_content} | Metadata: {doc.metadata}")
