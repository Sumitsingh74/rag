import os
from dotenv import load_dotenv
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 1) Load API key
load_dotenv()

# 2) Define your student records
students = [
    {"name": "Alice", "email": "alice@example.com", "marks": 92, "date": "2025-06-13", "subject": "Math"},
    {"name": "Bob",   "email": "bob@example.com",   "marks": 85, "date": "2025-06-13", "subject": "Math"},
    {"name": "Cara",  "email": "cara@example.com",  "marks": 78, "date": "2025-06-14", "subject": "Science"},
    {"name": "Dave",  "email": "dave@example.com",  "marks": 88, "date": "2025-06-13", "subject": "History"},
    {"name": "Eve",   "email": "eve@example.com",   "marks": 91, "date": "2025-06-14", "subject": "Math"},
]

# 3) Convert to LangChain Documents (text + metadata)
documents = []
for s in students:
    content = (
        f"Name: {s['name']}\n"
        f"Email: {s['email']}\n"
        f"Marks: {s['marks']}\n"
        f"Date: {s['date']}\n"
        f"Subject: {s['subject']}"
    )
    documents.append(Document(page_content=content, metadata=s))

# 4) Create (or load) a Chroma vector store
vectorstore = FAISS.from_documents(
    documents=documents,
    embedding=OpenAIEmbeddings()
)

# 5) Specify the date & subject to filter by
target_date = "2025-06-14"
target_subject = "Science"

# 6) Retrieve matching records via semantic search + metadata filter
#    (using the subject as the “query” and filtering by both fields)
results = vectorstore.similarity_search(
    query=target_subject,
    k=10,
    filter={  # top‐level “where” must be one operator
        "$and": [
            {"date":   {"$eq": target_date}},
            {"subject":{"$eq": target_subject}}
        ]
    }
)


# 7) Print out full details
print(f"\nStudents on {target_date} for {target_subject}:\n")
for doc in results:
    print(doc.page_content)
    print("Metadata →", doc.metadata)
    print("-" * 40)
