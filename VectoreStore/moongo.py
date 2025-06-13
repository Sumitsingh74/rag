from pymongo import MongoClient
from langchain.schema import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Connect to MongoDB
client = MongoClient("mongodb://localhost:27017")
collection = client["syncrous"]["users"]

# Load documents and convert to LangChain Document format
documents = []
for record in collection.find():
    if "email" in record:
        documents.append(Document(page_content=record["email"], metadata={"_id": str(record["_id"])}))

# Just print and verify
for document in documents:
    print(document)

# Extract just the email text for prompting
email_list = [doc.page_content for doc in documents]

# Prompt to sort
prompt = PromptTemplate(
    template="Arrange the following emails in ascending order:\n{docs}",
    input_variables=["docs"]
)

model = ChatOpenAI()
model2 = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",  # ✅ Free-tier supported
    temperature=0.7
)
parser = StrOutputParser()

chain = prompt | model2 | parser

# Run the chain
result = chain.invoke({'docs': ", ".join(email_list)})
print(result)
