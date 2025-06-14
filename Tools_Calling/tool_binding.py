from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import requests
from langchain_together import ChatTogether
from dotenv import load_dotenv
load_dotenv()

# tool create

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

print(multiply.invoke({'a':3, 'b':4}))
print(multiply.args)

# tool binding

llm=ChatTogether(model="meta-llama/Llama-3-70b-chat-hf")

new_llm=llm.bind_tools([multiply])

print(new_llm.invoke("how are you").tool_calls)
