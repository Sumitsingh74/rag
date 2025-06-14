from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_together import ChatTogether
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Step 1: Define your tool
@tool
def multiply(a: int, b: int) -> int:
    """Given 2 numbers a and b, this tool returns their product"""
    return a * b

# Step 2: Preview tool behavior
print("Tool output preview:", multiply.invoke({'a': 3, 'b': 4}))
print("Tool input schema:", multiply.args)

# Step 3: Set up the model (TogetherAI or OpenAI)
# Option A: Use Together
llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf")
# Option B: Uncomment below for OpenAI (more reliable tool use)
# llm = ChatOpenAI(model="gpt-3.5-turbo")

# Step 4: Bind tool to the model
new_llm = llm.bind_tools([multiply])

# Step 5: Create input prompt
query = HumanMessage("Use the multiply tool to multiply 3 and 1000")
messages = [query]

# Step 6: Get model response and check for tool call
result = new_llm.invoke(messages)
messages.append(result)
print(messages)

# Step 7: If tool was called, execute and return result
if result.tool_calls:
    for call in result.tool_calls:
        tool_output = multiply.invoke(call.args)
        messages.append(ToolMessage(tool_output, tool_call_id=call.id))
    # Get final model answer using tool output
    final_response = new_llm.invoke(messages)
    print("✅ Final answer:", final_response.content)
else:
    print("❌ No tool was invoked.")
    print("Model response:", result.content)
