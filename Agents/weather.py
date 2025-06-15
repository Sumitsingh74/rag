from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
import requests

from langchain_community.tools import DuckDuckGoSearchRun

search_tool = DuckDuckGoSearchRun()

# from langchain_community.tools.tavily_search import TavilySearchResults

# search_tool = TavilySearchResults(k=3)  # returns top 3 results

@tool
def get_weather_data(city: str) -> str:
  """
  This function fetches the current weather data for a given city
  """
  url = f'https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}'

  response = requests.get(url)

  return response.json()

from langchain_together import ChatTogether
# llm = ChatOpenAI()
llm = ChatTogether(model="meta-llama/Llama-3-70b-chat-hf")
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub
# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=llm,
    tools=[search_tool, get_weather_data],
    prompt=prompt
)
# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent,
    tools=[search_tool, get_weather_data],
    verbose=True
)
# Step 5: Invoke
response = agent_executor.invoke({"input": "Find the capital of Madhya Pradesh, then find it's current weather condition"})
print(response)
response = agent_executor.invoke({"input": "get me good questions on operating system from exam"})
print(response)
print(response['output'])
