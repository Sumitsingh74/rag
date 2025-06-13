from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from  dotenv import load_dotenv
from sympy.physics.units import temperature

load_dotenv()
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableParallel,RunnableSequence

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()
model1= ChatTogether(model="meta-llama/Llama-3-70b-chat-hf",temperature=1.2)

# model2 = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",
#     # model_kwargs={"timeout": 30},  # timeout in seconds
#     temperature=1.8
# )
model3= ChatTogether(model='lgai/exaone-3-5-32b-instruct',temperature=1.5)

parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Write a joke about{topic}',
    input_variables=['topic']
)

chain = RunnableParallel({
    'beta':RunnableSequence(prompt1, model1, parser),
    'gama':RunnableSequence(prompt2,model3,parser)
})

# print(chain.invoke({'topic':'AI'})['beta'])
# print()
print(chain.invoke({'topic':'AI'})['gama'])
