from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_together import ChatTogether
from  dotenv import load_dotenv
load_dotenv()
from  langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence

prompt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

model = ChatOpenAI()
model1= ChatTogether(model="meta-llama/Llama-3-70b-chat-hf",temperature=1.2)

# model2 = ChatGoogleGenerativeAI(
#     model="gemini-1.5-flash",  # ✅ Free-tier supported
#     temperature=0.7
# )
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

chain = RunnableSequence(prompt1, model1, parser,prompt2,model,parser)

print(chain.invoke({'topic':'AI'}))