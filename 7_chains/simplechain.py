from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt2=PromptTemplate(
    template='Generate 5 important points from {text}',
    input_variables=['text'],
    validate_template=True
)

prompt1=PromptTemplate(
    template='Generate detail explanation of {topic}',
    input_variables=['topic'],
    validate_template=True
)

model=ChatOpenAI()
parser=StrOutputParser()

chain =prompt1 | model | parser | prompt2 | model | parser

result = chain.invoke({'topic':'cricket'})
print(result)

chain.get_graph().print_ascii()