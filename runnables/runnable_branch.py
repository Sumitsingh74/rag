from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableSequence, RunnableLambda, RunnablePassthrough, RunnableParallel,RunnableBranch
from langchain_together import ChatTogether

load_dotenv()

prompt1 = PromptTemplate(
    template='Write a detailed report on {topic}',
    input_variables=['topic']
)

prompt2 = PromptTemplate(
    template='Summarize the following text \n {text}',
    input_variables=['text']
)
model = ChatOpenAI()
model1= ChatTogether(model="meta-llama/Llama-3-70b-chat-hf",temperature=1.2)
model2= ChatTogether(model='lgai/exaone-3-5-32b-instruct',temperature=1.5)


parser = StrOutputParser()

joke_gen_chain = RunnableSequence(prompt1, model1, parser)

branch_chain = RunnableBranch(
    (lambda x: len(x.split())>300, prompt2 | model2 | parser),
    RunnablePassthrough()
)


final_chain = RunnableSequence(joke_gen_chain, branch_chain)

result = final_chain.invoke({'topic': 'AI'})
print(result)
