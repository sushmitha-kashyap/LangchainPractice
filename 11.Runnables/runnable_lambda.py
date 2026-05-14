from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda,RunnablePassthrough,RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    task='conversational'
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

joke_chain = prompt | model | parser

def joke_word_count(text):
 return len(text.split())

parallel_chain = RunnableParallel({
 'joke': RunnablePassthrough(),
 'jokeLength': RunnableLambda(joke_word_count)
})

final_chain = RunnableSequence(joke_word_count,parallel_chain)

result = final_chain.invoke({'topic':'AI'})

print(result)

final_result = """{} \n word count - {}""".format(result['joke'], result['jokeLength'])

print(final_result)

final_chain.get_graph().print_ascii()