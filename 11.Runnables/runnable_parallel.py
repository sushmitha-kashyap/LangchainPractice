from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_classic.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableSequence

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    task='conversational'
)
model = ChatHuggingFace(llm=llm)

tweetPrompt = PromptTemplate(
    template = 'Give me a tweet from this {topic}',
    input_variables=['topic']
)

linkdinPrompt = PromptTemplate(
    template = 'Give me a linkdin post from this {topic}',
    input_variables=['topic']
)

parser = StrOutputParser()

parallelChain = RunnableParallel({
    'tweet': RunnableSequence(tweetPrompt,model,parser),
    'linkdin': RunnableSequence(linkdinPrompt,model,parser)
})

result = parallelChain.invoke({'topic':'Unemployment in India'})

print(result)

parallelChain.get_graph().print_ascii()

print(result['tweet'])
print(result['linkdin'])