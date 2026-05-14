from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id='deepseek-ai/DeepSeek-V4-Pro',
    task='conversational'
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

prompt = PromptTemplate(
    template= 'Generate a summary of the {poem} \n',
    input_variables=['poem']
)

loader = TextLoader('./poemForDocLoader.txt', encoding='utf-8')

docs = loader.load()

chain = prompt | model | parser

print(chain.invoke({'poem':docs[0].page_content}))

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)