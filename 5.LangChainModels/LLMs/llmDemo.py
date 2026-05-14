# you do not have openAI api key(it is paid), so this code won't work simply practice

from langchain_openai import Openai
from dotenv import load_dotenv

load_dotenv()

llm = Openai(model = "gpt-3.5-turbo-instruct")

result = llm.invoke("what is capital of india?")

print(result)
