from langchain_huggingface import ChatHuggingFace,HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = "conversational"
)
model = ChatHuggingFace(llm=llm)
parser=StrOutputParser()
prompt= PromptTemplate(
    template='Give a breif report on {topic} /n',
    input_variables=['topic']
)
user_input = input("Enter a topic of your choice: ")
chain = prompt | model | parser
result = chain.invoke({'topic': user_input})
print(result)
chain.get_graph().print_ascii()