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
prompt1= PromptTemplate(
    template='Give a detailed report on {topic} /n',
    input_variables=['topic']
)
prompt2= PromptTemplate(
    template='Give 5 pointer summary from the following {text} /n',
    input_variables=['text']
)
chain = prompt1 | model | parser | prompt2 | model | parser
result = chain.invoke({'topic':'hugs'})
print(result)
chain.get_graph().print_ascii()