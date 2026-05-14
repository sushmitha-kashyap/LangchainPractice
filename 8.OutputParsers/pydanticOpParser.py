from langchain_huggingface import HuggingFaceEndpoint,ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel,Field
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "Qwen/Qwen2.5-7B-Instruct",
    task = "conversational"
)
model = ChatHuggingFace(llm=llm)

class Topic(BaseModel):
    name: str = Field(description='Name of the person'),
    age: int = Field(description='age of the person'),
    city: str = Field(description='city the person lives in')

parser = PydanticOutputParser(pydantic_object=Topic)

template = PromptTemplate(
    template='Generate the name, age and city of a fictional {place} person \n {format_instruction}',
    input_variable=['place'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

chain = template | model | parser

result = chain.invoke({'place':'indian'})

print(result)
