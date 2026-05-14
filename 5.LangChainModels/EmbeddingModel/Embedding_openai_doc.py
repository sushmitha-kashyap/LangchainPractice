from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

document = [
    "Bangalore is the silicon valley of India",
    "Take this as the second statement of the document",
    "Pecock is the national bird of India"
]

vector = embedding.embed_documents(document)
print(str(vector))