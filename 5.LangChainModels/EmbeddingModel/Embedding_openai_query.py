from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=32)

result = embedding.embed_query("Bangalore is the silicon valley of India") #o/p is a 32 dimensional vector which represents the contextual meaning of the sentence- converts to vector format

print(str(result))