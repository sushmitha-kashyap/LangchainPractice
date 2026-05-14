from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

document = [
    "the addition of 2 and 2 gives 4",
    "the multiplication of 7 and 7 gives 49",
    "the cube of 3 is 27",
    "the division of 10 by 2 gives us 5",
    "subtracting 8 from 20 gives us  12"
]

query = "what is the cube of 3?"

embedding = OpenAIEmbeddings(model='text-embedding-3-large')
doc_embed = embedding.embed_documents(document)
query_embed = embedding.embed_query(query)

score = cosine_similarity([query_embed],doc_embed)[0]

index, highScore = sorted(list(enumerate(score), key = lambda x:x[1])[-1])

print(query)
print(document[index])
print(f"similarity score is {highScore}")