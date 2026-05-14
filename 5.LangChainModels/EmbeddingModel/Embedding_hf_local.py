from langchain_huggingface import HuggingFaceEmbeddings


embedding = HuggingFaceEmbeddings(model='openai/gpt-oss-20b')

vector = embedding.embed_query("Bangalore is the silicon valley of India") #o/p is a 32 dimensional vector which represents the contextual meaning of the sentence- converts to vector format

print(str(vector))