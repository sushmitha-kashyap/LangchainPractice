from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter

loader = TextLoader('./poemForDocLoader.txt', encoding='utf-8')

doc = loader.load()

text_split = CharacterTextSplitter(
    chunk_size = 190,
    chunk_overlap = 0,
    separator=''
)

result = text_split.split_documents(doc)

print(result[0].page_content) #displays content of the first chunk