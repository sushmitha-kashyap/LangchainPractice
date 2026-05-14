# this code is for running models locally,takes a lot of time
from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
# below two lines are just for downloading the model in that particular path, if not given model will be downloaded in the defaulf location
import os
os.environ['HF_HOME'] = 'D:/huggingface_cache'

llm = HuggingFacePipeline.from_model_id(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    task="conversational",
    pipeline_kwargs=dict(
        temperature = 0.5,
        max_new_tokens = 100
    )
)

model = ChatHuggingFace(llm = llm)

result = model.invoke("what is the capital of usa?")

print(result.content)