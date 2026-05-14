from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import streamlit as st

load_dotenv()

model = ChatOpenAI()

st.header('Reasearch Model')

static_prompt = st.text_input('Summarize the attention is all you need paper in 5 lines')

if st.button('Summarise'):
    result = model.invoke(static_prompt)
    st.write(result.content)