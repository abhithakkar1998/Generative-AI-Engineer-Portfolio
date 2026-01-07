import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# for Langsmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY") 
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

##Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Please respond to the question asked"),
        ("user", "Question: {question}"),
    ]
)

## streamlit framework
st.title("Langchain Demo with Ollama: Gemma-2b")
input_text = st.text_input("Enter your question here:")

## Ollama llm initialization
llm = OllamaLLM(model="gemma:2b", temperature=0.7)
output_parser = StrOutputParser()

app_chain = prompt | llm | output_parser

if st.button("Get Answer"):
    if input_text:
        response = app_chain.invoke({"question": input_text})
        st.write("Answer:", response)
    else:
        st.write("Please enter a question.")
