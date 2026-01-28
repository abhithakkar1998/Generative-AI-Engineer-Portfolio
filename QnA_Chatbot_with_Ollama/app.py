import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith / LangChain tracking: set before importing langchain modules
# """
# ## These are old env settings for langsmith.
# os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
# os.environ['LANGCHAIN_TRACING_V2'] = "true"
# os.environ['LANGCHAIN_PROJECT'] = "QnA Chatbot with Ollama"
# """

# The ols env settings work but we'll use the new ones below.
os.environ['LANGSMITH_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGSMITH_TRACING'] = "true"
os.environ['LANGSMITH_PROJECT'] = "QnA Chatbot with Ollama"

from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Please respond to the user's queries"),
    ("user","Question: {question}")
])

def generate_response(question, llm, max_tokens, temp):
    
    try:
        chat_model = OllamaLLM(model=llm, temperature=temp, max_tokens=max_tokens)

        output_parser = StrOutputParser()

        chain = prompt | chat_model | output_parser
        response = chain.invoke({"question": question})
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None
    
## Title of the app
st.title("Q&A Chatbot with Ollama")

## Sidebar for settings
st.sidebar.title("Settings")

## Dropdown for model selection
llm_dropdown = st.sidebar.selectbox(
    "Select Ollama Model",
    options=["gemma:2b", "phi3:latest", "falcon3:1b"]
)

## Adjustable response parameters
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=4000, value=1000, step=100)
temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.5, step=0.1) 

## Main Interface for user input
st.write("Ask any question and get answers powered by OpenAI GPT Models!")
user_input = st.text_input("You:")

if user_input and st.button("Send"):
    with st.spinner("Generating response..."):
        answer = generate_response(user_input, llm_dropdown, max_tokens, temperature)
    if answer:
        st.markdown(f"**Bot:** {answer}")
    else:
        st.error("Failed to generate a response.")
else:
    st.write("Please enter a question to get started.")