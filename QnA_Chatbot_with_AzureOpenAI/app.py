import streamlit as st
import openai
import os
from dotenv import load_dotenv

load_dotenv()

# LangSmith / LangChain tracking: set before importing langchain modules
os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'QnA Chatbot with Azure OpenAI'

from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

#Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system","You are a helpful assistant. Please respond to the user's queries"),
    ("user","Question: {question}")
])

def generate_response(question, api_key, llm, max_tokens):
    
    try:
        chat_model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=llm,
            api_version="2025-04-01-preview",
            api_key=api_key,
            max_tokens=max_tokens,
        )

        output_parser = StrOutputParser()

        chain = prompt | chat_model | output_parser
        response = chain.invoke({"question": question})
        return response
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None
    
## Title of the app
st.title("Q&A Chatbot with Azure OpenAI")

## Sidebar for settings
st.sidebar.title("Settings")
api_key = st.sidebar.text_input("Azure OpenAI API Key", type="password")

## Dropdown for model selection
llm_dropdown = st.sidebar.selectbox(
    "Select Azure OpenAI Deployment",
    options=["gpt-5-chat", "gpt-5.1-chat", "gpt-5.2-chat"]
)

## Adjustable response parameters
max_tokens = st.sidebar.slider("Max Tokens", min_value=100, max_value=4000, value=1000, step=100)
#temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, step=0.1) 
#Temperature is not supported for OpenAI chat models from GPT-5 onwards. Thus commenting out.

## Main Interface for user input
st.write("Ask any question and get answers powered by OpenAI GPT Models!")
user_input = st.text_input("You:")

if user_input and api_key:
    with st.spinner("Generating response..."):
        answer = generate_response(user_input, api_key, llm_dropdown, max_tokens)
    if answer:
        st.markdown(f"**Bot:** {answer}")
    else:
        st.error("Failed to generate a response.")
elif not api_key:
    st.write("Please enter your Azure OpenAI API Key in the sidebar to proceed.")
else:
    st.write("Please enter a question to get started.")