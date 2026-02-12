import streamlit as st

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import tool
import os
from dotenv import load_dotenv
load_dotenv("/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/.env")

st.set_page_config(page_title="Text to Math Problem Solver", page_icon=":pencil2:")
st.title("Text to Math Problem Solver") 
st.write("Enter a math problem in natural language, and I'll solve it for you!")

llm = ChatOllama(model='ministral-3:latest')

wikipedia_wrapper = WikipediaAPIWrapper()

@tool("wikipedia_search")
def wikipedia_search(query: str) -> str:
    """Search Wikipedia for a query and return the summary."""
    return wikipedia_wrapper.run(query)

@tool("math_solver")
def math_solver(query: str) -> str:
    """Solve a math problem and return the solution."""
    prompt = f"Solve the following math problem step by step:\n\n {query}"
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))

@tool("reasoning_tool")
def reasoning_tool(query: str) -> str:
    """Handle logical/reasoning questions"""
    prompt = (
        "You are a reasoning assistant. Use logical reasoning to answer the following question:\n\n"
        f"{query}\n\n"
    )
    response = llm.invoke(prompt)
    return getattr(response, "content", str(response))

# The Ollama LLM wrapper used here doesn't provide LangChain's `bind_tools` API
# (which `create_agent` expects). Instead of relying on automatic tool binding,
# use a simple router to call the appropriate tool function directly.

def handle_question(question: str) -> str:
    q = question.lower()
    math_keywords = ["solve", "calculate", "integral", "derivative", "sum", "equation", "compute"]
    wiki_keywords = ["wikipedia", "who is", "what is", "when", "where", "history", "wiki", "tell me about"]
    reasoning_keywords = ["reason", "prove", "why", "how", "logical", "puzzle"]

    # Heuristic: arithmetic expressions or explicit solve requests -> math_solver
    if any(k in q for k in math_keywords) or (any(ch.isdigit() for ch in q) and any(op in q for op in "+-*/")):
        try:
            return math_solver(question)
        except Exception:
            pass

    # Wikipedia-style queries
    if any(k in q for k in wiki_keywords):
        try:
            return wikipedia_search(question)
        except Exception:
            pass

    # Reasoning / logical queries
    if any(k in q for k in reasoning_keywords):
        try:
            return reasoning_tool(question)
        except Exception:
            pass

    # Fallback: direct LLM response
    try:
        response = llm.invoke(question)
        return getattr(response, "content", str(response))
    except Exception as e:
        return f"Model error: {e}"

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role":"assistant", "content":"Hello! Ask me a math problem, a general question, or a reasoning question, and I'll do my best to help you!"}
    ]  

for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question = st.text_area("Enter your question here:", key="input")

if st.button("Submit"):
    if question.strip() == "" or question.strip() == None or question.strip() == " ":
        st.warning("Please enter a question before submitting.")
    else:
        st.session_state["messages"].append({"role": "user", "content": question})
        st.chat_message("user").write(question)

        with st.spinner("Thinking..."):
            response_text = handle_question(question)
            st.session_state["messages"].append({"role": "assistant", "content": response_text})
            st.chat_message("assistant").write(response_text)
