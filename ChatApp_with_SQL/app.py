import streamlit as st
from pathlib import Path
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain_openai import AzureChatOpenAI
from langchain.agents import create_agent
from sqlalchemy import create_engine
import sqlite3
import os
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv("/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/.env")

st.set_page_config(page_title="LangChain: Chat with SQL DB", page_icon=":robot:")
st.title("LangChain: Chat with SQL DB")

LOCALDB = "USE_LOCALDB"
MYSQL = "USE_MYSQL"

radio_opt = ["Use SQLite Local DB - student.db", "Connect to your SQL Database (MySQL)"]
selected_opt = st.sidebar.radio("Select Database Option", radio_opt)

if radio_opt.index(selected_opt) == 0:
    db_uri = LOCALDB
else:
    db_uri = MYSQL
    mysql_user = st.sidebar.text_input("MySQL User", value="root")
    mysql_password = st.sidebar.text_input("MySQL Password", value="password", type="password")
    mysql_host = st.sidebar.text_input("MySQL Host", value="localhost")
    mysql_port = st.sidebar.text_input("MySQL Port", value="3306")
    mysql_db = st.sidebar.text_input("MySQL Database Name")

api_key = os.getenv("AZURE_OPENAI_API_KEY", "")

if not api_key:
    st.warning("Please set your Azure OpenAI API key in the .env file.")
    st.stop()

chat_model = AzureChatOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
    api_key=api_key,
    api_version="2025-04-01-preview",
    max_tokens=1024
)

@st.cache_resource(ttl="2h")
def configure_db(db_uri, mysql_user=None, mysql_password=None, mysql_host=None, mysql_port=None, mysql_db=None):
    if db_uri == LOCALDB:
        db_path = str((Path(__file__).parent / "student.db").absolute())
        return SQLDatabase.from_uri(f"sqlite:///{db_path}")
    elif db_uri == MYSQL:
        if not all([mysql_user, mysql_password, mysql_host, mysql_port, mysql_db]):
            st.error("Please provide all MySQL connection details.")
            st.stop()
        # URL encode password to handle special characters
        encoded_password = quote_plus(mysql_password)
        return SQLDatabase.from_uri(f"mysql+mysqlconnector://{mysql_user}:{encoded_password}@{mysql_host}:{mysql_port}/{mysql_db}")
        
        
if db_uri == MYSQL:
    if not all([mysql_user, mysql_password, mysql_host, mysql_port, mysql_db]):
        st.warning("Please provide all MySQL connection details in the sidebar.")
        st.stop()
    db = configure_db(db_uri, mysql_user, mysql_password, mysql_host, mysql_port, mysql_db)
else:
    db = configure_db(db_uri)


# Create the toolkit
toolkit = SQLDatabaseToolkit(db=db, llm=chat_model)
agent = create_agent(
    model=chat_model,
    tools=toolkit.get_tools(),
    system_prompt="You are a helpful assistant who can answer questions about the database.",
)

if "messages" not in st.session_state or st.sidebar.button("Clear Conversation"):
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I am your assistant who can help you queries referring to SQL Database. How can I help you today?"}
    ]

for msg in st.session_state["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

user_query = st.chat_input(placeholder="Type your query here...")

if user_query:
    st.session_state["messages"].append({"role": "user", "content": user_query})
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.invoke(
            {"messages": [{"role": "user", "content": user_query}]},
            callbacks=[streamlit_callback]
        )
        assistant_response = response["messages"][-1].content
        st.write(assistant_response)
        st.session_state["messages"].append({"role": "assistant", "content": assistant_response})