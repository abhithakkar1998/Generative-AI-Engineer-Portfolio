from fastapi import FastAPI
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langserve import add_routes

import os
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm_groq = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

generic_template = "Translate the following English text to {language}."

promptTemplate = ChatPromptTemplate.from_messages([
    ("system", generic_template),
    ("user", "{text}")
])

parser = StrOutputParser()

chain = promptTemplate | llm_groq | parser

##App Definition
app = FastAPI(title="LangChain Server", version="0.1", description="LangChain server with FastAPI")

add_routes(
    app,
    chain,
    path="/chain"
)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8003)