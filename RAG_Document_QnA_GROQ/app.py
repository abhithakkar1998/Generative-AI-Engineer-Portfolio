import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

#load GROQ API key
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')
groq_api_key = os.getenv("GROQ_API_KEY")

chat_model = ChatGroq(api_key=groq_api_key, model="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the question based on the provided context only. 
    If the answer is not contained within the text below, respond with "I don't know".
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Question: {question}
    """
)

def create_vector_embedding():
    if "vectors" not in st.session_state:
        #st.session_state.embeddings = OllamaEmbeddings(model="gemma:2b")
        st.session_state.embeddings = AzureOpenAIEmbeddings(
            api_key = os.getenv("AZURE_OPENAI_API_KEY"),
            azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
            model = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL"),
            api_version ="2024-12-01-preview",
            dimensions = 1024
        )
        st.session_state.loader = PyPDFDirectoryLoader("research_papers")
        st.session_state.docs= st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.split_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
        st.session_state.vectors = FAISS.from_documents(st.session_state.split_docs, st.session_state.embeddings)


user_question = st.text_input("Enter your question about the research papers:")

if st.button("Document Embedding"):
    with st.spinner("Creating document embeddings..."):
        create_vector_embedding()
    st.success("Document embeddings created successfully!")

import time

if user_question and st.button("Get Answer"):
    if "vectors" not in st.session_state:
        st.error("Please create document embeddings first.")
    else:
        with st.spinner("Generating answer..."):
            retriever = st.session_state.vectors.as_retriever(search_type="similarity", search_kwargs={"k":3})
            relevant_docs = retriever.invoke(user_question)
            context = "\n".join([doc.page_content for doc in relevant_docs])
            chain = prompt | chat_model | StrOutputParser()

            start = time.process_time()
            answer = chain.invoke({"context": context, "question": user_question})
            stop = time.process_time()
            st.success(f"Answer generated in {stop - start} seconds!")
        st.markdown(f"**Answer:** {answer}")

        with st.expander("Show Retrieved Context"):
            for i, doc in enumerate(relevant_docs):
                st.markdown(f"**Document {i+1}:**")
                st.write(doc.page_content)
                st.markdown("---")