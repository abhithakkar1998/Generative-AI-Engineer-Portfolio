import os
import streamlit as st
from dotenv import load_dotenv
from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
import numpy as np
from time import sleep
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

load_dotenv('/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/.env')
os.environ["NVIDIA_API_KEY"] = os.getenv("NVIDIA_API_KEY")

## Intialize embeddings
def initialize_nvidia_embeddings():
    try:
        nvidia_embeddings = NVIDIAEmbeddings(
            model='nvidia/nv-embed-v1',
            api_key=os.getenv("NVIDIA_API_KEY"),
            trucate='NONE'
        )
        return nvidia_embeddings
    except Exception  as e:
        st.error(f"Error initializing nvidia Embeddings: {e}")
        st.stop()
        return None

##Data Ingestion and Preprocessing
def data_ingestion():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(script_dir, "data")
    
    if not os.path.exists(data_dir):
        raise ValueError(f"Data directory not found at: {data_dir}")
    
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        raise ValueError(f"No PDF files found in: {data_dir}")
    
    loader = PyPDFDirectoryLoader(data_dir)
    documents = loader.load()
    
    if not documents:
        raise ValueError(f"No documents could be loaded from PDFs in: {data_dir}")

    # Text splitting
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=1000)
    split_docs = text_splitter.split_documents(documents)
    return split_docs

## Vector Embedding and Storage
def vector_embedding_and_storage(split_docs, nvidia_embeddings):
    try:
        if not split_docs:
            raise ValueError("No documents to embed.")
        
        vectorstore_faiss = FAISS.from_documents(split_docs, nvidia_embeddings)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(script_dir, "faiss_index")
        
        vectorstore_faiss.save_local(data_dir)
        retriever = vectorstore_faiss.as_retriever(search_kwargs={"k": 5})
        return retriever
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        st.stop()
        return None

## Initialize LLM
def initialize_llm():
    try:
        llm = ChatNVIDIA(
            model="qwen/qwen3.5-397b-a17b",
            api_key=os.getenv("NVIDIA_API_KEY"),
            temperature=0.7,
            top_p=0.9,
            max_completion_tokens=4096,
        )
        return llm
    except Exception  as e:
        st.error(f"Error initializing nvidia LLM: {e}")
        st.stop()
        return None

## RAG Pipeline
def create_rag_pipeline(retriever, llm):
    #Function to join content of all relevant context fetched
    def concat_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    prompt = ChatPromptTemplate.from_template(
    """
    You are an AI Assistant that answer's user queries based on the provided context.
    
    Question:
    {question}

    Context:
    {context}
    """  
    )

    rag_chain = (
        {
            "context": retriever | concat_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )   

    return rag_chain

# # Function to get response from RAG pipeline
# def get_llm_response(rag_chain, query):
#     response = rag_chain.invoke(query)
#     return response


def main():
    st.set_page_config(
        page_title="Document QnA Chat",
        page_icon="📚",
        layout="wide"
    )
    st.title("📚 Document QnA with RAG using AWS nvidia")

    # Sidebar for Vector Store Configuration
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")
        st.subheader("Vector Store Setup")
        
        if st.button("🔄 Create/Update Vector Store", use_container_width=True):
            with st.spinner("Creating/Updating vector store..."):
                try:
                    split_docs = data_ingestion()
                    st.session_state.nvidia_embeddings = initialize_nvidia_embeddings()
                    st.session_state.retriever = vector_embedding_and_storage(split_docs, st.session_state.nvidia_embeddings)
                    st.success("✅ Vector store created/updated successfully!")
                except ValueError as e:
                    st.error(f"❌ Error: {e}")
        
        st.markdown("---")
        st.caption("Click the button above to load and process PDFs from the data folder.")

    # Initialize components
    if 'llm' not in st.session_state:
        st.session_state.llm = initialize_llm()
    
    if 'retriever' not in st.session_state:
        st.warning("⚠️ Please create/update the vector store first by clicking the button in the sidebar.")
        st.stop()

    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    rag_chain = create_rag_pipeline(st.session_state.retriever, st.session_state.llm)

    # Display chat history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")
    
    if user_query:
        # Add user message to history
        st.session_state.chat_history.append({"role": "user", "content": user_query})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_query)
        
        # Generate and display assistant response
        response = ""
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response_placeholder = st.empty()
                sleep(3)  # Small sleep to ensure the placeholder is rendered before streaming starts
            with st.spinner("Generating response..."):
                for txt in rag_chain.stream(user_query):
                    response += txt
                    response_placeholder.markdown(response)
        
        # Add assistant message to history
        st.session_state.chat_history.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()