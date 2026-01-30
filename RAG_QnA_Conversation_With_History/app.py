import os
import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

load_dotenv()

os.environ['LANGCHAIN_API_KEY'] = os.getenv('LANGCHAIN_API_KEY')
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_PROJECT'] = 'RAG QnA Conversation with History'

embeddings = AzureOpenAIEmbeddings(
    api_key = os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    model = os.getenv("AZURE_OPENAI_EMBEDDINGS_MODEL"),
    api_version = "2024-12-01-preview"
)

# Format documents into context string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Get Chat History for a given session ID
def get_session_chat_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state.store:
        st.session_state.store[session_id] = ChatMessageHistory()
    return st.session_state.store[session_id]

# Remove session documents (temp files and retriever)
def remove_session_docs(session_id: str):
    """Delete temporary PDF files and clear retriever for the session."""
    import glob
    # Delete temp files associated with this session
    temp_files = glob.glob(f"./temp_{session_id}_*.pdf")
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
        except Exception as e:
            st.error(f"Error removing {temp_file}: {e}")
    
    # Clear retriever from session state
    if "retriever" in st.session_state:
        del st.session_state.retriever
    
    st.success(f"Removed all documents for session: {session_id}")

# Clear session chat history
def clear_session_history(session_id: str):
    """Clear chat message history for the given session."""
    if session_id in st.session_state.store:
        st.session_state.store[session_id].clear()
        st.success(f"Cleared chat history for session: {session_id}")
    else:
        st.info(f"No chat history found for session: {session_id}")

#Set up Streamlit app
st.title("Conversational RAG with PDF Upload and Chat History")
st.markdown("Upload a PDF document and ask questions about its content.")

# Initialize session state
if 'store' not in st.session_state:
    st.session_state.store = {}

# Sidebar for API key and status
with st.sidebar:
    st.header("Configuration")
    
    # Show LLM status
    if "chat_model" in st.session_state:
        st.success("✓ LLM Initialized")
    else:
        st.warning("⚠ LLM Not Initialized")
    
    api_key = st.text_input("Azure Endpoint API Key", type="password")
    
    if api_key and st.button("Initialize LLM"):
        st.session_state.chat_model = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            azure_deployment=os.getenv("AZURE_OPENAI_LLM_MODEL"),
            api_version="2025-04-01-preview",
            api_key=api_key
        )
        st.rerun()
    
    st.markdown("---")
    st.header("Session Management")
    
    # Session ID input for management buttons
    if "chat_model" in st.session_state:
        # Get current session_id from main area if it exists
        current_session = st.session_state.get("current_session_id", "default_session")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Remove Session Docs"):
                remove_session_docs(current_session)
                st.rerun()
        
        with col2:
            if st.button("Clear Session History"):
                clear_session_history(current_session)
                st.rerun()

# Check if chat_model exists in session state
if "chat_model" in st.session_state:
    chat_model = st.session_state.chat_model
    
    ## Chat Interface
    session_id = st.text_input("Enter a session ID for chat history (e.g., 'session1'):", value="default_session")
    
    # Store current session ID in session state for sidebar buttons
    st.session_state.current_session_id = session_id

    uploaded_files = st.file_uploader("Upload a PDF document", type=["pdf"], accept_multiple_files=True)

    if uploaded_files and st.button("Process Document"):
        documents = []
        for uploaded_file in uploaded_files:
            # OLD APPROACH: Generic temp file naming - not session-specific
            # temp_pdf = f"./temp_{uploaded_file.name}"
            # REASON: Multiple sessions could overwrite each other's temp files
            
            # NEW APPROACH: Session-specific temp file naming
            # Temp files are REQUIRED because PyPDFLoader needs a file path, not in-memory bytes
            temp_pdf = f"./temp_{session_id}_{uploaded_file.name}"
            with open(temp_pdf, "wb") as f:
                f.write(uploaded_file.getvalue())

            loader = PyPDFLoader(temp_pdf)
            docs = loader.load()
            documents.extend(docs)

        #Split and create embeddings for all documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
        splits = text_splitter.split_documents(documents)
        vector_store = Chroma.from_documents(splits, embeddings)
        st.session_state.retriever = vector_store.as_retriever()
        
        st.success(f"Processed {len(uploaded_files)} PDF document(s) successfully!")
    
    # Only show Q&A interface if retriever exists
    if "retriever" in st.session_state:
        retriever = st.session_state.retriever
        
        # Define prompts
        contextualize_q_system_prompt = (
            "Given a chat history and the latest user question "
            "which may refer to previous messages in the chat history, "
            "provide a standalone question that can be answered without the chat history. "
            "Do NOT answer the question, only rephrase it if needed. "
            "Otherwise, return the question as is."
        )

        contextualize_q_prompt = ChatPromptTemplate.from_messages([
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}")
        ])

        # LCEL chain to rephrase question if chat_history exists
        def contextualize_question(inputs):
            if inputs.get("chat_history"):
                chain = contextualize_q_prompt | chat_model | StrOutputParser()
                return chain.invoke(inputs)
            else:
                return inputs["input"]

        # Create history-aware retriever using LCEL
        history_aware_retriever = (
            RunnablePassthrough.assign(
                rephrased_question=RunnableLambda(contextualize_question)
            )
            | RunnableLambda(lambda x: retriever.invoke(x["rephrased_question"]))
        )

        system_prompt = (
            "You are a helpful AI assistant for question-answering tasks. "
            "Use the provided context to answer the question. "
            "If the answer is not contained within the text below, respond with 'I don't know'. "
            "Use maximum of three sentences to answer the question accurately."
            "\n\n"
            "<context>\n{context}\n</context>\n"
        )

        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {input}")
        ])

        # LCEL RAG chain
        rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: format_docs(
                    history_aware_retriever.invoke({
                        "input": x["input"],
                        "chat_history": x.get("chat_history", [])
                    })
                )
            )
            | qa_prompt
            | chat_model
            | StrOutputParser()
        )

        conversational_rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_chat_history,
            input_messages_key="input",
            history_messages_key="chat_history"
        )

        user_input = st.text_input("Your Question:")
        if user_input:
            session_history = get_session_chat_history(session_id)
            with st.spinner("Generating answer..."):
                answer = conversational_rag_chain.invoke(
                    {"input": user_input},
                    config={
                        "configurable": {"session_id": session_id}
                    }
                )
            st.markdown(f"**Answer:** {answer}")
            st.markdown("---")
            with st.expander("Chat History"):
                st.markdown(f"**Session ID:** {session_id}")
                for msg in session_history.messages:
                    st.write(f"**{msg.type}:** {msg.content}")
    else:
        st.info("Please upload and process a PDF document to start asking questions.")
        
else:
    st.warning("Please enter your Azure OpenAI API key to initialize the LLM.")




