import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from langchain_ollama import OllamaLLM
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

# Page configuration
st.set_page_config(
    page_title="Code Chat Assistant",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .stMessageInput > div > div > input {
        font-size: 1rem;
    }
    .stChatMessage {
        padding: 1rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_name = st.selectbox(
        "Select Model",
        ["qwen2.5-coder:7b", "codellama", "ministral-3:latest", "gemma:2b"],
        help="Choose an available Ollama model"
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Lower values = more deterministic, Higher values = more creative"
    )
    
    # Max tokens slider
    max_tokens = st.slider(
        "Max Tokens",
        min_value=100,
        max_value=4000,
        value=1000,
        step=100,
        help="Maximum length of the response"
    )
    
    # Top-p slider
    top_p = st.slider(
        "Top P",
        min_value=0.0,
        max_value=1.0,
        value=0.9,
        step=0.1,
        help="Nucleus sampling parameter"
    )
    
    st.divider()
    
    # Clear chat history button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.success("Chat history cleared!")

# Hardcoded system prompt
SYSTEM_PROMPT = """You are a CODING-ONLY assistant. NOTHING ELSE.

**YOUR ROLE:** Answer ONLY questions about programming, code, software development, and technical implementation.

**EXAMPLES TO ANSWER:**
"How do I write a recursive function?"
"Explain the difference between arrays and linked lists"
"What's the best way to debug Python code?"

**YOU MUST REFUSE:**
- History questions (e.g., "Who was Newton?")
- Science questions (e.g., "What is gravity?")
- General knowledge (e.g., "What's the capital of France?")
- Philosophy, politics, entertainment, weather, etc.

**REFUSAL TEMPLATE:**
If the question is NOT about coding/programming, respond EXACTLY with:
"I can only help with coding and programming questions. Please ask about code, algorithms, debugging, or software development."

If unsure whether a question is about coding: REFUSE IT."""

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_config" not in st.session_state:
    st.session_state.current_config = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }

# Generate response function
def generate_response(question, llm_model, temp, max_tok):
    """Generate response using LCEL chain with StrOutputParser"""
    try:
        # Initialize Ollama LLM
        llm = OllamaLLM(
            model=llm_model,
            temperature=temp,
            max_tokens=max_tok,
        )
        
        # Output parser
        output_parser = StrOutputParser()
        
        # Create prompt template with chat history
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        # Build the chain using LCEL: prompt | llm | parser
        chain = prompt_template | llm | output_parser
        
        # Build the message list for context (last 10 messages)
        messages = []
        for msg in st.session_state.messages[-10:]:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        
        # Invoke the chain
        response = chain.invoke({
            "chat_history": messages,
            "question": question
        })
        
        return response
    
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

# Main chat interface
st.title("💻 Code Chat Assistant")
st.markdown("*Powered by LangChain + Ollama*")

# Check if we need to update config
config_changed = (
    st.session_state.current_config["model"] != model_name or
    st.session_state.current_config["temperature"] != temperature or
    st.session_state.current_config["max_tokens"] != max_tokens or
    st.session_state.current_config["top_p"] != top_p
)

if config_changed:
    st.session_state.current_config = {
        "model": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p
    }

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about coding..."):
    # Add user message to history
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("Generating response..."):
            response = generate_response(
                prompt,
                model_name,
                temperature,
                max_tokens
            )
        
        if response:
            message_placeholder.markdown(response)
            # Add assistant message to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response
            })
        else:
            error_msg = "❌ Failed to generate response. Please check your Ollama connection."
            message_placeholder.markdown(error_msg)

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: #888; font-size: 0.85rem;'>
    Made with ❤️ using Streamlit + LangChain + Ollama
    </div>
    """,
    unsafe_allow_html=True
)
