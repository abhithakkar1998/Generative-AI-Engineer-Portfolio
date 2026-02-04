import os
from dotenv import load_dotenv
load_dotenv("/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/.env")

import validators
import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader

st.set_page_config(page_title="Text Summarizer from YouTube Video or Website", page_icon="📝")
st.title("📝 Text Summarizer from YouTube Video or Website")
st.subheader("Summarize text content from a YouTube video or a website URL using LangChain and Azure OpenAI")

with st.sidebar:
    st.header("Input Options")
    open_ai_api_key = st.text_input("Enter your Azure OpenAI API Key", type="password")
    if open_ai_api_key and st.button("Initialize LLM"):
        if "chat_model" not in st.session_state:
            chat_model = AzureChatOpenAI(
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                azure_deployment="gpt-5.2-chat",
                api_version="2025-04-01-preview",
                api_key=open_ai_api_key,
                max_tokens=20000,
            )
            st.session_state.chat_model = chat_model
            st.success("✅ LLM initialized successfully!")
        else:
            st.info("LLM is already initialized.")
    elif not open_ai_api_key and st.button("Initialize LLM"):
        st.session_state.chat_model = None
        st.error("Please enter your Azure OpenAI API Key to initialize the LLM.")


url = st.text_input("Enter YouTube Video URL or Website URL.", label_visibility="collapsed")

map_prompt_template = PromptTemplate.from_template(
    """Summarize the piece of text below:
    Text: {text}
    """
)

reduce_prompt_template = PromptTemplate.from_template(
    """
    Provide the final summary of the entire text with these important points.
    Add a suitable title, start the precise summary with an introduction and provide the summary in bullet points.
    <Points>
    {summaries}
    </Points>
    """
)

def map_reduce_from_split_docs(split_docs: list):
    # → Map step — summarize each chunk
    intermediate_summaries = []
    
    for doc in split_docs:
        text = doc.page_content
        partial_summary = map_runnable.invoke({"text": text})
        intermediate_summaries.append(partial_summary)

    # → Reduce step — combine intermediate summaries
    joined_summaries = "\n\n".join(intermediate_summaries)
    final_summary = reduce_runnable.invoke({"summaries": joined_summaries})
    
    return final_summary

if st.button("Summarize Context"):
    if not open_ai_api_key.strip() or not url.strip():
        st.error("Please provide both the Azure OpenAI API Key and a valid URL.")
    elif not validators.url(url):
            st.error("Please enter a valid URL.")
    else:
        # Create expander for progress tracking
        progress_expander = st.expander("📋 Processing Steps", expanded=True)
        
        # Step 1: Load content from URL/YouTube
        try:
            with st.spinner("Loading content from URL..."):
                if "youtube.com" in url or "youtu.be" in url:
                    loader = YoutubeLoader.from_youtube_url(
                        url, 
                        add_video_info=False,  # Avoid extra API calls that might fail
                        language=["en", "en-US"]  # Specify transcript languages
                    )
                else:
                    loader = UnstructuredURLLoader(
                        urls=[url],
                        ssl_verify=False,
                        headers={
                            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                        },
                    )
                data = loader.load()
                with progress_expander:
                    st.success("✅ Data loaded from URL")
        except Exception as e:
            st.error(f"❌ Loader error: {e}")
            if "youtube.com" in url or "youtu.be" in url:
                st.info("💡 Possible fixes:\n- Video may not have transcripts/captions\n- Video might be private or age-restricted\n- Try a different public YouTube video")
            st.stop()
        
        # Step 2: Split and summarize
        try:
            with st.spinner("Splitting and summarizing content..."):
                split_docs = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100).split_documents(data)
                
                with progress_expander:
                    st.success(f"✅ Splitting data into {len(split_docs)} chunks")
                
                map_runnable = map_prompt_template | st.session_state.chat_model | StrOutputParser()
                reduce_runnable = reduce_prompt_template | st.session_state.chat_model | StrOutputParser()
                
                # Map step with progress tracking
                intermediate_summaries = []
                for idx, doc in enumerate(split_docs, 1):
                    with progress_expander:
                        st.info(f"🔄 Summarizing chunk {idx}/{len(split_docs)}...")
                    
                    text = doc.page_content
                    partial_summary = map_runnable.invoke({"text": text})
                    intermediate_summaries.append(partial_summary)
                    
                    with progress_expander:
                        st.success(f"✅ Done summarizing chunk {idx}/{len(split_docs)}")
                
                # Reduce step
                with progress_expander:
                    st.info("🔄 Combining all summaries into final output...")
                
                joined_summaries = "\n\n".join(intermediate_summaries)
                summary = reduce_runnable.invoke({"summaries": joined_summaries})
                
                with progress_expander:
                    st.success("✅ Final summary generated!")
                
                # Collapse expander and show final result
                st.success("✅ Summarization complete!")
                st.subheader("📄 Summary")
                st.write(summary)

                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()