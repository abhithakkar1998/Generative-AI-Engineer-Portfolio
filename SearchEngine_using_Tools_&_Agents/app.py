import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
import os
from dotenv import load_dotenv

load_dotenv("/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/.env")

api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

search = DuckDuckGoSearchRun(name="Search")

st.title("LangChain - Chat with Search")

if "openai_api_key" not in st.session_state:
    st.session_state["openai_api_key"] = os.getenv("AZURE_OPENAI_API_KEY", "")

if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "assistant", "content": "Hi! I am your assistant who can search the web. How can I help you today?"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input(placeholder="Type your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    chat_model = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        azure_deployment="gpt-5.2-chat",
        api_version="2025-04-01-preview",
        api_key=st.session_state["openai_api_key"],
        max_tokens=1024,
        streaming=True,
    )

    tools = [wiki, arxiv, search]

    agent = create_agent(
        model=chat_model,
        tools=tools,
        system_prompt="You are a helpful assistant who can search the web and other tools to answer user queries.",
    )

    with st.chat_message("assistant"):
        # Collapsible tool usage section
        tool_expander = st.expander("🔧 Tool Usage", expanded=True)
        tool_container = tool_expander.container()
        
        # Response section - completely separate
        response_placeholder = st.empty()
        
        full_response = ""
        tools_used = set()  # Track which tools we've already displayed
        
        # Stream the agent
        for chunk in agent.stream(
            {"messages": [{"role": "user", "content": prompt}]},
            stream_mode="values"
        ):
            if "messages" in chunk:
                messages = chunk["messages"]
                
                for message in messages:
                    # Track tool calls
                    if isinstance(message, AIMessage) and message.tool_calls:
                        for tool_call in message.tool_calls:
                            tool_id = tool_call.get("id", "")
                            
                            # Only display each tool call once
                            if tool_id and tool_id not in tools_used:
                                tools_used.add(tool_id)
                                
                                tool_name = tool_call.get("name", "Unknown Tool")
                                tool_args = tool_call.get("args", {})
                                
                                with tool_container:
                                    st.markdown(f"**🔍 Using: {tool_name}**")
                                    
                                    # Extract and display the query
                                    query = tool_args.get('query') or tool_args.get('q') or str(tool_args)
                                    if query and query != '{}':
                                        st.caption(f"Query: {query}")
                    
                    # Track tool results
                    elif isinstance(message, ToolMessage):
                        with tool_container:
                            st.success("✓ Result received")
                    
                    # Display final AI response
                    elif isinstance(message, AIMessage) and message.content and not message.tool_calls:
                        full_response = message.content
                        response_placeholder.markdown(full_response + "▌")
        
        # Remove typing cursor
        response_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# # Previous version of the code using StreamlitCallbackHandler
# import streamlit as st
# from langchain_openai import AzureChatOpenAI
# from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
# from langchain_community.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
# from langchain.agents import create_agent
# from langchain_community.callbacks import StreamlitCallbackHandler
# from langchain_core.messages import AIMessage, ToolMessage
# import os
# from dotenv import load_dotenv

# load_dotenv("/home/abhi/AI_Workspace/personal/Generative-AI-Engineer-Portfolio/.env")

# api_wrapper_wiki = WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=250)
# wiki = WikipediaQueryRun(api_wrapper=api_wrapper_wiki)

# api_wrapper_arxiv = ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=250)
# arxiv = ArxivQueryRun(api_wrapper=api_wrapper_arxiv)

# search = DuckDuckGoSearchRun(name="Search")

# st.title("LangChain - Chat with Search")

# if "openai_api_key" not in st.session_state:
#     st.session_state["openai_api_key"] = os.getenv("AZURE_OPENAI_API_KEY", "")

# if "messages" not in st.session_state:
#     st.session_state["messages"] = [
#         {"role": "assistant", "content": "Hi! I am your assistant who can search the web. How can I help you today?"}
#     ]

# for msg in st.session_state.messages:
#     st.chat_message(msg["role"]).write(msg["content"])

# if prompt := st.chat_input(placeholder="Type your question here..."):
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     st.chat_message("user").write(prompt)

#     chat_model = AzureChatOpenAI(
#         azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
#         azure_deployment="gpt-5.2-chat",
#         api_version="2025-04-01-preview",
#         api_key=st.session_state["openai_api_key"],
#         max_tokens=1024,
#         streaming=True,
#     )

#     tools = [wiki, arxiv, search]

#     agent = create_agent(
#         model=chat_model,
#         tools=tools,
#         system_prompt="You are a helpful assistant who can search the web and other tools to answer user queries.",
#     )

#     with st.chat_message("assistant"):
#         # Collapsible tool usage section
#         with st.expander("🔧 Tool Usage", expanded=False):
#             tool_container = st.container()
#             st_callback = StreamlitCallbackHandler(tool_container)
        
#         # Response section - completely separate
#         response_placeholder = st.empty()
        
#         full_response = ""
        
#         # Stream the agent
#         for chunk in agent.stream(
#             {"messages": [{"role": "user", "content": prompt}]},
#             config={"callbacks": [st_callback]},
#             stream_mode="values"
#         ):
#             if "messages" in chunk:
#                 messages = chunk["messages"]
#                 if messages:
#                     latest_message = messages[-1]
                    
#                     # Only show the final AI response, not intermediate thinking
#                     if isinstance(latest_message, AIMessage) and latest_message.content:
#                         # Check if this is actually the final response (not tool output)
#                         if not latest_message.tool_calls:
#                             full_response = latest_message.content
#                             response_placeholder.markdown(full_response + "▌")
        
#         # Remove typing cursor
#         response_placeholder.markdown(full_response)
    
#     st.session_state.messages.append({"role": "assistant", "content": full_response})