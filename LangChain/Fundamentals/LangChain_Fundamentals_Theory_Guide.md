# LangChain Fundamentals - Theory Guide

A comprehensive study guide covering LangChain fundamentals derived from real-world code usage.

---

## Table of Contents

1. [Introduction to LangChain](#1-introduction-to-langchain)
2. [Language Models (LLMs)](#2-language-models-llms)
3. [Messages and Message Types](#3-messages-and-message-types)
4. [Prompts and Prompt Templates](#4-prompts-and-prompt-templates)
5. [Output Parsers](#5-output-parsers)
6. [LCEL (LangChain Expression Language)](#6-lcel-langchain-expression-language)
7. [Document Loaders](#7-document-loaders)
8. [Document Objects](#8-document-objects)
9. [Text Splitters](#9-text-splitters)
10. [Embeddings](#10-embeddings)
11. [Vector Stores](#11-vector-stores)
12. [Retrievers](#12-retrievers)
13. [RAG (Retrieval-Augmented Generation)](#13-rag-retrieval-augmented-generation)
14. [Chat History and Memory](#14-chat-history-and-memory)
15. [Message History Management](#15-message-history-management)
16. [Conversational RAG](#16-conversational-rag)
17. [Async Operations](#17-async-operations)
18. [Batch Processing](#18-batch-processing)
19. [Session Management](#19-session-management)
20. [Custom Chain Functions](#20-custom-chain-functions)
21. [BeautifulSoup Integration](#21-beautifulsoup-integration)
22. [LangServe (API Deployment)](#22-langserve-api-deployment)
23. [Streamlit Integration](#23-streamlit-integration)
24. [LangSmith (Tracing and Monitoring)](#24-langsmith-tracing-and-monitoring)

---

## 1. Introduction to LangChain

### What is LangChain?

LangChain is a framework for building context-aware reasoning applications powered by language models. It provides abstractions and tools to:
- Connect LLMs with external data sources
- Chain multiple operations together
- Build stateful conversational applications
- Implement Retrieval-Augmented Generation (RAG) patterns

### Core Philosophy

LangChain enables you to build applications by composing smaller, reusable components into pipelines (chains) using LCEL (LangChain Expression Language).

---

## 2. Language Models (LLMs)

### Concept

Language Models are the core components that generate text responses. LangChain provides unified interfaces to interact with various LLM providers.

### Classes and Functions

#### `AzureChatOpenAI`

**Definition:** Chat model interface for Azure OpenAI services.

**Common Parameters:**
- `azure_endpoint` - Azure OpenAI service endpoint URL
- `azure_deployment` - Name of the deployed model
- `api_version` - API version to use
- `api_key` - Authentication key for Azure OpenAI

**When to Use:** When integrating with Azure-hosted OpenAI models for chat-based interactions.

**Signature:**
```python
from langchain_openai import AzureChatOpenAI

llm = AzureChatOpenAI(
    azure_endpoint="<endpoint>",
    azure_deployment="<deployment_name>",
    api_version="2025-01-01-preview",
    api_key="<api_key>"
)
```

#### `ChatGroq`

**Definition:** Chat model interface for Groq's fast inference API.

**Common Parameters:**
- `api_key` - Groq API authentication key
- `model` - Model identifier (e.g., "llama-3.1-8b-instant")

**When to Use:** When you need fast inference speeds with open-source models through Groq's infrastructure.

**Signature:**
```python
from langchain_groq import ChatGroq

llm = ChatGroq(api_key="<api_key>", model="llama-3.1-8b-instant")
```

#### `OllamaLLM`

**Definition:** Interface for locally-run Ollama models.

**Common Parameters:**
- `model` - Model name (e.g., "gemma:2b")
- `temperature` - Controls randomness (0.0-1.0)

**When to Use:** When running models locally without cloud dependencies.

**Signature:**
```python
from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="gemma:2b", temperature=0.7)
```

### Methods

#### `invoke()`

**Definition:** Synchronously calls the LLM with input and returns response.

**Parameters:**
- Input can be a string, list of messages, or dictionary

**Returns:** Model response (string or AIMessage object)

---

## 3. Messages and Message Types

### Concept

Messages represent individual units of conversation in chat-based interactions. They maintain structure and context in multi-turn conversations.

### Message Types

#### `HumanMessage`

**Definition:** Represents user input in a conversation.

**Signature:**
```python
from langchain_core.messages import HumanMessage

message = HumanMessage(content="What is AI?")
```

#### `AIMessage`

**Definition:** Represents assistant/model responses in conversation history.

**Signature:**
```python
from langchain_core.messages import AIMessage

message = AIMessage(content="AI is artificial intelligence...")
```

#### `SystemMessage`

**Definition:** Sets system-level instructions or behavior for the model.

**Signature:**
```python
from langchain_core.messages import SystemMessage

message = SystemMessage(content="You are a helpful assistant.")
```

**When to Use:** Use message types when building chatbots or maintaining conversation context across multiple turns.

---

## 4. Prompts and Prompt Templates

### Concept

Prompts are structured input formats that guide LLM behavior. Templates allow dynamic content insertion with variables.

### Classes and Functions

#### `ChatPromptTemplate`

**Definition:** Creates reusable prompt templates with placeholders for dynamic content.

**Common Methods:**
- `from_messages()` - Creates template from list of message tuples
- `from_template()` - Creates simple template from single string

**When to Use:** When you need to structure prompts with system instructions and user inputs, or when building reusable prompt patterns.

**Signature:**
```python
from langchain_core.prompts import ChatPromptTemplate

# Method 1: from_messages
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant. Answer in {language}."),
    ("user", "{user_input}")
])

# Method 2: from_template
prompt = ChatPromptTemplate.from_template(
    "Answer based on context: {context}\nQuestion: {input}"
)
```

#### `MessagesPlaceholder`

**Definition:** Special placeholder that accepts a list of messages (for chat history).

**Common Parameters:**
- `variable_name` - Name of the variable containing message list

**When to Use:** When building conversational applications that need to inject chat history into prompts.

**Signature:**
```python
from langchain_core.prompts import MessagesPlaceholder

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{question}")
])
```

---

## 5. Output Parsers

### Concept

Output parsers transform raw LLM outputs into structured, usable formats.

### Classes and Functions

#### `StrOutputParser`

**Definition:** Extracts string content from LLM responses.

**When to Use:** When you need clean text output from chat models (removes metadata, returns only content).

**Signature:**
```python
from langchain_core.output_parsers import StrOutputParser

parser = StrOutputParser()
result = parser.invoke(llm_response)
```

---

## 6. LCEL (LangChain Expression Language)

### Concept

LCEL is a declarative way to compose chains using the pipe operator (`|`). It enables:
- Sequential execution of operations
- Parallel operations via dictionaries
- Automatic schema validation
- Streaming and async support

### Chain Composition

**Basic Chain Pattern:**
```python
chain = prompt | llm | output_parser
result = chain.invoke({"variable": "value"})
```

**How It Works:**
1. Input flows through the pipe operator
2. Output of each component becomes input to the next
3. Each component must be a "Runnable"

### Parallel Execution

**Pattern:** Use dictionaries to run operations in parallel.

```python
chain = {
    "context": retriever | format_docs,
    "question": RunnablePassthrough()
} | prompt | llm
```

**How It Works:**
- Dictionary keys become variable names for the next stage
- All values execute in parallel
- Results are collected into a dictionary

### Runnables

#### `RunnablePassthrough`

**Definition:** Passes input through unchanged or extracts specific fields.

**When to Use:** When you need to preserve the original input alongside transformed data.

**Signature:**
```python
from langchain_core.runnables import RunnablePassthrough

chain = {"input": RunnablePassthrough()} | next_component
```

#### `RunnableLambda`

**Definition:** Wraps any Python function as a Runnable.

**When to Use:** When integrating custom functions into LCEL chains.

**Signature:**
```python
from langchain_core.runnables import RunnableLambda

def custom_function(x):
    return x.upper()

runnable = RunnableLambda(custom_function)
```

#### `itemgetter`

**Definition:** Extracts specific keys from dictionaries (from Python's operator module).

**When to Use:** When selecting specific fields from previous chain outputs.

**Signature:**
```python
from operator import itemgetter

chain = itemgetter("messages") | next_component
```

---

## 7. Document Loaders

### Concept

Document Loaders ingest data from various sources and convert them into LangChain `Document` objects with content and metadata.

### Classes and Functions

#### `TextLoader`

**Definition:** Loads plain text files.

**When to Use:** For loading `.txt` files for processing.

**Signature:**
```python
from langchain_community.document_loaders import TextLoader

loader = TextLoader("path/to/file.txt")
documents = loader.load()
```

#### `PyPDFLoader`

**Definition:** Loads and extracts text from PDF documents.

**When to Use:** For processing PDF files page by page.

**Signature:**
```python
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader("path/to/file.pdf")
documents = loader.load()
```

#### `WebBaseLoader`

**Definition:** Scrapes and loads content from web pages.

**Common Parameters:**
- `web_paths` - Tuple of URLs to load
- `bs_kwargs` - BeautifulSoup parsing options (e.g., `SoupStrainer` for filtering)

**When to Use:** For ingesting web content, articles, or documentation.

**Signature:**
```python
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://example.com")
documents = loader.load()

# With filtering
from bs4 import SoupStrainer
loader = WebBaseLoader(
    web_paths=("https://example.com",),
    bs_kwargs=dict(parse_only=SoupStrainer(class_="content"))
)
```

#### `WikipediaLoader`

**Definition:** Loads articles from Wikipedia.

**Common Parameters:**
- `query` - Search term
- `load_max_docs` - Maximum number of documents to retrieve

**When to Use:** For fetching encyclopedia content or background information.

**Signature:**
```python
from langchain_community.document_loaders import WikipediaLoader

loader = WikipediaLoader(query="Artificial Intelligence", load_max_docs=2)
documents = loader.load()
```

#### `ArxivLoader`

**Definition:** Loads academic papers from ArXiv.

**Common Parameters:**
- `query` - ArXiv paper ID or search query
- `load_max_docs` - Maximum documents to retrieve

**When to Use:** For research papers and academic content.

**Signature:**
```python
from langchain_community.document_loaders import ArxivLoader

loader = ArxivLoader(query="1706.03762", load_max_docs=2)
documents = loader.load()
```

### Document Object

**Definition:** Standard data structure for loaded content.

**Attributes:**
- `page_content` - The actual text content
- `metadata` - Dictionary with source info, page numbers, etc.

---

## 8. Document Objects

### Concept

Document objects are the standard data structure in LangChain for representing text content with associated metadata. All loaders return lists of Document objects, and they flow through the entire processing pipeline.

### Structure

**`Document` Class**

**Definition:** Container for a piece of text and its metadata.

**Attributes:**
- `page_content` (str) - The actual text content
- `metadata` (dict) - Dictionary containing source information, page numbers, timestamps, etc.

**When to Use:** Documents are created automatically by loaders, but you can create custom ones for specific use cases.

**Signature:**
```python
from langchain_core.documents import Document

doc = Document(
    page_content="This is the text content.",
    metadata={"source": "my-source", "page": 1}
)
```

### Common Metadata Fields

- `source` - Origin of the document (file path, URL, database ID)
- `page` - Page number (for PDFs and books)
- `title` - Document or section title
- `author` - Content creator
- `created_at` - Timestamp
- Custom fields - Any domain-specific information

### Creating Document Lists

**Pattern 1: From Loader**
```python
loader = TextLoader("file.txt")
documents = loader.load()  # Returns list of Documents
```

**Pattern 2: Manual Creation**
```python
documents = [
    Document(
        page_content="Content 1",
        metadata={"source": "doc1", "category": "A"}
    ),
    Document(
        page_content="Content 2",
        metadata={"source": "doc2", "category": "B"}
    )
]
```

### Accessing Document Data

```python
for doc in documents:
    print(doc.page_content)  # Access text
    print(doc.metadata["source"])  # Access metadata
```

### Why Documents Matter

- **Traceability:** Metadata helps track where information came from
- **Filtering:** Can filter by metadata in vector searches
- **Context:** Provides additional context for retrieval and generation
- **Standardization:** Uniform structure across different data sources

---

## 9. Text Splitters

### Concept

Text splitters divide large documents into smaller chunks suitable for embedding and retrieval. Proper chunking maintains context while fitting within model token limits.

### Classes and Functions

#### `RecursiveCharacterTextSplitter`

**Definition:** Splits text recursively using a hierarchy of separators (paragraphs → sentences → words).

**Common Parameters:**
- `chunk_size` - Maximum size of each chunk (in characters)
- `chunk_overlap` - Number of overlapping characters between chunks

**When to Use:** Default choice for most text splitting tasks; preserves semantic coherence.

**Methods:**
- `split_documents(documents)` - Splits Document objects
- `create_documents([texts])` - Creates Documents from raw strings
- `split_text(text)` - Returns list of text strings

**Signature:**
```python
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
chunks = splitter.split_documents(documents)
```

#### `CharacterTextSplitter`

**Definition:** Splits text based on a specific separator.

**Common Parameters:**
- `separator` - Character(s) to split on (e.g., "\n\n")
- `chunk_size` - Maximum chunk size
- `chunk_overlap` - Overlap between chunks

**When to Use:** When you have structured text with clear delimiters.

**Signature:**
```python
from langchain_text_splitters import CharacterTextSplitter

splitter = CharacterTextSplitter(
    separator="\n\n",
    chunk_size=100,
    chunk_overlap=20
)
chunks = splitter.split_documents(documents)
```

#### `HTMLHeaderTextSplitter`

**Definition:** Splits HTML content based on header tags (h1, h2, h3, etc.).

**Common Parameters:**
- `headers_to_split_on` - List of tuples: `[("h1", "Header 1"), ("h2", "Header 2")]`

**When to Use:** For structured HTML documents where headers define sections.

**Methods:**
- `split_text(html_content)` - Splits HTML string
- `split_text_from_url(url)` - Fetches and splits from URL

**Signature:**
```python
from langchain_text_splitters import HTMLHeaderTextSplitter

headers = [("h1", "Header 1"), ("h2", "Header 2"), ("h3", "Header 3")]
splitter = HTMLHeaderTextSplitter(headers_to_split_on=headers)
chunks = splitter.split_text(html_content)
```

#### `RecursiveJsonSplitter`

**Definition:** Splits JSON data while preserving structure.

**Common Parameters:**
- `max_chunk_size` - Maximum size of JSON chunks

**Methods:**
- `split_json(json_data, convert_lists=True)` - Returns JSON chunks
- `create_documents(texts, convert_lists=True)` - Creates Document objects
- `split_text(json_data, convert_lists=True)` - Returns text chunks

**When to Use:** For processing API responses or structured JSON data.

**Signature:**
```python
from langchain_text_splitters import RecursiveJsonSplitter

splitter = RecursiveJsonSplitter(max_chunk_size=200)
chunks = splitter.split_json(json_data, convert_lists=True)
```

---

## 10. Embeddings

### Concept

Embeddings convert text into numerical vector representations that capture semantic meaning. Similar texts have similar vector representations, enabling semantic search.

### Classes and Functions

#### `AzureOpenAIEmbeddings`

**Definition:** Embedding model interface for Azure OpenAI services.

**Common Parameters:**
- `api_key` - Azure OpenAI API key
- `azure_endpoint` - Service endpoint URL
- `model` - Embedding model name
- `api_version` - API version
- `dimensions` - Output vector dimensions (optional)

**When to Use:** For production applications using Azure infrastructure.

**Methods:**
- `embed_query(text)` - Embeds a single query string
- `embed_documents(texts)` - Embeds multiple documents

**Signature:**
```python
from langchain_openai import AzureOpenAIEmbeddings

embeddings = AzureOpenAIEmbeddings(
    api_key="<key>",
    azure_endpoint="<endpoint>",
    model="text-embedding-ada-002",
    api_version="2024-12-01-preview",
    dimensions=1024  # Optional
)

vector = embeddings.embed_query("Sample text")
```

#### `OllamaEmbeddings`

**Definition:** Embeddings using locally-hosted Ollama models.

**Common Parameters:**
- `model` - Model name (e.g., "gemma:2b", "mxbai-embed-large")

**When to Use:** For local development or when data privacy requires on-premise processing.

**Signature:**
```python
from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(model="mxbai-embed-large")
vector = embeddings.embed_query("Sample text")
```

#### `HuggingFaceEmbeddings`

**Definition:** Uses HuggingFace's sentence transformers for embeddings.

**Common Parameters:**
- `model_name` - HuggingFace model identifier (e.g., "all-MiniLM-L6-v2")
- `model_kwargs` - Additional arguments like `{"device": "cpu"}`

**When to Use:** For free, open-source embedding models with good performance.

**Signature:**
```python
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"}
)
```

---

## 11. Vector Stores

### Concept

Vector stores are specialized databases that store embeddings and enable efficient similarity searches. They power semantic search and retrieval.

### Classes and Functions

#### `FAISS`

**Definition:** Facebook AI Similarity Search - fast in-memory vector store.

**Common Methods:**
- `from_documents(docs, embeddings)` - Creates vector store from documents
- `similarity_search(query, k=4)` - Returns k most similar documents
- `similarity_search_with_score(query)` - Returns documents with similarity scores
- `similarity_search_by_vector(vector)` - Searches using embedding vector
- `as_retriever()` - Converts to retriever interface
- `save_local(path)` - Persists to disk
- `load_local(path, embeddings, allow_dangerous_deserialization)` - Loads from disk

**When to Use:** For fast prototyping and small to medium datasets; requires local storage.

**Signature:**
```python
from langchain_community.vectorstores import FAISS

db = FAISS.from_documents(documents, embeddings)
results = db.similarity_search("query", k=4)

# Persistence
db.save_local("faiss_index")
loaded_db = FAISS.load_local("faiss_index", embeddings, 
                             allow_dangerous_deserialization=True)
```

#### `Chroma`

**Definition:** Open-source embedding database with persistence support.

**Common Methods:**
- `from_documents(docs, embedding)` - Creates vector store
- `similarity_search(query, k=4)` - Retrieves similar documents
- `similarity_search_with_score(query)` - Returns documents with scores
- `as_retriever(search_kwargs={"k": n})` - Converts to retriever

**Common Parameters:**
- `persist_directory` - Directory for persistent storage
- `embedding_function` - Embedding model to use

**When to Use:** For applications requiring persistent storage and easy deployment.

**Signature:**
```python
from langchain_chroma import Chroma

# In-memory
db = Chroma.from_documents(documents, embeddings)

# Persistent
db = Chroma.from_documents(
    documents,
    embeddings,
    persist_directory="./chroma_db"
)

# Load existing
db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)
```

### Similarity Search Methods

**`similarity_search(query, k=4)`**
- Returns: List of Document objects
- Use when: You only need the documents

**`similarity_search_with_score(query)`**
- Returns: List of (Document, score) tuples
- Score: Lower = more similar (distance metric)
- Use when: You need to threshold or rank by relevance

**`similarity_search_by_vector(embedding_vector)`**
- Returns: List of Document objects
- Use when: You already have an embedding vector

---

## 12. Retrievers

### Concept

Retrievers provide a standard interface for fetching relevant documents. They bridge vector stores and LangChain pipelines, enabling seamless integration in LCEL chains.

### Classes and Functions

#### `as_retriever()`

**Definition:** Converts a vector store into a retriever Runnable.

**Common Parameters:**
- `search_type` - Type of search ("similarity", "mmr", etc.)
- `search_kwargs` - Dictionary with search parameters like `{"k": 3}`

**When to Use:** When building RAG pipelines or chains requiring document retrieval.

**Signature:**
```python
retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

# Use in chain
chain = retriever | process_documents
```

### Retriever Methods

**`invoke(query)`**
- Synchronously retrieves documents for a query

**`batch([queries])`**
- Processes multiple queries in parallel

---

## 13. RAG (Retrieval-Augmented Generation)

### Concept

RAG combines retrieval and generation: retrieve relevant documents, inject them as context, and generate answers grounded in that context. This reduces hallucinations and provides up-to-date information.

### RAG Pipeline Pattern

```python
# 1. Retrieve documents
retriever = vector_store.as_retriever()

# 2. Format documents into string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 3. Build RAG chain
rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("What is X?")
```

### Components

**Retriever Stage:**
- Fetches relevant documents based on query
- Runs in parallel with question passthrough

**Context Formatting:**
- Converts list of Documents into single context string
- Custom function wrapped in chain

**Prompt Template:**
- Combines context and question
- Instructs model to answer based only on context

**LLM Generation:**
- Generates answer using provided context
- Output parser extracts clean text

---

## 14. Chat History and Memory

### Concept

Chat history enables stateful conversations by storing previous messages. LangChain provides abstractions to manage conversation memory across sessions.

### Classes and Functions

#### `ChatMessageHistory`

**Definition:** In-memory storage for conversation messages.

**When to Use:** For storing chat history in simple applications or testing.

**Signature:**
```python
from langchain_community.chat_message_histories import ChatMessageHistory

history = ChatMessageHistory()
history.add_user_message("Hello")
history.add_ai_message("Hi there!")
```

#### `BaseChatMessageHistory`

**Definition:** Abstract base class for message history implementations.

**When to Use:** For implementing custom storage backends (databases, Redis, etc.).

#### `RunnableWithMessageHistory`

**Definition:** Wraps a chain to automatically inject and manage chat history.

**Common Parameters:**
- Runnable chain to wrap
- `get_session_history` - Function returning history for a session ID
- `input_messages_key` - Key containing input messages (if using dict input)
- `history_messages_key` - Key for history in prompt template

**When to Use:** For building conversational applications with persistent context.

**Signature:**
```python
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="chat_history"
)

config = {"configurable": {"session_id": "user_123"}}
response = conversational_chain.invoke(
    {"messages": [HumanMessage(content="Hello")]},
    config=config
)
```

---

## 15. Message History Management

### Concept

Long conversations can exceed token limits. Message trimming and summarization strategies manage conversation length while preserving important context.

### Classes and Functions

#### `trim_messages`

**Definition:** Trims message history based on token count or message count.

**Common Parameters:**
- `max_tokens` - Maximum tokens to keep
- `strategy` - Trimming strategy ("last", "first")
- `token_counter` - Method to count tokens ("approximate", model instance)
- `include_system` - Whether to always keep system messages
- `allow_partial` - Allow partial message splitting
- `start_on` - Ensure trimmed history starts with specific role ("human", "ai")

**When to Use:** For managing conversation history in long-running chats.

**Signature:**
```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(
    max_tokens=500,
    strategy="last",
    token_counter="approximate",
    include_system=True,
    start_on="human"
)

trimmed_messages = trimmer.invoke(messages)
```

**Note:** Use `token_counter="approximate"` with `AzureChatOpenAI` to avoid tokenization errors.

### Integration Pattern

```python
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough

chain = (
    RunnablePassthrough.assign(
        messages=itemgetter("messages") | trimmer
    )
    | prompt
    | llm
    | StrOutputParser()
)
```

---

## 16. Conversational RAG

### Concept

Conversational RAG combines retrieval-augmented generation with chat history. It reformulates questions based on conversation context before retrieving documents.

### Pattern

#### Step 1: Contextualize Question

**Purpose:** Convert follow-up questions into standalone queries.

```python
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", "Reformulate question based on chat history..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

contextualize_chain = (
    contextualize_q_prompt
    | llm
    | StrOutputParser()
)
```

#### Step 2: History-Aware Retriever

**Purpose:** Retrieve documents using contextualized question.

```python
history_aware_retriever = contextualize_chain | retriever
```

#### Step 3: QA Chain with History

**Purpose:** Generate answer using retrieved context and conversation history.

```python
qa_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer based on context..."),
    MessagesPlaceholder("chat_history"),
    ("human", "{question}")
])

rag_chain = (
    {
        "context": history_aware_retriever | format_docs,
        "question": lambda x: x["question"],
        "chat_history": lambda x: x["chat_history"]
    }
    | qa_prompt
    | llm
    | StrOutputParser()
)
```

#### Step 4: Wrap with Message History

```python
conversational_rag = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="chat_history"
)

response = conversational_rag.invoke(
    {"question": "What is X?"},
    config={"configurable": {"session_id": "abc123"}}
)
```

### Key Components

**Contextualize Chain:**
- Reformulates questions using chat history
- Ensures retrieval isn't dependent on conversational context

**History-Aware Retriever:**
- Uses reformulated standalone question for document retrieval
- Maintains relevance across conversation turns

**QA Chain:**
- Receives both retrieved documents and chat history
- Generates contextually-aware responses

---

## 17. Async Operations

### Concept

Async (asynchronous) operations enable concurrent execution, improving performance when making multiple API calls or processing multiple queries simultaneously. LangChain provides async versions of most operations.

### Why Use Async?

- **Performance:** Process multiple queries concurrently
- **Efficiency:** Don't block while waiting for I/O operations
- **Scalability:** Handle more requests with same resources
- **Better UX:** Keep applications responsive

### Async Methods in Vector Stores

#### `asimilarity_search()`

**Definition:** Async version of similarity search.

**When to Use:** When searching from async contexts (FastAPI endpoints, async chains).

**Signature:**
```python
results = await vector_store.asimilarity_search("query", k=2)
```

#### `asimilarity_search_with_score()`

**Definition:** Async similarity search with relevance scores.

**Returns:** List of (Document, score) tuples

**Signature:**
```python
results = await vector_store.asimilarity_search_with_score("query", k=2)
```

### Async Pattern

```python
import asyncio

async def process_queries():
    query1 = vector_store.asimilarity_search("What is X?")
    query2 = vector_store.asimilarity_search("What is Y?")
    
    # Execute concurrently
    results = await asyncio.gather(query1, query2)
    return results

# Run async function
results = asyncio.run(process_queries())
```

### Async in LCEL Chains

LCEL chains automatically support async when using `ainvoke()` instead of `invoke()`:

```python
response = await chain.ainvoke({"input": "query"})
```

### Best Practices

- Use async operations when handling multiple concurrent requests
- Combine with `asyncio.gather()` for parallel processing
- Prefer async in web frameworks (FastAPI, async web servers)
- Keep sync for simple scripts and notebooks

---

## 18. Batch Processing

### Concept

Batch processing allows you to process multiple inputs in a single call, optimizing performance through parallelization and reduced overhead.

### Batch Methods

#### `batch()`

**Definition:** Process multiple inputs concurrently.

**Available on:** Retrievers, chains, runnables, LLMs

**When to Use:** When you have multiple queries to process simultaneously.

**Signature:**
```python
# Retriever batch
results = retriever.batch([
    "What is LangChain?",
    "What is RAG?",
    "What are embeddings?"
])

# Chain batch
responses = chain.batch([
    {"input": "query1"},
    {"input": "query2"}
])
```

### Benefits

**Performance:**
- Processes queries in parallel
- Reduces total execution time
- Better resource utilization

**Use Cases:**
- Processing multiple user queries
- Bulk document analysis
- Testing with multiple inputs
- Evaluation and benchmarking

### Example: Batch Retrieval

```python
retriever = vector_store.as_retriever(search_kwargs={"k": 2})

questions = [
    "Tell me about dogs.",
    "Tell me about cats.",
    "Tell me about birds."
]

# Returns list of document lists
all_results = retriever.batch(questions)

for i, docs in enumerate(all_results):
    print(f"Question {i+1} results: {len(docs)} documents")
```

### Async Batch Processing

For even better performance, combine batch with async:

```python
results = await retriever.abatch(["query1", "query2", "query3"])
```

---

## 19. Session Management

### Concept

Session management enables multi-user applications by maintaining separate conversation contexts for different users or sessions. Each session has isolated chat history.

### Session ID Pattern

**Configuration Object:**
```python
config = {"configurable": {"session_id": "unique_user_id"}}
```

**Purpose:**
- Identifies which conversation history to use
- Isolates different users' conversations
- Enables conversation persistence and recovery

### Implementation Pattern

#### Step 1: Session Storage

```python
from langchain_community.chat_message_histories import ChatMessageHistory

# In-memory storage (simple applications)
store = {}

def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]
```

#### Step 2: Configure Session-Aware Chain

```python
from langchain_core.runnables.history import RunnableWithMessageHistory

conversational_chain = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="messages",
    history_messages_key="chat_history"
)
```

#### Step 3: Use with Session IDs

```python
# User 1 conversation
response1 = conversational_chain.invoke(
    {"messages": [HumanMessage(content="Hello, I'm Alice")]},
    config={"configurable": {"session_id": "user_alice"}}
)

# User 2 conversation (separate context)
response2 = conversational_chain.invoke(
    {"messages": [HumanMessage(content="Hello, I'm Bob")]},
    config={"configurable": {"session_id": "user_bob"}}
)
```

### Session ID Strategies

**User-Based:**
- `session_id = f"user_{user_id}"`
- One persistent conversation per user

**Conversation-Based:**
- `session_id = f"conv_{conversation_id}"`
- Multiple conversations per user

**Time-Based:**
- `session_id = f"user_{user_id}_{date}"`
- Fresh context daily/weekly

**Random/Unique:**
- `session_id = str(uuid.uuid4())`
- Temporary sessions

### Production Considerations

**Persistent Storage:**
- Replace in-memory dict with database (Redis, PostgreSQL, MongoDB)
- Use LangChain integrations (e.g., `RedisChatMessageHistory`)

**Session Cleanup:**
- Implement TTL (time-to-live) for old sessions
- Periodic cleanup of inactive sessions

**Scaling:**
- Ensure session storage is accessible across multiple app instances
- Use distributed caching (Redis, Memcached)

---

## 20. Custom Chain Functions

### Concept

Custom functions enable data transformation and processing within LCEL chains. They can be used directly or wrapped as Runnables.

### Common Patterns

#### `format_docs()`

**Definition:** Standard function to convert list of Documents to single string.

**Purpose:** Prepares retrieved documents for prompt context.

**Signature:**
```python
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Usage in chain
chain = retriever | format_docs | next_component
```

**Variations:**
```python
# With source attribution
def format_docs_with_source(docs):
    return "\n\n".join(
        f"Source: {doc.metadata['source']}\n{doc.page_content}"
        for doc in docs
    )

# With numbering
def format_docs_numbered(docs):
    return "\n\n".join(
        f"Document {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(docs)
    )

# With empty check
def format_docs(docs):
    if not docs:
        return "No relevant context found."
    return "\n\n".join(doc.page_content for doc in docs)
```

### Lambda Functions in Chains

**Purpose:** Inline data extraction or transformation.

**Common Uses:**

**Extract Dictionary Values:**
```python
chain = {
    "question": lambda x: x["question"],
    "chat_history": lambda x: x["chat_history"]
} | prompt | llm
```

**Transform Data:**
```python
chain = {
    "uppercase_input": lambda x: x["text"].upper()
} | next_component
```

**Conditional Logic:**
```python
def route_query(input_dict):
    query = input_dict["query"]
    if "technical" in query.lower():
        return technical_chain
    return general_chain
```

### Wrapping Functions as Runnables

**Using RunnableLambda:**
```python
from langchain_core.runnables import RunnableLambda

def custom_processor(text):
    # Complex processing logic
    return processed_text

processor_runnable = RunnableLambda(custom_processor)
chain = input_source | processor_runnable | llm
```

### Best Practices

- Keep functions pure (no side effects when possible)
- Handle edge cases (empty lists, None values)
- Use descriptive names for clarity
- Consider error handling for production
- Document expected input/output types

---

## 21. BeautifulSoup Integration

### Concept

BeautifulSoup integration with `WebBaseLoader` enables precise HTML parsing and content extraction, filtering out navigation, ads, and other irrelevant content.

### Classes and Functions

#### `SoupStrainer`

**Definition:** Filters HTML to parse only specific parts of a webpage.

**Common Parameters:**
- `class_` - CSS class name(s) to extract
- `id` - HTML element ID
- `name` - HTML tag name
- Custom functions for advanced filtering

**When to Use:** When you want only specific sections of a webpage (main content, articles, specific divs).

**Signature:**
```python
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

# Extract by class
loader = WebBaseLoader(
    web_paths=("https://example.com",),
    bs_kwargs=dict(
        parse_only=SoupStrainer(class_="article-content")
    )
)

# Extract by ID
loader = WebBaseLoader(
    web_paths=("https://example.com",),
    bs_kwargs=dict(
        parse_only=SoupStrainer(id="main-content")
    )
)

# Extract by tag
loader = WebBaseLoader(
    web_paths=("https://example.com",),
    bs_kwargs=dict(
        parse_only=SoupStrainer(name="article")
    )
)
```

### Multiple Criteria

**Extract Multiple Classes:**
```python
loader = WebBaseLoader(
    web_paths=("https://example.com",),
    bs_kwargs=dict(
        parse_only=SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    )
)
```

### Benefits

**Cleaner Data:**
- Removes navigation menus, footers, sidebars
- Focuses on main content
- Reduces noise in embeddings

**Faster Processing:**
- Parses less HTML
- Smaller document sizes
- Faster chunking and embedding

**Better Retrieval:**
- More relevant content in vector store
- Improved search quality
- Less irrelevant context in RAG responses

### Common Patterns

**Blog Posts:**
```python
SoupStrainer(class_=("post-content", "article-body"))
```

**Documentation:**
```python
SoupStrainer(class_="documentation")
```

**News Articles:**
```python
SoupStrainer(class_=("article-content", "story-body"))
```

**Wikipedia:**
```python
SoupStrainer(class_="mw-body-content")
```

### Advanced Filtering

**Custom Function:**
```python
def custom_filter(tag):
    return tag.name == "p" and "important" in tag.get("class", [])

loader = WebBaseLoader(
    web_paths=("https://example.com",),
    bs_kwargs=dict(parse_only=SoupStrainer(custom_filter))
)
```

---

## 22. LangServe (API Deployment)

### Concept

LangServe deploys LangChain chains as REST APIs using FastAPI, enabling easy integration with web applications and services.

### Classes and Functions

#### `add_routes`

**Definition:** Adds LangChain chain endpoints to FastAPI app.

**Common Parameters:**
- `app` - FastAPI application instance
- `runnable` - LangChain chain to expose
- `path` - URL path for the endpoint

**When to Use:** For deploying chains as production APIs.

**Signature:**
```python
from fastapi import FastAPI
from langserve import add_routes

app = FastAPI(
    title="LangChain Server",
    version="0.1",
    description="API for LangChain"
)

add_routes(app, chain, path="/chain")

# Endpoints created:
# POST /chain/invoke
# POST /chain/batch
# POST /chain/stream
```

**Running the Server:**
```python
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
```

---

## 23. Streamlit Integration

### Concept

Streamlit enables rapid development of interactive web UIs for LangChain applications. It provides simple APIs for user input, displaying responses, and building conversational interfaces.

### Key Streamlit Functions for LangChain

#### `st.title()`

**Definition:** Sets the main title/heading of the app.

**Signature:**
```python
import streamlit as st

st.title("LangChain Demo with Ollama")
```

#### `st.text_input()`

**Definition:** Creates a text input field for user queries.

**Common Parameters:**
- First parameter: Label text
- `value`: Default value
- `placeholder`: Placeholder text

**Returns:** User's input as a string

**Signature:**
```python
user_query = st.text_input("Enter your question:")
```

#### `st.button()`

**Definition:** Creates a clickable button.

**Returns:** `True` when clicked, `False` otherwise

**Signature:**
```python
if st.button("Get Answer"):
    # Process query
    pass
```

#### `st.write()`

**Definition:** Displays text, markdown, or data.

**Signature:**
```python
st.write("Answer:", response)
st.write("# Markdown heading")
st.write(dataframe)
```

### Basic LangChain + Streamlit Pattern

```python
import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Setup chain (outside button to avoid recreation)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are helpful."),
    ("user", "Question: {question}")
])

llm = OllamaLLM(model="gemma:2b", temperature=0.7)
chain = prompt | llm | StrOutputParser()

# UI
st.title("LangChain Chatbot")
user_input = st.text_input("Ask a question:")

if st.button("Submit"):
    if user_input:
        response = chain.invoke({"question": user_input})
        st.write("**Answer:**", response)
    else:
        st.write("Please enter a question.")
```

### Advanced Features

**Session State (for chat history):**
```python
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    st.write(f"**{msg['role']}:** {msg['content']}")

# Add new messages
if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
```

**Spinner (loading indicator):**
```python
with st.spinner("Thinking..."):
    response = chain.invoke({"question": user_input})
```

**Sidebar Configuration:**
```python
with st.sidebar:
    st.header("Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    model = st.selectbox("Model", ["gemma:2b", "llama2"])
```

### When to Use Streamlit

- **Rapid prototyping:** Build UI in minutes
- **Internal tools:** Quick demos and MVPs
- **Data apps:** Combine LLMs with data visualization
- **Non-web developers:** Python-only, no HTML/CSS/JS needed

### Deployment

**Run locally:**
```bash
streamlit run app.py
```

**Deploy to Streamlit Cloud:**
- Push code to GitHub
- Connect repository to Streamlit Cloud
- Automatic deployment and hosting

---

## 24. LangSmith (Tracing and Monitoring)

### Concept

LangSmith provides observability for LangChain applications through tracing, debugging, and evaluation capabilities.

### Configuration

**Environment Variables:**
- `LANGCHAIN_API_KEY` - LangSmith API key
- `LANGCHAIN_TRACING_V2` - Enable tracing ("true")
- `LANGCHAIN_PROJECT` - Project name for organizing traces

**When to Use:** For debugging chains, monitoring production apps, and evaluating performance.

**Setup:**
```python
import os

os.environ["LANGCHAIN_API_KEY"] = "your_api_key"
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "my_project"
```

**Benefits:**
- Visual trace of chain execution
- Latency and token tracking
- Input/output inspection
- Error debugging

---

## Summary

This guide covers the fundamental concepts and components of LangChain:

1. **Core Components:** LLMs, prompts, output parsers, messages
2. **LCEL:** Composing chains with pipes and runnables
3. **Data Processing:** Document loaders, document objects, text splitters, embeddings
4. **Retrieval:** Vector stores and retrievers
5. **Advanced Patterns:** RAG, conversational AI, memory management
6. **Performance:** Async operations, batch processing
7. **Multi-user:** Session management and isolation
8. **Customization:** Custom chain functions, BeautifulSoup integration
9. **Deployment:** LangServe APIs, Streamlit UIs, LangSmith monitoring

### Learning Path

1. Start with simple LLM calls and prompt templates
2. Learn LCEL for chain composition
3. Understand document objects and their structure
4. Master document loading and chunking
5. Understand embeddings and vector stores
6. Build basic RAG applications
7. Add conversational memory and session management
8. Explore async and batch processing for performance
9. Create custom chain functions for specific needs
10. Build UIs with Streamlit or APIs with LangServe
11. Deploy and monitor production applications with LangSmith

### Best Practices

- Use `RecursiveCharacterTextSplitter` for most text splitting tasks
- Choose vector stores based on scale (FAISS for prototyping, Chroma for production)
- Always use `StrOutputParser` for clean text outputs
- Implement message trimming for long conversations
- Use session management for multi-user applications
- Leverage async operations for concurrent processing
- Use batch processing when handling multiple queries
- Filter web content with `SoupStrainer` for cleaner data
- Create reusable `format_docs()` functions for consistency
- Enable LangSmith tracing during development and production
- Use `RunnablePassthrough` for preserving inputs in parallel stages
- Build quick UIs with Streamlit for prototyping
- Test chains incrementally before adding complexity
- Handle edge cases in custom functions (empty lists, None values)

---

**End of Guide**
