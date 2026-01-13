# LangChain Advanced Concepts - Theory Guide

A comprehensive study guide covering advanced LangChain concepts derived from real-world code usage.

---

## Table of Contents

1. [Introduction to Advanced LangChain](#1-introduction-to-advanced-langchain)
2. [Model Integration](#2-model-integration)
3. [Advanced Message Types](#3-advanced-message-types)
4. [Streaming](#4-streaming)
5. [Advanced Batch Processing](#5-advanced-batch-processing)
6. [Structured Output](#6-structured-output)
7. [Tools](#7-tools)
8. [Agents](#8-agents)
9. [Middleware](#9-middleware)

---

## 1. Introduction to Advanced LangChain

### What This Guide Covers

This guide builds upon the fundamentals and explores advanced LangChain features:
- Flexible model integration across providers
- Advanced message handling with metadata
- Real-time streaming responses
- Structured output generation
- Tool calling and execution
- Autonomous agents
- Middleware for conversation management

### Prerequisites

Before diving into these concepts, you should be familiar with:
- Basic LangChain components (LLMs, prompts, chains)
- LCEL (LangChain Expression Language)
- Message types and chat models
- Basic prompt templates

---

## 2. Model Integration

### Concept

LangChain provides unified interfaces to integrate with multiple LLM providers. This abstraction allows you to switch between providers without rewriting application code.

### Classes and Functions

#### `init_chat_model()`

**Definition:** Universal function to initialize chat models from any provider.

**Common Parameters:**
- First parameter: Model name or deployment name
- `model_provider` - Provider identifier (e.g., "azure_openai", "groq", "google_genai")
- `deployment_name` - Azure-specific deployment name
- `azure_endpoint` - Azure OpenAI endpoint URL
- `openai_api_key` - API key for authentication
- `openai_api_version` - API version string

**When to Use:** When you want provider-agnostic code or need to dynamically switch between models.

**Signature:**
```python
from langchain.chat_models import init_chat_model

# Azure OpenAI
model = init_chat_model(
    "gpt-4o",
    model_provider="azure_openai",
    deployment_name="gpt-4o-deployment",
    azure_endpoint="https://your-resource.openai.azure.com",
    openai_api_key="your-key",
    openai_api_version="2023-05-15"
)

# Google Gemini
model = init_chat_model("google_genai:gemini-3-flash-preview")

# Groq
model = init_chat_model("groq:llama-3.1-8b-instant")
```

### Provider-Specific Classes

#### `AzureChatOpenAI`

**Definition:** Direct Azure OpenAI integration.

**Common Parameters:**
- `deployment_name` - Azure deployment identifier
- `azure_endpoint` - Service endpoint
- `openai_api_key` - Authentication key
- `api_version` - API version

**When to Use:** When you're exclusively using Azure and want direct control over Azure-specific features.

**Signature:**
```python
from langchain_openai import AzureChatOpenAI

model = AzureChatOpenAI(
    deployment_name="gpt-4o-deployment",
    azure_endpoint="https://resource.openai.azure.com",
    openai_api_key="key",
    api_version="2023-05-15"
)
```

#### `ChatGoogleGenerativeAI`

**Definition:** Google Gemini model integration.

**Common Parameters:**
- `model` - Model identifier (e.g., "gemini-3-flash-preview")
- Requires `GOOGLE_API_KEY` environment variable

**Signature:**
```python
from langchain_google_genai import ChatGoogleGenerativeAI

model = ChatGoogleGenerativeAI(model="gemini-3-flash-preview")
```

#### `ChatGroq`

**Definition:** Groq inference API integration for fast open-source models.

**Common Parameters:**
- `model` - Model name (e.g., "llama-3.1-8b-instant")
- Requires `GROQ_API_KEY` environment variable

**Signature:**
```python
from langchain_groq import ChatGroq

model = ChatGroq(model="llama-3.1-8b-instant")
```

### Benefits of Unified Integration

**Provider Flexibility:**
- Switch providers without code changes
- A/B test different models
- Fallback to alternative providers

**Consistent Interface:**
- Same methods across all providers
- Predictable behavior
- Simplified maintenance

**Cost Optimization:**
- Route queries to cheapest suitable model
- Use different providers for different tasks

---

## 3. Advanced Message Types

### Concept

Beyond basic messages, LangChain supports rich message types with metadata, roles, and tool interactions. Understanding these enables sophisticated conversational patterns.

### Message Components

Every message has three core components:

1. **Role** - Identifies the speaker (system, human, ai, tool)
2. **Content** - The actual message text or data
3. **Metadata** - Additional context and information

### Message Types Deep Dive

#### `SystemMessage`

**Definition:** Sets model behavior, context, and instructions.

**Purpose:**
- Define assistant personality
- Set domain expertise
- Provide operational constraints
- Establish response format

**When to Use:** At the start of conversations to establish context.

**Advanced Usage:**
```python
from langchain.messages import SystemMessage

system_message = SystemMessage("""
You are a Python programming expert with specialization in ML and AI.
Always provide code snippets and explain your reasoning.
Be concise but thorough in your explanations.
""")
```

#### `HumanMessage`

**Definition:** Represents user input with optional metadata.

**Common Parameters:**
- `content` - The message text
- `name` - Optional identifier for the user
- `id` - Unique message identifier

**When to Use:** For all user queries and inputs.

**With Metadata:**
```python
from langchain.messages import HumanMessage

human_msg = HumanMessage(
    content="Explain quantum computing.",
    name="Alice",
    id="msg-001"
)
```

#### `AIMessage`

**Definition:** Represents assistant responses, including tool calls.

**Common Parameters:**
- `content` - Response text (can be empty if using tools)
- `tool_calls` - List of tool invocations

**When to Use:** 
- When building conversation history manually
- When simulating multi-turn conversations
- When implementing tool execution loops

**Basic Usage:**
```python
from langchain.messages import AIMessage

ai_msg = AIMessage(content="I'd be happy to help!")
```

**With Tool Calls:**
```python
ai_msg = AIMessage(
    content=[],
    tool_calls=[{
        "name": "get_weather",
        "args": {"location": "San Francisco"},
        "id": "call_123"
    }]
)
```

#### `ToolMessage`

**Definition:** Contains results from tool execution.

**Common Parameters:**
- `content` - Tool output (string or structured data)
- `tool_call_id` - Must match the tool call ID from AIMessage

**When to Use:** After executing tools to provide results back to the model.

**Signature:**
```python
from langchain.messages import ToolMessage

tool_msg = ToolMessage(
    content="Sunny, 68°F",
    tool_call_id="call_123"
)
```

### Message Flow Pattern

**Complete Tool Execution Flow:**
```python
messages = [
    SystemMessage("You are a weather assistant."),
    HumanMessage("What's the weather in SF?"),
    AIMessage(
        content=[],
        tool_calls=[{
            "name": "get_weather",
            "args": {"location": "San Francisco"},
            "id": "call_123"
        }]
    ),
    ToolMessage(
        content="Sunny, 68°F",
        tool_call_id="call_123"
    )
]

response = model.invoke(messages)
```

### Why Message Types Matter

**Context Preservation:**
- Maintains conversation flow
- Enables multi-turn reasoning
- Supports complex interactions

**Tool Integration:**
- Structured tool calling
- Result tracking
- Error handling

**Debugging:**
- Clear role separation
- Traceable message flow
- Metadata for analysis

---

## 4. Streaming

### Concept

Streaming provides real-time response generation, displaying tokens as they're produced rather than waiting for complete responses. This improves user experience and perceived performance.

### Benefits

**User Experience:**
- Immediate feedback
- Reduced perceived latency
- Better engagement

**Long Responses:**
- Progressive display
- Early cancellation option
- Memory efficient

**Real-time Applications:**
- Chatbots and conversational UIs
- Live content generation
- Interactive assistants

### Methods

#### `stream()`

**Definition:** Returns an iterator that yields response chunks as they're generated.

**Returns:** Iterator of message chunks

**When to Use:** For real-time display in CLIs, web UIs, or streaming APIs.

**Signature:**
```python
stream = model.stream("Write a 200 word paragraph on AI.")

for chunk in stream:
    print(chunk.text, end='', flush=True)
```

### Accessing Chunk Data

**Chunk Attributes:**
- `chunk.text` or `chunk.content` - Text content
- `chunk.response_metadata` - Additional info
- `chunk.id` - Chunk identifier

### Streaming Patterns

**Console Output:**
```python
for chunk in model.stream("Long question"):
    print(chunk.text, end='', flush=True)
```

**Web Application (FastAPI/Flask):**
```python
def generate():
    for chunk in model.stream(query):
        yield chunk.text

return StreamingResponse(generate())
```

**Streamlit:**
```python
response_placeholder = st.empty()
full_response = ""

for chunk in model.stream(query):
    full_response += chunk.text
    response_placeholder.markdown(full_response)
```

### Best Practices

- Use `end=''` and `flush=True` for smooth console output
- Handle connection drops in web streaming
- Consider rate limiting for API endpoints
- Buffer chunks if processing is required
- Provide cancellation mechanisms for long streams

---

## 5. Advanced Batch Processing

### Concept

Batch processing optimizes performance by processing multiple inputs concurrently. Advanced batching includes concurrency control and error handling strategies.

### Methods

#### `batch()`

**Definition:** Process multiple inputs in parallel with configurable concurrency.

**Common Parameters:**
- List of inputs (queries, messages, or dictionaries)
- `config` - Configuration dictionary with options

**When to Use:** When you have multiple independent queries to process.

**Signature:**
```python
responses = model.batch([
    "Question 1",
    "Question 2",
    "Question 3"
])
```

### Concurrency Control

#### `max_concurrency` Configuration

**Purpose:** Limit parallel execution to prevent rate limiting or resource exhaustion.

**Common Values:**
- `1` - Sequential processing (no parallelism)
- `2-5` - Conservative for rate-limited APIs
- `10+` - Aggressive for high-throughput scenarios

**Signature:**
```python
responses = model.batch(
    [query1, query2, query3, query4, query5],
    config={'max_concurrency': 2}
)
```

### Use Cases

**Testing:**
- Run test suite against multiple inputs
- Compare responses across models
- Validate behavior with edge cases

**Data Processing:**
- Classify multiple documents
- Extract information from dataset
- Generate summaries for articles

**Multi-user Applications:**
- Process queued requests
- Handle concurrent user queries
- Background job processing

### Performance Considerations

**Rate Limits:**
- Set `max_concurrency` below provider limits
- Implement retry logic for failures
- Monitor API usage

**Memory:**
- Large batches consume more memory
- Consider chunking very large datasets
- Stream results when possible

**Cost:**
- Batch requests may have different pricing
- Monitor token usage across batches
- Consider provider-specific batch APIs

### Error Handling Pattern

```python
responses = []
for query in queries:
    try:
        response = model.invoke(query)
        responses.append({"query": query, "response": response})
    except Exception as e:
        responses.append({"query": query, "error": str(e)})
```

---

## 6. Structured Output

### Concept

Structured output forces models to return data in predefined schemas (Pydantic models, TypedDict, dataclasses). This ensures consistent, parseable responses suitable for downstream processing.

### Why Structured Output?

**Reliability:**
- Guaranteed schema compliance
- Type safety
- Validation

**Integration:**
- Direct database insertion
- API responses
- Data pipelines

**Consistency:**
- Predictable format
- Easier testing
- Reduced parsing errors

### Methods

#### `with_structured_output()`

**Definition:** Configures model to return responses matching a schema.

**Common Parameters:**
- Schema class (Pydantic BaseModel, TypedDict, or dataclass)
- `include_raw` - Boolean to include raw message alongside parsed output

**Returns:** Configured model that outputs structured data

**When to Use:** When you need reliable, structured data extraction or generation.

### Schema Types

#### Pydantic BaseModel

**Definition:** Feature-rich schema with validation, type hints, and documentation.

**When to Use:** Production applications requiring validation and type safety.

**Signature:**
```python
from pydantic import BaseModel, Field

class Movie(BaseModel):
    title: str = Field(description="The title of the movie")
    year: int = Field(description="The release year")
    director: str = Field(description="The director's name")
    rating: float = Field(description="IMDb rating")

model_with_structure = model.with_structured_output(Movie)
response = model_with_structure.invoke("Tell me about Inception")
# Returns Movie instance with validated data
```

**Nested Structures:**
```python
class Actor(BaseModel):
    name: str = Field(..., description="Actor's name")
    birth_year: int = Field(..., description="Birth year")

class MovieDetails(BaseModel):
    title: str = Field(..., description="Movie title")
    cast: list[Actor] = Field(..., description="Main actors")
    genre: str = Field(..., description="Genre")
    budget: int = Field(..., description="Budget in USD")

model_with_nested = model.with_structured_output(MovieDetails)
```

#### TypedDict

**Definition:** Lightweight schema without runtime validation.

**When to Use:** Simple structures where validation overhead isn't needed.

**Signature:**
```python
from typing_extensions import TypedDict, Annotated

class MovieDict(TypedDict):
    """A movie with details."""
    title: Annotated[str, "The title"]
    year: Annotated[int, "Release year"]
    director: Annotated[str, "Director's name"]
    rating: Annotated[float, "IMDb rating"]

model_with_typedict = model.with_structured_output(MovieDict)
response = model_with_typedict.invoke("Tell me about Interstellar")
# Returns dict with specified keys
```

**Nested TypedDict:**
```python
class ActorDict(TypedDict):
    name: Annotated[str, "Actor's name"]
    birth_year: Annotated[int, "Birth year"]

class MovieDetailsDict(TypedDict):
    title: Annotated[str, "Movie title"]
    cast: Annotated[list[ActorDict], "Main actors"]
    genre: Annotated[str, "Genre"]
```

#### Dataclasses

**Definition:** Python standard library dataclass for structured data.

**When to Use:** When you prefer standard library over external dependencies.

**Signature:**
```python
from dataclasses import dataclass

@dataclass
class ContactInfo:
    name: str
    email: str
    phone: str

# Usage with agents
from langchain.agents import create_agent

agent = create_agent(model, response_format=ContactInfo)
result = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "Extract: John Doe, john@example.com, +1-234-567"
    }]
})
print(result["structured_response"])
```

### Including Raw Messages

**Purpose:** Get both structured output and original AI message.

**Signature:**
```python
model_with_structure = model.with_structured_output(
    Movie, 
    include_raw=True
)

response = model_with_structure.invoke("Tell me about Inception")
# Returns: {"raw": AIMessage(...), "parsed": Movie(...)}
```

### Comparison: Pydantic vs TypedDict vs Dataclass

| Feature | Pydantic | TypedDict | Dataclass |
|---------|----------|-----------|-----------|
| Runtime Validation | ✅ Yes | ❌ No | ❌ No |
| Type Hints | ✅ Full | ✅ Full | ✅ Full |
| Nested Structures | ✅ Easy | ✅ Supported | ✅ Supported |
| Documentation | ✅ Field descriptions | ✅ Annotations | ⚠️ Limited |
| Dependencies | Pydantic library | typing_extensions | ✅ Built-in |
| Performance | Slower (validation) | Fast | Fast |
| Best For | Production APIs | Simple schemas | Standard lib preference |

### Best Practices

- Use Pydantic for production applications requiring validation
- Use TypedDict for lightweight, high-performance scenarios
- Use dataclasses when avoiding external dependencies
- Always provide field descriptions for better extraction
- Test schema with edge cases
- Handle validation errors gracefully

---

## 7. Tools

### Concept

Tools extend LLM capabilities by allowing them to execute functions, query APIs, or access external systems. The model decides when and how to use tools based on the user's request.

### How Tools Work

1. **Definition:** Create functions with type hints and descriptions
2. **Binding:** Attach tools to the model
3. **Invocation:** Model receives query and available tools
4. **Decision:** Model decides whether to call tools
5. **Execution:** Your code executes the tool
6. **Result:** Tool output is provided back to model
7. **Response:** Model generates final answer using tool results

### Creating Tools

#### `@tool` Decorator

**Definition:** Converts Python functions into LangChain tools.

**Requirements:**
- Function must have type hints
- Must include docstring (used as tool description)
- Docstring describes what the tool does

**When to Use:** For all tool definitions - simple or complex.

**Signature:**
```python
from langchain.tools import tool

@tool
def get_current_weather(location: str) -> str:
    """Get the current weather in a given location"""
    # Implementation
    return f"Weather in {location}: Sunny, 75°F"

@tool
def get_news_headlines() -> str:
    """Get the latest news headlines"""
    return "Latest news: AI advances, climate summit"
```

### Tool Binding

#### `bind_tools()`

**Definition:** Attaches tools to a model, making them available for invocation.

**Common Parameters:**
- List of tool functions

**Returns:** Model configured with tools

**Signature:**
```python
model_with_tools = model.bind_tools([
    get_current_weather,
    get_news_headlines
])
```

### Tool Calling Flow

#### Basic Pattern

```python
# 1. User query
response = model_with_tools.invoke("What's the weather in NYC?")

# 2. Check for tool calls
for tool_call in response.tool_calls:
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")

# 3. Execute tools (manual)
result = get_current_weather.invoke(tool_call)

# 4. Provide results back to model for final answer
```

### Tool Execution Loop

**Purpose:** Automatically execute tools until model provides final answer.

**Pattern:**
```python
def tool_execution_loop(model, user_query):
    messages = [{"role": "user", "content": user_query}]
    
    while True:
        ai_response = model.invoke(messages)
        messages.append(ai_response)
        
        # No tool calls = final answer ready
        if not ai_response.tool_calls:
            break
        
        # Execute each tool call
        for tool_call in ai_response.tool_calls:
            if tool_call["name"] == "get_current_weather":
                result = get_current_weather.invoke(tool_call)
            elif tool_call["name"] == "get_news_headlines":
                result = get_news_headlines.invoke(tool_call)
            else:
                result = f"Unknown tool: {tool_call['name']}"
            
            messages.append(result)
    
    return ai_response.text
```

### Tool Call Structure

**Tool Call Object:**
```python
{
    "name": "get_current_weather",
    "args": {"location": "New York"},
    "id": "call_abc123"
}
```

**Components:**
- `name` - Tool function name
- `args` - Dictionary of arguments
- `id` - Unique identifier for tracking

### Multiple Tool Usage

**Pattern:** Model can call multiple tools in sequence or parallel.

```python
@tool
def search_database(query: str) -> str:
    """Search the database for information"""
    return f"Found results for: {query}"

@tool
def calculate(expression: str) -> float:
    """Evaluate mathematical expressions"""
    return eval(expression)  # Use safely in production!

@tool
def send_email(to: str, subject: str) -> str:
    """Send an email"""
    return f"Email sent to {to}"

model_with_tools = model.bind_tools([
    search_database,
    calculate,
    send_email
])
```

### Best Practices

**Tool Design:**
- Write clear, specific docstrings
- Use descriptive function names
- Keep tools focused (single responsibility)
- Include type hints for all parameters

**Error Handling:**
- Validate tool inputs
- Handle exceptions gracefully
- Return informative error messages
- Log tool execution for debugging

**Security:**
- Validate and sanitize tool inputs
- Limit tool capabilities appropriately
- Implement access controls
- Avoid dangerous operations (arbitrary code execution)

**Performance:**
- Cache tool results when applicable
- Implement timeouts for slow operations
- Monitor tool usage and costs
- Optimize frequent tool calls

---

## 8. Agents

### Concept

Agents are autonomous LLM-powered systems that can use tools, make decisions, and execute multi-step workflows to accomplish goals. Unlike simple chains, agents can dynamically decide which tools to use and in what order.

### How Agents Work

1. **Goal:** User provides a high-level objective
2. **Planning:** Agent decides what tools/actions to take
3. **Execution:** Agent calls tools and processes results
4. **Reasoning:** Agent evaluates if goal is achieved
5. **Iteration:** Repeat until goal is met or max steps reached
6. **Response:** Agent provides final answer

### Creating Agents

#### `create_agent()`

**Definition:** Creates an autonomous agent with tools and optional system prompt.

**Common Parameters:**
- `model` - Language model to use
- `tools` - List of tool functions
- `system_prompt` - Instructions for agent behavior
- `checkpointer` - Optional state persistence
- `middleware` - Optional middleware for conversation management

**Returns:** Agent executor that can invoke tools autonomously

**When to Use:** When you need autonomous decision-making and multi-step workflows.

**Signature:**
```python
from langchain.agents import create_agent

# Basic agent without tools
agent = create_agent(
    model,
    tools=[],
    system_prompt="You are a helpful assistant."
)

# Agent with tools
agent = create_agent(
    model,
    tools=[get_weather, search_database],
    system_prompt="You are a research assistant."
)
```

### Agent with Tools

**Pattern:**
```python
from langchain.tools import tool

@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"Weather in {location}: Sunny, 75°F"

agent = create_agent(
    model,
    tools=[get_weather],
    system_prompt="You are a helpful weather assistant."
)

# Invoke agent
response = agent.invoke({
    "messages": [{
        "role": "user",
        "content": "What's the weather in New York City?"
    }]
})
```

### Agent Response Structure

**Response Dictionary:**
```python
{
    "messages": [
        HumanMessage(content="What's weather in NYC?"),
        AIMessage(content="", tool_calls=[...]),
        ToolMessage(content="Sunny, 75°F"),
        AIMessage(content="The weather in NYC is sunny...")
    ]
}
```

**Accessing Final Response:**
```python
final_answer = response['messages'][-1].content
```

### Agent vs Chain vs Tool

| Feature | Chain | Tool | Agent |
|---------|-------|------|-------|
| Autonomy | ❌ Fixed flow | ❌ Single function | ✅ Dynamic decisions |
| Multi-step | ✅ Yes | ❌ Single action | ✅ Yes |
| Tool Usage | ⚠️ Manual | ✅ Direct | ✅ Autonomous |
| Reasoning | ⚠️ Limited | ❌ No | ✅ Yes |
| Complexity | Low | Low | High |
| Use Case | Predictable workflows | Single operations | Open-ended tasks |

### When to Use Agents

**Good Use Cases:**
- Research and information gathering
- Multi-step planning tasks
- Dynamic decision-making scenarios
- Tasks requiring tool selection
- Open-ended problem solving

**Not Ideal For:**
- Simple, predictable workflows (use chains)
- Single tool executions (use tools directly)
- Cost-sensitive applications (more LLM calls)
- Latency-critical applications
- Highly deterministic processes

### Agent Limitations

**Reliability:**
- Can make incorrect decisions
- May get stuck in loops
- Requires careful prompt engineering

**Cost:**
- Multiple LLM calls per task
- Higher token usage
- Unpredictable costs

**Latency:**
- Sequential tool execution
- Multiple reasoning steps
- Longer response times

### Best Practices

**Design:**
- Provide clear system prompts
- Limit tool count (5-10 max recommended)
- Use descriptive tool names and docstrings
- Test with diverse inputs

**Control:**
- Implement maximum iteration limits
- Add timeouts for long-running agents
- Monitor tool usage and costs
- Log agent reasoning for debugging

**Optimization:**
- Use faster models for planning
- Cache common tool results
- Parallelize independent operations
- Provide examples in system prompt

---

## 9. Middleware

### Concept

Middleware intercepts and modifies agent workflows, enabling features like conversation summarization, human approval, logging, and state management. Middleware runs between agent steps, providing control and observability.

### Why Middleware?

**Conversation Management:**
- Summarize long conversations
- Prevent token limit overflow
- Maintain context efficiently

**Human Oversight:**
- Require approval for sensitive actions
- Edit tool calls before execution
- Reject inappropriate operations

**Monitoring:**
- Log all agent decisions
- Track tool usage
- Debug agent behavior

**State Management:**
- Persist conversation history
- Implement checkpointing
- Enable conversation recovery

### Middleware Types

#### `SummarizationMiddleware`

**Definition:** Automatically summarizes conversation history when it exceeds specified limits.

**Common Parameters:**
- `model` - Model to use for summarization
- `trigger` - When to trigger summarization (tuple of type and value)
- `keep` - How much to retain after summarization (tuple of type and value)

**Trigger/Keep Types:**
- `("messages", n)` - Based on message count
- `("tokens", n)` - Based on estimated token count
- `("fraction", n)` - Based on fraction of model's context window

**When to Use:** For long-running conversations that approach token limits.

**Signature:**
```python
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model,
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(
            model=model,
            trigger=("messages", 10),  # Summarize after 10 messages
            keep=("messages", 5)        # Keep last 5 messages
        )
    ]
)
```

**Message-Based Trigger:**
```python
SummarizationMiddleware(
    model=model,
    trigger=("messages", 10),
    keep=("messages", 5)
)
```
- Triggers when conversation has 10+ messages
- Keeps most recent 5 messages
- Summarizes older messages

**Token-Based Trigger:**
```python
SummarizationMiddleware(
    model=model,
    trigger=("tokens", 550),
    keep=("tokens", 200)
)
```
- Triggers when ~550 tokens in history
- Keeps ~200 tokens of recent context
- Summarizes the rest

**Fraction-Based Trigger:**
```python
summarize_model = init_chat_model(
    model_name,
    profile={"max_input_tokens": 128000}
)

SummarizationMiddleware(
    model=summarize_model,
    trigger=("fraction", 0.005),  # 0.5% of context window
    keep=("fraction", 0.002)       # 0.2% of context window
)
```
- Triggers at 0.5% of model's 128k token limit (640 tokens)
- Keeps 0.2% (256 tokens)
- Dynamic based on model capabilities

#### `HumanInTheLoopMiddleware`

**Definition:** Pauses agent execution to request human approval, editing, or rejection of tool calls.

**Common Parameters:**
- `interrupt_on` - Dictionary mapping tool names to approval settings

**Approval Settings:**
- `allowed_decisions` - List of options: `["approve", "edit", "reject"]`
- `False` - Never interrupt for this tool
- `True` or dict - Interrupt for approval

**When to Use:** For sensitive operations requiring human oversight (sending emails, financial transactions, data deletion).

**Signature:**
```python
from langchain.agents.middleware import HumanInTheLoopMiddleware

agent = create_agent(
    model,
    tools=[read_email_tool, send_email_tool],
    checkpointer=InMemorySaver(),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email_tool": {
                    "allowed_decisions": ["approve", "edit", "reject"]
                },
                "read_email_tool": False  # Never interrupt
            }
        )
    ]
)
```

### Human-in-the-Loop Flow

**Step 1: Agent Execution (Interrupted)**
```python
config = {"configurable": {"thread_id": "user-123"}}

result = agent.invoke(
    {
        "messages": [{
            "role": "user",
            "content": "Send email to john@example.com"
        }]
    },
    config=config
)
```

**Step 2: Check for Interruption**
```python
if "__interrupt__" in result:
    print("Human intervention required")
    print(f"Details: {result['__interrupt__']}")
```

**Step 3: Resume with Decision**
```python
from langgraph.types import Command

result = agent.invoke(
    Command(
        resume={
            "decisions": [{"type": "approve"}]
        }
    ),
    config=config
)
```

**Decision Types:**
- `{"type": "approve"}` - Execute tool as planned
- `{"type": "reject"}` - Cancel tool execution
- `{"type": "edit", "args": {...}}` - Modify tool arguments

### Checkpointing

**Purpose:** Persist conversation state across interruptions and sessions.

**InMemorySaver:**
```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(
    model,
    checkpointer=InMemorySaver(),
    middleware=[...]
)
```

**Thread IDs:**
- Unique identifier for each conversation
- Required for stateful interactions
- Enables conversation recovery

```python
config = {"configurable": {"thread_id": "unique-id"}}
```

### Combining Middleware

**Multiple Middleware:**
```python
agent = create_agent(
    model,
    tools=[...],
    checkpointer=InMemorySaver(),
    middleware=[
        SummarizationMiddleware(...),
        HumanInTheLoopMiddleware(...)
    ]
)
```

**Execution Order:** Middleware executes in list order.

### Best Practices

**Summarization:**
- Choose trigger points based on model context limits
- Keep enough context for coherent conversations
- Test summarization quality with real conversations
- Use faster/cheaper models for summarization

**Human-in-the-Loop:**
- Only interrupt for truly sensitive operations
- Provide clear context in interrupt messages
- Implement timeout handling
- Log all human decisions

**Checkpointing:**
- Use persistent storage (database, Redis) in production
- Implement cleanup for old sessions
- Handle checkpoint recovery gracefully
- Test with simulated failures

**General:**
- Monitor middleware performance impact
- Keep middleware logic simple
- Test edge cases thoroughly
- Document middleware behavior

---

## Summary

This guide covers advanced LangChain concepts:

1. **Model Integration:** Unified interfaces across providers (Azure, Google, Groq)
2. **Advanced Messages:** Rich message types with metadata and tool interactions
3. **Streaming:** Real-time response generation for better UX
4. **Batch Processing:** Concurrent processing with concurrency control
5. **Structured Output:** Reliable data extraction with Pydantic, TypedDict, dataclasses
6. **Tools:** Extending LLM capabilities with external functions and APIs
7. **Agents:** Autonomous systems for multi-step reasoning and tool orchestration
8. **Middleware:** Conversation management, human oversight, and state persistence

### Learning Path

1. Master model integration for provider flexibility
2. Understand advanced message types and metadata
3. Implement streaming for better user experience
4. Use batch processing for efficient multi-query handling
5. Create structured outputs for reliable data extraction
6. Build tools to extend LLM capabilities
7. Develop agents for autonomous task execution
8. Apply middleware for production-ready conversation management

### Best Practices

**Model Integration:**
- Use `init_chat_model()` for provider flexibility
- Set appropriate API versions
- Handle provider-specific errors
- Test across multiple providers

**Messages:**
- Always include descriptive system messages
- Use metadata for tracking and debugging
- Match tool_call_id between AI and Tool messages
- Build conversation history carefully

**Streaming:**
- Use for responses over 100 tokens
- Implement proper buffering
- Handle connection failures
- Provide cancellation options

**Batch Processing:**
- Set `max_concurrency` based on rate limits
- Implement retry logic
- Monitor token usage and costs
- Handle partial failures gracefully

**Structured Output:**
- Use Pydantic for production (validation)
- Use TypedDict for performance-critical paths
- Provide detailed field descriptions
- Test with edge cases
- Handle validation errors

**Tools:**
- Write clear, specific docstrings
- Keep tools focused and simple
- Validate inputs thoroughly
- Implement timeouts and error handling
- Log tool executions

**Agents:**
- Provide clear system prompts with examples
- Limit tool count (5-10 recommended)
- Set maximum iterations
- Monitor costs and latency
- Use for truly autonomous tasks only

**Middleware:**
- Choose summarization triggers wisely
- Only interrupt for sensitive operations
- Use persistent checkpointers in production
- Test recovery scenarios
- Monitor performance impact

### When to Use What

**Chains vs Tools vs Agents:**
- **Chains:** Predictable, fixed workflows
- **Tools:** Single-purpose operations
- **Agents:** Open-ended, multi-step tasks

**Pydantic vs TypedDict vs Dataclass:**
- **Pydantic:** Production APIs needing validation
- **TypedDict:** High-performance, simple schemas
- **Dataclass:** Standard library preference

**Streaming vs Batch:**
- **Streaming:** Single long response, real-time UX
- **Batch:** Multiple independent queries, efficiency

**Model Providers:**
- **Azure OpenAI:** Enterprise, compliance, scaling
- **Google Gemini:** Multimodal, latest features
- **Groq:** Fast inference, open models
- **Decision:** Use `init_chat_model()` for flexibility

---

**End of Guide**
