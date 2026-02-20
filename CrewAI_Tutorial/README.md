# CrewAI Tutorial

⚠️ **WARNING: This project is not currently working**

## Issue

The `YoutubeChannelSearchTool` is failing during initialization when attempting to embed documents using Azure OpenAI embeddings.

### Error Details

```
openai.AuthenticationError: Error code: 401 - Incorrect API key provided
```

**Root Cause:** The embedder configuration is not properly routing to Azure OpenAI. Instead, it's attempting to use OpenAI's standard API endpoint, which rejects the Azure API key.

### Stack Trace

The error occurs in the RAG adapter when trying to upsert documents to ChromaDB:
- `crewai_tools/adapters/crewai_rag_adapter.py` → `add_documents()`
- `chromadb/api/models/Collection.py` → `upsert()`
- `chromadb/utils/embedding_functions/openai_embedding_function.py` → embedding creation fails with 401 auth error

## Next Steps

1. Verify Azure OpenAI embedder is properly configured in CrewAI
2. Check if `api_type="azure"` is being respected by the embedder
3. Consider using LangChain's Azure OpenAI embeddings directly instead of relying on CrewAI's config
4. Review CrewAI version compatibility with Azure OpenAI provider

## Files

- `tools.py` - Tool definitions (broken)
- `agents.py` - Agent configurations
- `tasks.py` - Task definitions
- `crew.py` - Crew orchestration