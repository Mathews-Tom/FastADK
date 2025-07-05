# RFC: Semantic Memory

- **RFC Number**: 0001
- **Author(s)**: FastADK Team
- **Status**: Accepted
- **Created**: 2025-05-01
- **Last Updated**: 2025-05-15
- **Related Issues**: #42, #57

## Summary

This RFC proposes adding semantic memory capabilities to FastADK, enabling agents to store, retrieve, and search information based on meaning rather than exact matches. This will allow for more natural conversations, better context awareness, and enhanced long-term memory.

## Motivation

Current memory implementations in FastADK support basic conversation history but lack the ability to:

1. Find related information based on semantic similarity
2. Automatically summarize conversation history
3. Prioritize relevant information in context windows

These limitations make it difficult to build agents that can maintain coherent conversations over extended interactions or recall information from past sessions in a meaningful way.

## Detailed Design

We propose implementing a semantic memory system with the following components:

### 1. Vector Storage Interface

```python
class VectorStore(ABC):
    @abstractmethod
    async def add_vector(self, id: str, vector: List[float], metadata: Dict) -> None:
        """Store a vector with its metadata."""
        pass
    
    @abstractmethod
    async def search(self, query_vector: List[float], limit: int = 10) -> List[Dict]:
        """Search for similar vectors."""
        pass
    
    @abstractmethod
    async def delete(self, id: str) -> None:
        """Delete a vector by ID."""
        pass
```

### 2. Implementations for Different Backends

```python
class RedisVectorStore(VectorStore):
    """Vector storage using Redis with RediSearch."""
    # Implementation...

class PineconeVectorStore(VectorStore):
    """Vector storage using Pinecone."""
    # Implementation...

class ChromaVectorStore(VectorStore):
    """Vector storage using Chroma."""
    # Implementation...
```

### 3. Enhanced Memory Manager

```python
class SemanticMemoryManager:
    """Manages semantic memory for agents."""
    
    def __init__(self, vector_store: VectorStore, embedding_model: str = "text-embedding-ada-002"):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
    
    async def add_memory(self, text: str, metadata: Dict) -> str:
        """Add text to semantic memory."""
        # Generate embeddings
        # Store in vector store
        
    async def search_memory(self, query: str, limit: int = 10) -> List[Dict]:
        """Search semantic memory for relevant information."""
        # Generate query embedding
        # Search vector store
        
    async def summarize_context(self, session_id: str) -> str:
        """Generate a summary of the conversation context."""
        # Retrieve recent messages
        # Generate summary with LLM
```

### 4. Agent Integration

```python
@Agent(
    model="gemini-2.0-pro",
    memory_backend="redis",
    semantic_memory=True,  # Enable semantic memory
    semantic_memory_config={
        "vector_store": "redis",  # Options: redis, pinecone, chroma
        "embedding_model": "text-embedding-ada-002",
        "summary_interval": 10,  # Summarize every 10 messages
        "similarity_threshold": 0.7  # Minimum similarity score
    }
)
class AgentWithSemanticMemory(BaseAgent):
    @tool
    def search_memories(self, query: str) -> List[Dict]:
        """Search through semantic memories."""
        return self.memory_manager.search_memory(query)
    
    @tool
    def save_important_info(self, info: str, tags: List[str] = None) -> None:
        """Explicitly save important information to semantic memory."""
        self.memory_manager.add_memory(info, {"tags": tags or []})
```

### 5. Context Management

Semantic memory will be used to enhance context windows by:

1. Automatically summarizing older context to save tokens
2. Retrieving relevant past information based on current conversation
3. Prioritizing important information in the context window

## Alternatives Considered

1. **Simple keyword search**: Less sophisticated but easier to implement. Rejected because it wouldn't capture semantic meaning.
2. **Third-party memory service**: Could use LangChain or LlamaIndex memory. Rejected to avoid additional dependencies.
3. **Document-based storage**: Using MongoDB or similar. Rejected due to lack of vector search capabilities.

## API Impact

New public APIs:

- `SemanticMemoryManager` class for direct memory manipulation
- New options in `@Agent` decorator for semantic memory configuration
- Enhanced `memory_backend` options to include vector stores
- New methods on `BaseAgent` for semantic memory operations

## Implementation Plan

1. **Phase 1**: Implement core vector store interface and Redis implementation (2 weeks)
2. **Phase 2**: Add embedding generation and memory manager (2 weeks) 
3. **Phase 3**: Integrate with agent context management (1 week)
4. **Phase 4**: Add summarization capabilities (1 week)
5. **Phase 5**: Additional vector store implementations (2 weeks)

## Migration Path

This feature will be backward compatible with existing agents. To migrate:

1. Enable semantic memory in the `@Agent` decorator
2. Configure a vector store backend
3. Optionally implement custom memory tools

## Risks and Concerns

1. **Performance**: Vector operations can be computationally expensive
2. **Storage requirements**: Embeddings will increase memory usage
3. **API costs**: Generating embeddings will incur additional API costs
4. **Complexity**: Adds significant complexity to the memory system

## Open Questions

1. Should we implement our own embedding generation or rely on external services?
2. What is the optimal approach for context summarization?
3. How should we handle versioning of embedding models?
