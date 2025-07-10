# Mermaid Diagrams Test

This page demonstrates how Mermaid diagrams are now rendered in the FastADK documentation.

## Flowchart Example

```mermaid
flowchart TD
    A[Start] --> B{Is FastADK installed?}
    B -->|Yes| C[Create Agent]
    B -->|No| D[Install FastADK]
    D --> C
    C --> E[Add Tools]
    E --> F[Run Agent]
    F --> G[Process Results]
    G --> H[End]
```

## Sequence Diagram Example

```mermaid
sequenceDiagram
    participant User
    participant Agent
    participant Tool
    participant LLM

    User->>Agent: Send prompt
    Agent->>LLM: Generate response
    LLM-->>Agent: Response with tool calls
    Agent->>Tool: Execute tool
    Tool-->>Agent: Tool result
    Agent->>LLM: Generate final response
    LLM-->>Agent: Final response
    Agent->>User: Return response
```

## Class Diagram Example

```mermaid
classDiagram
    class BaseAgent {
        +tools: List[Tool]
        +memory: MemoryBackend
        +run(input: str): str
        +register_tool(tool: Tool): void
    }
    
    class Tool {
        +name: str
        +description: str
        +execute(*args): Any
    }
    
    class MemoryBackend {
        <<abstract>>
        +get(key: str): Any
        +set(key: str, value: Any): void
        +search_semantic(query: str): List[Any]
    }
    
    BaseAgent "1" *-- "many" Tool : has
    BaseAgent "1" *-- "1" MemoryBackend : uses
    
    class VectorMemoryBackend {
        +search_semantic(query: str): List[Any]
    }
    
    MemoryBackend <|-- VectorMemoryBackend
```

## State Diagram Example

```mermaid
stateDiagram-v2
    [*] --> Initialization
    Initialization --> Ready
    Ready --> Processing: Receive prompt
    Processing --> ToolExecution: Needs tools
    Processing --> Responding: Direct response
    ToolExecution --> Processing: Tool completed
    Responding --> Ready: Response sent
    Ready --> [*]: Shutdown
```

## Entity Relationship Diagram

```mermaid
erDiagram
    AGENT ||--o{ TOOL : uses
    AGENT ||--|| MEMORY : has
    MEMORY ||--o{ MEMORY_ENTRY : contains
    AGENT {
        string name
        string model
        string provider
    }
    TOOL {
        string name
        string description
        function execute
    }
    MEMORY {
        string type
        int capacity
    }
    MEMORY_ENTRY {
        string key
        json data
        timestamp created_at
    }
```