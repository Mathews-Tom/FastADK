# Memory Base Classes

FastADK provides a flexible memory system with various backend options. This page documents the base classes and interfaces for memory management.

## Memory Backend Base Class

::: fastadk.memory.base.MemoryBackend

## Memory Entry

::: fastadk.memory.base.MemoryEntry

## Usage Example

```python
from fastadk.memory.base import MemoryBackend, MemoryEntry
from typing import Dict, List, Optional, Any
import time

class CustomMemoryBackend(MemoryBackend):
    """A custom memory backend implementation."""
    
    def __init__(self):
        self.storage = {}
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Store a value with optional TTL."""
        expiry = time.time() + ttl if ttl else None
        self.storage[key] = {
            "value": value,
            "expiry": expiry
        }
        
    async def get(self, key: str) -> Any:
        """Retrieve a value by key."""
        if key not in self.storage:
            return None
            
        item = self.storage[key]
        if item["expiry"] and time.time() > item["expiry"]:
            del self.storage[key]
            return None
            
        return item["value"]
        
    async def delete(self, key: str) -> None:
        """Delete a value by key."""
        if key in self.storage:
            del self.storage[key]
            
    async def exists(self, key: str) -> bool:
        """Check if a key exists."""
        if key not in self.storage:
            return False
            
        item = self.storage[key]
        if item["expiry"] and time.time() > item["expiry"]:
            del self.storage[key]
            return False
            
        return True
        
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching a pattern."""
        # Simple implementation without pattern matching
        return list(self.storage.keys())
        
    async def clear(self) -> None:
        """Clear all values."""
        self.storage.clear()
        
    async def clear_pattern(self, pattern: str) -> None:
        """Clear all values matching a pattern."""
        # Simple implementation without pattern matching
        keys_to_delete = [k for k in self.storage.keys() if pattern in k]
        for key in keys_to_delete:
            del self.storage[key]
            
    async def ttl(self, key: str) -> Optional[int]:
        """Get the remaining TTL for a key."""
        if key not in self.storage or not self.storage[key]["expiry"]:
            return None
            
        remaining = self.storage[key]["expiry"] - time.time()
        return max(0, int(remaining))
        
    async def health_check(self) -> bool:
        """Check if the memory backend is healthy."""
        return True
```
