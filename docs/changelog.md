# Changelog

For a complete list of changes, please refer to the [CHANGELOG.md](https://github.com/fastadk/blob/main/CHANGELOG.md) file in the repository root.

## [0.2.0] - 2025-07-09

### Added

#### Token and Cost Tracking

- Added `TokenUsage` data class to record prompt, completion, and total tokens
- Extended each provider client to capture usage from responses
- Created `CostCalculator` utility for estimating costs based on token usage
- Added `TokenBudget` component with configurable budgets and alerts
- Added structured logging for token usage and costs
- Added Prometheus metrics for token usage and cost estimation

#### Context Management and Memory Improvements

- Added `ContextPolicy` abstract class with built-in policies (MostRecent, SummarizeOlder, HybridVectorRetrieval)
- Implemented `SummarizationService` for LLM-based conversation summarization
- Added vector store backend integration with `VectorMemoryBackend`
- Implemented persistent memory backends for Redis and SQL
- Added configuration options for memory policies

#### Scalability and Performance Optimizations

- Improved async execution for all model and tool calls
- Enhanced `Workflow` class to support concurrent sub-tasks
- Implemented `CacheManager` with in-memory LRU and Redis options
- Added lazy tool execution logic to improve efficiency
- Created `BatchUtils` for common bulk operations

#### Extensibility and Integration

- Defined `ModelProviderABC` interface for custom providers
- Implemented `PluginManager` for discovering and loading custom modules
- Added Slack and Discord adapters for integration
- Added multi-agent orchestration API
- Included fine-tuning helper module

#### Developer Experience and Tooling

- Extended CLI with `repl`, `init`, and `config validate` commands
- Added verbose debugging with chain-of-thought capture
- Created `MockLLM` and `MockTool` classes for testing
- Added test scenario decorators for structured testing
- Added VSCode snippets and configuration support
- Implemented config reload endpoint

#### Observability and Monitoring

- Added structured JSON logging for all agent events
- Integrated OpenTelemetry for traces and spans
- Added Prometheus metrics endpoint
- Implemented redaction filter for sensitive data

#### UI/UX and Documentation

- Added cookbook, plugin guide, and performance tuning documentation
- Expanded examples folder with new agent templates
- Implemented project scaffolding with `fastadk init`
- Added Streamlit chat UI example
- Added community feedback links

### Changed

- Improved async performance and error handling
- Enhanced memory management for long conversations
- Optimized token usage and cost calculations
- Upgraded provider integrations with the latest APIs

### Fixed

- Various bug fixes and stability improvements
- Fixed memory leaks in long-running conversations
- Addressed race conditions in async workflows
- Improved error reporting and exception handling

## [0.1.0] - Initial Release

- First public release of FastADK
- Basic agent functionality with OpenAI, Anthropic, and Google support
- Memory backend for conversations
- Tool integration system
- Command-line interface
