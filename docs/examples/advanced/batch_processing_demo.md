# Batch Processing Demo

This example demonstrates how to use FastADK's batch processing capabilities to efficiently process multiple inputs with AI agents. Batch processing is essential for applications that need to handle large volumes of data or requests.

## Overview

The Batch Processing Demo shows how to:

1. Process multiple inputs efficiently with parallel execution
2. Configure parallelism with adjustable batch sizes
3. Monitor and report progress during batch operations
4. Apply post-processing to aggregate results
5. Handle errors gracefully in batch contexts

## Key Features

### Batch Processor

The example uses FastADK's `BatchProcessor` to manage batch operations:

```python
from fastadk.core.batch import BatchProcessor
from sentiment_analysis_agent import SentimentAnalysisAgent

# Create an agent for sentiment analysis
agent = SentimentAnalysisAgent()

# Create a batch processor with the agent
processor = BatchProcessor(
    agent=agent,
    max_concurrent=5,  # Process up to 5 items in parallel
    timeout=60,        # Timeout after 60 seconds per item
    retry_attempts=2   # Retry failed items up to 2 times
)
```

### Parallel Processing

The example demonstrates both sequential and parallel approaches:

```python
# Sequential processing (one at a time)
async def process_sequential(texts):
    results = []
    for text in texts:
        result = await agent.run(f"Analyze the sentiment of: {text}")
        results.append(result)
    return results

# Parallel processing with batch processor
async def process_parallel(texts):
    results = await processor.process_batch(
        items=texts,
        process_fn=lambda text: f"Analyze the sentiment of: {text}",
        result_fn=lambda response, text: {
            "text": text,
            "sentiment": extract_sentiment(response),
            "score": extract_score(response)
        }
    )
    return results
```

### Progress Monitoring

The example shows how to track progress during batch operations:

```python
async def process_with_progress(texts):
    # Create a progress tracker
    progress = {"processed": 0, "total": len(texts)}
    
    # Define progress callback
    def update_progress(result, item):
        progress["processed"] += 1
        print(f"Progress: {progress['processed']}/{progress['total']} ({progress['processed']/progress['total']*100:.1f}%)")
        return result
    
    # Process batch with progress tracking
    results = await processor.process_batch(
        items=texts,
        process_fn=lambda text: f"Analyze the sentiment of: {text}",
        result_fn=update_progress
    )
    
    return results
```

### Error Handling

The example demonstrates how to handle errors during batch processing:

```python
async def process_with_error_handling(texts):
    results = []
    errors = []
    
    # Define error handler
    def handle_error(error, item):
        errors.append({"item": item, "error": str(error)})
        return None  # Return None for failed items
    
    # Process batch with error handling
    batch_results = await processor.process_batch(
        items=texts,
        process_fn=lambda text: f"Analyze the sentiment of: {text}",
        error_fn=handle_error
    )
    
    # Filter out None results (from errors)
    results = [r for r in batch_results if r is not None]
    
    return {
        "successful": results,
        "failed": errors,
        "success_rate": len(results) / len(texts) if texts else 0
    }
```

### Performance Comparison

The example includes code to compare the performance of sequential vs. parallel processing:

```python
import time
import asyncio

async def performance_comparison(texts):
    # Measure sequential processing time
    start_time = time.time()
    sequential_results = await process_sequential(texts)
    sequential_time = time.time() - start_time
    
    # Measure parallel processing time
    start_time = time.time()
    parallel_results = await process_parallel(texts)
    parallel_time = time.time() - start_time
    
    # Calculate speedup
    speedup = sequential_time / parallel_time if parallel_time > 0 else float('inf')
    
    return {
        "sequential_time": sequential_time,
        "parallel_time": parallel_time,
        "speedup": speedup,
        "sequential_results": len(sequential_results),
        "parallel_results": len(parallel_results)
    }
```

## Implementation Details

### Sentiment Analysis Agent

The example uses a sentiment analysis agent:

```python
from fastadk import Agent, BaseAgent, tool
from typing import Dict

@Agent(model="gemini-1.5-pro")
class SentimentAnalysisAgent(BaseAgent):
    @tool
    def analyze_sentiment(self, text: str) -> Dict:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary containing sentiment (positive, negative, neutral) and score
        """
        # This is a tool definition - the LLM will implement the analysis
        pass
```

### Batch Configuration

The example demonstrates how to configure batch processing parameters:

```python
# Configure different batch sizes to find optimal performance
async def optimize_batch_size(texts, batch_sizes=[1, 5, 10, 20]):
    results = {}
    
    for batch_size in batch_sizes:
        # Create processor with current batch size
        processor = BatchProcessor(
            agent=SentimentAnalysisAgent(),
            max_concurrent=batch_size
        )
        
        # Measure processing time
        start_time = time.time()
        await processor.process_batch(
            items=texts,
            process_fn=lambda text: f"Analyze the sentiment of: {text}"
        )
        processing_time = time.time() - start_time
        
        # Store result
        results[batch_size] = {
            "batch_size": batch_size,
            "processing_time": processing_time,
            "items_per_second": len(texts) / processing_time
        }
    
    # Find optimal batch size
    optimal_batch_size = max(results.items(), key=lambda x: x[1]["items_per_second"])[0]
    
    return {
        "optimal_batch_size": optimal_batch_size,
        "all_results": results
    }
```

### Results Aggregation

The example shows how to aggregate results from batch processing:

```python
async def analyze_and_aggregate(texts):
    # Process all texts in batch
    results = await processor.process_batch(
        items=texts,
        process_fn=lambda text: f"Analyze the sentiment of: {text}",
        result_fn=lambda response, text: {
            "text": text,
            "sentiment": extract_sentiment(response),
            "score": extract_score(response)
        }
    )
    
    # Aggregate results
    sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
    average_score = 0
    
    for result in results:
        sentiment_counts[result["sentiment"]] += 1
        average_score += result["score"]
    
    if results:
        average_score /= len(results)
    
    # Return aggregated analysis
    return {
        "total_analyzed": len(results),
        "sentiment_distribution": sentiment_counts,
        "average_score": average_score,
        "detailed_results": results
    }
```

## Usage Example

Here's how you might use the batch processing capabilities:

```python
from batch_processing_demo import analyze_and_aggregate

async def main():
    # Sample texts to analyze
    texts = [
        "I absolutely love this product! It's amazing.",
        "This is the worst experience I've ever had.",
        "The service was okay, nothing special.",
        "I'm very disappointed with the quality.",
        "The customer support team was incredibly helpful.",
        # ... more texts
    ]
    
    # Process and aggregate results
    results = await analyze_and_aggregate(texts)
    
    # Print summary
    print(f"Analyzed {results['total_analyzed']} texts")
    print(f"Sentiment distribution: {results['sentiment_distribution']}")
    print(f"Average sentiment score: {results['average_score']:.2f}")
    
    # Optimize batch size
    optimization = await optimize_batch_size(texts)
    print(f"Optimal batch size: {optimization['optimal_batch_size']}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

## Full Source Code

For the complete implementation of the Batch Processing Demo, refer to the [batch_processing_demo.py](https://github.com/Mathews-Tom/FastADK/blob/main/examples/advanced/batch_processing_demo.py) file in the examples directory.

## Key Takeaways

The Batch Processing Demo example demonstrates several important patterns:

1. **Parallel Processing**: Using concurrency to improve throughput
2. **Resource Management**: Controlling concurrency to avoid overwhelming resources
3. **Progress Tracking**: Monitoring batch operations in real-time
4. **Error Handling**: Gracefully managing failures in batch contexts
5. **Performance Optimization**: Finding the optimal batch size for best performance

By using these patterns, you can efficiently process large volumes of data with AI agents, making your applications more scalable and responsive.
