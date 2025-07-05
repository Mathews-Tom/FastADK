---
name: Bug report
about: Create a report to help us improve FastADK
title: '[BUG] '
labels: bug
assignees: ''
---

## Bug Description

A clear and concise description of what the bug is.

## Reproduction Steps

Steps to reproduce the behavior:

1. Install '...'
2. Create file with '...'
3. Run command '...'
4. See error

## Expected Behavior

A clear and concise description of what you expected to happen.

## Actual Behavior

What actually happened, including error messages, stack traces, or unexpected results.

## Environment

- FastADK version: [e.g. 0.1.0]
- Python version: [e.g. 3.10.6]
- OS: [e.g. Ubuntu 22.04, macOS 12.6, Windows 11]
- Installation method: [e.g. pip, from source]
- Any other relevant environment details

## Code Samples

```python
# If applicable, add code samples to help explain your problem
from fastadk.core import Agent, BaseAgent, tool

@Agent(model="gemini-2.0-pro")
class MyAgent(BaseAgent):
    # ...
```

## Error Messages/Logs

```
Paste any error messages or logs here
```

## Possible Solution

If you have any ideas about what might be causing the issue or how to fix it, please share them here.

## Additional Context

Add any other context about the problem here, such as when it started happening or if you've found any workarounds.
