# Task 1: `calculate_cps()` helper

**Files:** Modify `translator/engine.py` (insert before the `ConversationMemory` class)

## Requirements

Add a pure-utility function:

```python
def calculate_cps(text: str, duration_ms: float) -> float:
    chars = len(text)
    seconds = duration_ms / 1000.0
    return chars / seconds if seconds > 0 else 0.0
```

- text: the caption text
- duration_ms: caption duration in milliseconds
- returns: characters per second (float)
- No dependencies beyond stdlib
- No test file needed — validated implicitly by Task 3's CPS fallback

## Exact insertion point

In `engine.py`, find the `ConversationMemory` class. Insert `calculate_cps` just before it.
