# Task 1: `calculate_cps()` helper — Report

## What was implemented

Added `calculate_cps(text, duration_ms)` pure function to `translator/engine.py`, inserted just before the `ConversationMemory` class (section 4b). Accepts caption text and duration in milliseconds, returns characters-per-second as float. Returns 0.0 for empty text or zero duration.

## How verified

- Loaded module from venv: `from engine import calculate_cps`
- Tested: `("Hello", 2000.0)` → 2.5, `("", 1000.0)` → 0.0, `("Hello", 0.0)` → 0.0, `("Hello world!", 2000.0)` → 6.0

## Files changed

- `translator/engine.py` — 11 lines added

## Self-review findings

None. Function is trivially correct per spec.

## Issues or concerns

None.
