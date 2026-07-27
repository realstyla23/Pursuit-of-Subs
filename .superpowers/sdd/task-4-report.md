# Task 4 Report: Wire sentence_aware into translate_fast(), Config, subtranslate.py, __init__.py

## Changes Made

### 1. `translator/engine.py` — Config dataclass (line 85-86)
Added `sentence_aware: bool = False` and `merge_gap_ms: int = 500` after `polish_parallel`.

### 2. `translator/engine.py` — translate_fast() merge step (after line 2054)
Inserted merge block: calls `merge_sentence_blocks(subs, cfg.merge_gap_ms)` if `cfg.sentence_aware` is True, replaces `eng_texts` with merged block texts, updates `n`.

### 3. `translator/engine.py` — translate_fast() split step (after line 2271)
Inserted split block: calls `split_sentence_blocks(ger_texts, merge_blocks, subs)` if `cfg.sentence_aware and merge_blocks`, updates `ger_texts` and `n`.

### 4. `subtranslate.py` — CLI arguments (lines 52-55)
Added `--sentence-aware` (store_true) and `--merge-gap-ms` (int, default=500) after `--polish-parallel`.

### 5. `subtranslate.py` — Config construction (lines 140-141)
Added `sentence_aware=a.sentence_aware` and `merge_gap_ms=a.merge_gap_ms` to `Config(...)` call.

### 6. `translator/__init__.py` — Imports (line 28)
Added `calculate_cps, merge_sentence_blocks, split_sentence_blocks` to the import from `engine`.

### 7. `translator/__init__.py` — `__all__` (line 65)
Added `"calculate_cps", "merge_sentence_blocks", "split_sentence_blocks"` to the list.

## Verification
- **AST parse**: all three files (`engine.py`, `subtranslate.py`, `__init__.py`) parse without syntax errors.
- **Import test**: `from translator.engine import Config, calculate_cps, merge_sentence_blocks, split_sentence_blocks` succeeds. Config fields include `sentence_aware` and `merge_gap_ms`. All three new names present in `__all__`.
- **No pytest** — no test specifically covers these integration points in this session.

## Concerns
- The `calculate_cps` function was already defined in engine.py and is now exported via `__init__.py` as the task brief requested. It's not directly used by the new code paths but is correctly exported alongside the other two.
