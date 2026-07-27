# Task 3 Report: `split_sentence_blocks()`

## What I Implemented

Added `split_sentence_blocks()` function to `translator/engine.py` at line 2366, immediately after `merge_sentence_blocks()` (ends at line 2363) and before the `# 5a. Lightweight conversation memory` section header.

The function distributes translated merged-block text back to individual per-caption slots:

1. **Single-index blocks** — direct passthrough from `ger_blocks` (or original `blk["text"]` as fallback)
2. **Multi-index blocks** — splits translated text using ratio-based position computation (proportional to source text lengths), snaps to word boundaries, then applies CPS check on each part (fallback to original block text if any chunk > 35 CPS)
3. **Edge cases** — empty source text (distributes same text to all), short `ger_blocks` list (uses original text), zero `total_src` (assigns ger_text to all)

## How I Verified It

- Syntax checked: `py -m py_compile translator\engine.py` — clean
- Smoke tested via project venv with 5 scenarios:
  - Single-block passthrough
  - Multi-block ratio split with word-boundary snapping
  - Fallback when `ger_blocks` shorter than `merge_blocks`
  - CPS fallback (high-CPS chunk triggers revert to original text)
  - Empty source text edge case

## Files Changed

- `translator/engine.py` — inserted 56 lines (function body) at line 2366

## Self-Review Findings

- The code matches the spec exactly — no deviations
- `calculate_cps()` is already defined at line 2313 and accessible at module scope
- No new imports needed (`re` already imported)
- The `src_text` variable on line 2375 is assigned but not used in the actual split logic (it's used indirectly: original spec keeps it for clarity); removing it would be a deviation from the spec — kept as-is
- CPS fallback correctly checks each individual part against its own caption window duration

## Issues or Concerns

- No `total_src == 0` check exists in the current implementation (added one per the spec logic — the spec code includes it in the exact implementation). Actually this is present in the spec.
- Initial smoke test used wrong CPS threshold expectation (50ms window for a 1-char chunk = 20 CPS, which passed) — corrected the test to use a shorter second window (10ms) to reliably trigger fallback
