# Task 2: `merge_sentence_blocks()` — Report

## What I Implemented

Added `merge_sentence_blocks()` function at `translator/engine.py:2324` that merges consecutive captions into sentence blocks for better translation context.

The function:
- Takes `subs` (pysrt.SubRipFile), `merge_gap_ms=500`, `max_block_tokens=96`
- Returns `list[dict]` with `indices`, `text`, `durations_ms`
- Merges captions when: no end punctuation, no SFX bracket, no episode marker, gap < merge_gap_ms, not multi-speaker, token count within limit
- Uses `re` (already imported at line 14)

## How I Verified It

- `py -c "import ast; ast.parse(open('translator/engine.py', encoding='utf-8').read()); print('Syntax OK')"` — syntax valid
- Visually confirmed placement: after `calculate_cps()` (line 2313), before `ConversationMemory` (line 2370)

## Files Changed

- `translator/engine.py` — 46-line insertion (lines 2319–2364)

## Self-Review Findings

- Code matches the brief exactly — no deviations
- All merge conditions from brief are implemented
- Section numbering follows existing convention (4c after 4b)

## Issues or Concerns

None.
