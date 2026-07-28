# Task P0-1 Report: Checkpoint validation

## What was implemented

1. **`_save_checkpoint()`** — Added `sentence_aware: bool` and `merge_gap_ms: int` parameters (both defaulting to 0/False for backward compat), included both in the checkpoint dict.

2. **`_load_checkpoint()`** — After existing input/batch_size/version validation, added a loop that checks loaded values against `cfg.sentence_aware` and `cfg.merge_gap_ms`. Logs: (a) info on legacy checkpoints missing keys, (b) warning on mismatch. Always uses config values (never aborts).

3. **Three call sites** in `translate_fast()` (lines 2449, 2453, 2459) — Passed `cfg.sentence_aware` and `cfg.merge_gap_ms` to `_save_checkpoint()`.

## Testing

- `python -c "from translator.engine import _save_checkpoint, _load_checkpoint; print('OK')"` — passed (both system py launcher and venv)

## Files changed

- `translator/engine.py` — 16 insertions, 4 deletions

## Self-review

- All checkpoints now persist the two config fields used during translation
- Legacy checkpoints (missing keys) handled gracefully with info message
- Mismatches warned but never block loading — always favors config
- Call sites pass the correct `cfg.*` attributes
- Function signature defaults preserve backward compat for any external callers

## Concerns

None.
