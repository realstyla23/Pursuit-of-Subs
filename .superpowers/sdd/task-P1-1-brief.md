# Task P1-1: Wire e01_reference.srt as a regression oracle

## Context

This is part of a subtitle translation pipeline. There's already a `run_regression(cfg)` function at `translator/engine.py:3906` that translates corpus files via `translate_fast_to_texts` and compares against `_expected.srt` files.

The spec wants us to also compare E01 output against `config/e01_reference.srt` (a pre-recorded gold standard), and fix a false-positive issue in the comparison.

## Files

- `translator/engine.py` — existing `run_regression()` at line 3906, `translate_fast_to_texts()` at line 4012
- `config/e01_reference.srt` — gold standard output for drama_01_eng.srt
- `tests/corpus/drama_01_eng.srt` — the English source for episode 01
- `subtranslate.py` — CLI that calls `run_regression(cfg)` in regression mode (line 161)

## What to implement

### 1. Fix false-positive comparison

The comparison at `translator/engine.py:3964` does `ger_texts[i] != expected_texts[i]`. This flags trivial differences (trailing whitespace, line ending artifacts) as changed. Fix by stripping both sides before comparison:

```python
if ger_texts[i].strip() != expected_texts[i].strip():
```

### 2. Add E01 reference regression

In `run_regression()`, after the existing corpus loop, add a second phase that:
a. Locates `config/e01_reference.srt` (relative to project root or CONFIG_DIR)
b. Locates `tests/corpus/drama_01_eng.srt`
c. Runs `translate_fast_to_texts()` on it to get actual output
d. Loads the reference SRT using `safe_open_srt()` and extracts texts
e. Compares them (with .strip() as above)
f. Reports line-by-line differences (line number, expected, got — first 5 only)
g. Sums up "changed / improved / regressed" like existing code

### 3. Wire into subtranslate.py

The `run_regression()` function is already called from `subtranslate.py:161` when `cfg.mode == "regression"`. No changes needed to the CLI wiring.

## Key constraint

Do NOT break the existing corpus regression. The E01 reference check should be additive.

## Verify

Run: `python -c "import sys; sys.path.insert(0, '.'); from translator.engine import run_regression; from translator.engine import Config; c=Config(mode='fast', device='cpu'); run_regression(c)"`

(NLLB model loading takes time but the function should at least not crash on import.)
