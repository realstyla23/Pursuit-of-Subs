# Learn Mode — Self-Improving Translation Pipeline

**Date:** 2026-07-19
**Status:** Design Approved
**Target:** 99% translation accuracy for Chinese→German subtitle pipeline

## 1. Purpose

Add a `--mode learn` flag that transforms the subtitle translation pipeline from a one-shot translator into a self-improving system. Each run detects its own errors, generates fixes, persists them to `german_fixes.json`, and re-applies — making every subsequent run better automatically.

## 2. Architecture

```
NLLB 600M → QA → Polish Pass 1 → Polish Pass 2 → German Fixes → 
Error Scanner → Fix Verifier → Persist → Final Re-apply
```

### Mode flow

`--mode learn` runs the same base pipeline as `--mode full` but adds three new stages after fixes are applied:

1. **Polish Pass 2** — second LLM pass focused specifically on eliminating English words and unnatural phrasing
2. **Error Scanner** — per-line LLM audit that identifies remaining errors and generates `(find, replace)` pairs
3. **Fix Verifier** — validates candidates before persisting to `german_fixes.json`

## 3. Existing code locations

| Component | File | Line |
|---|---|---|
| Base pipeline | `translator/engine.py` | `translate_polish_multi()` ~3409 |
| Polish prompt | `translator/engine.py` | `POLISH_PROMPT` ~2760 |
| `apply_german_fixes` | `translator/engine.py` | 1318 |
| `_filter_english_words` | `translator/engine.py` | 1341 |
| `load_german_fixes` | `translator/engine.py` | 1312 |
| German fixes data | `config/german_fixes.json` | — |
| QA scoring | `translator/engine.py` | `qa_report()` 1134 |
| CLI entry point | `subtranslate.py` | — |

## 4. New components

### 4.1 Polish Pass 2 — English killer

A second polish pass with a focused prompt. Runs after Pass 1 and the standard German fixes.

**Prompt:**
```
You are a German proofreader. Your ONLY job: find and eliminate any English 
words that remain in this German subtitle.

Rules:
- EVERY word must be German
- If you see an English word, replace it with the correct German word
- If you see a made-up word that isn't real German (like "panizierte"), 
  replace it with proper German ("geriet in Panik")
- Fix adjective declensions: "leuchtende Augen" → "leuchtenden Augen"
- Fix wrong noun compounds

Return ONLY the corrected text. If unchanged, return the original.
```

**Implementation:** Same function as Pass 1, different prompt string.

### 4.2 Error Scanner

Per-line LLM audit. For each `(eng_text, ger_text)` pair, calls qwen2.5:7b with a structured audit prompt and parses the response.

**Prompt:**
```
You are a quality inspector for German subtitles. Compare this English source 
line with its German translation.

English: {eng_text}
German: {ger_text}

Does the German contain ANY error?

Error categories (return the FIRST matching category):
1. ENGLISH_WORD — any English word left untranslated
2. MADE_UP — word that isn't real German (e.g. "Gefühlungen", "panizierte")
3. WRONG_WORD — real German word but wrong meaning (e.g. "gierig" for "stingy")
4. GRAMMAR — wrong declension, word order, separable verb placement
5. FORMAT — missing space, wrong capitalization, missing punctuation

Respond in EXACTLY this format. One line per error found:
FIX|exact text from German|correct replacement

Example:
FIX|Gefühlungen|Gefühlen
FIX|Schweineinnereien, innereien|Innereien

If NO errors, respond only: OK
```

**Implementation:**
```python
def learn_scan(eng_texts: list[str], ger_texts: list[str], polisher) -> list[dict]:
    """Scan all line pairs for errors. Returns list of {find, replace} dicts."""
```

One LLM call per line pair (~800 per episode). Sequential (not batched) for maximum accuracy.

### 4.3 Fix Verifier

Validates each error scanner candidate before adding to `german_fixes.json`.

**Checks in order:**

1. **Exists check** — `find` must appear (case-insensitive) in at least one line of the current German output. If not, skip (false positive).

2. **Format check** — if `find` matches a known format error pattern (missing space after punctuation, duplicate word, capitalization), skip QA verification and auto-approve.

3. **QA check** (semantic fixes only) — apply the fix, re-score changed lines with QA model. If score drops, skip the fix.

4. **Duplicate check** — if an identical `find` already exists in `german_fixes.json`, skip.

5. **Overlap check** — if `find` is a substring of any existing `find`, compare coverage. If the new fix is more specific, add it; otherwise skip.

**Output:** list of verified fixes ready for persistence.

### 4.4 Persister

Appends new verified fixes to `config/german_fixes.json`.

```python
def learn_persist(fixes: list[dict]) -> int:
    """Append new fixes to german_fixes.json. Returns count added."""
    existing = load_json("german_fixes.json")
    existing_finds = {f["find"].lower() for f in existing}
    new_count = 0
    for fix in fixes:
        if fix["find"].lower() not in existing_finds:
            existing.append(fix)
            existing_finds.add(fix["find"].lower())
            new_count += 1
    save_json("german_fixes.json", existing)
    return new_count
```

## 5. Data flow (detailed)

```
For one episode (763 lines):

1. NLLB 600M produces German[763]
2. QA scores German[763], flags lines below SUSPICIOUS_THRESHOLD=2
3. Polish Pass 1 improves German[763] → German_v2[763]
4. Polish Pass 2 kills remaining English → German_v3[763]
5. apply_german_fixes(German_v3, known_fixes) → German_v4[763]
6. For each i in 0..762:
     response = polisher.chat(ERROR_SCAN_PROMPT(eng[i], ger_v4[i]))
     if response starts with "FIX|":
         parse (find, replace), add to candidates[]
7. verified = verify(candidates, German_v4)
8. new_count = persist(verified)
9. apply_german_fixes(German_v4, verified) → German_final[763]
10. QA scores German_final
11. Print summary: errors found, fixes added, QA before/after
```

## 6. Performance

| Stage | Time | Notes |
|---|---|---|
| NLLB 600M | ~33s | Unchanged |
| Polish Pass 1 | ~44s | Current behavior |
| Polish Pass 2 | ~44s | New, same model |
| Error Scanner | ~13min (800 calls × ~1s) | New, sequential |
| Fix Verifier | ~30s | New, only semantic fixes need QA |
| Persist + re-apply | <1s | New |

**Total per episode:** ~15min (was ~2min)
**User constraint:** "doesn't matter if it takes hours"

## 7. Convergence toward 99%

The system is designed to improve over time. Each run:
- Scans all output for errors
- Persists unique fixes to `german_fixes.json`
- Those fixes apply automatically on future runs

After ~40 episodes, `german_fixes.json` should contain ~500-800 entries covering:
- Vocabulary-specific fixes (proper names, domain terms)
- Common NLLB mistranslation patterns
- Grammar patterns the polish model misses

Expected fix discovery rate:
- First 10 episodes: ~15-25 new fixes each
- Episodes 11-20: ~5-10 new fixes each
- Episodes 21-40: ~1-3 new fixes each
- After 40 episodes: ~0-1 per episode (99%+ accuracy)

## 8. CLI interface

```python
p.add_argument("--mode", choices=["fast", "polish", "full", "test", "benchmark", "regression", "llm", "learn"])
```

```bash
# Script for convenient use
python subtranslate.py --mode learn --input-dir D:\DBZ\episodes --polish-model qwen2.5:7b
```

## 9. Files to modify

| File | Change |
|---|---|
| `translator/engine.py` | Add `POLISH_PASS2_PROMPT`, `learn_scan()`, `learn_verify()`, `learn_persist()`, wire into pipeline |
| `subtranslate.py` | Add `"learn"` to mode choices |
| `config/german_fixes.json` | Auto-populated by learn mode (no manual change needed) |

## 10. Out of scope

- LLM model upgrades (NLLB 1.3B, qwen2.5:14b) — possible future improvement, not part of this design
- Cloud API integration — user wants local-only
- LanguageTool integration — optional enhancement, not required for learn mode
- Web GUI changes — learn mode is CLI-only
