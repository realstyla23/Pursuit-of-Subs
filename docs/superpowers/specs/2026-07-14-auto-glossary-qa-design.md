# Design: Automated Glossary + QA for Zero-Touch Episode Pipeline

## Motivation

Scale the existing NLLB+DeepSeek subtitle pipeline from manual per-episode work to a fully automatic mode for E06–E40. The system should improve its own glossary over time without human intervention.

## Flags

All new flags are combinable with existing modes (`--mode fast`, `--mode full`, etc.):

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--auto-glossary` | bool | False | Before each episode, extract domain terms from source SRT, auto-merge into glossary.json, then translate with the updated glossary |
| `--fix-timing` | bool | False | After translation, run `seconv --fix-common-errors` on the output SRT (Subtitle Edit CLI) |
| `--fix-aggressive` | bool | False | Run `--fix-timing` twice to resolve stubborn overlap issues |
| `--qa-report` | bool | False | After translation, scan output and print per-episode QA summary |
| `--qa-spotcheck-lines` | int | 50 | Number of leading lines to scan for `--qa-report` |

## Pipeline Flow (with all flags enabled)

```
for each .srt file in input-dir:
  │
  ├── 1. generate_glossary(file)        → config/glossary_auto.json
  ├── 2. merge_glossary_auto()          → non-interactive merge into glossary.json
  │
  ├── 3. translate_fast(file)           → NLLB with updated glossary
  ├── 4. translate_polish(file)         → DeepSeek with updated glossary
  │
  ├── 5. seconv --fix-common-errors     (if --fix-timing)
  │       └── run twice                 (if --fix-aggressive)
  │
  └── 6. qa_report(out_file)           → print PASS/FAIL + warnings (if --qa-report)
```

## Component Details

### generate_glossary (existing, unchanged)
- Reads SRT text, chunks to 12k chars, sends to DeepSeek with narrow domain prompt
- Skips API call if `glossary_auto.json` already exists and was generated for the same input file (future optimization)
- Saves to `config/glossary_auto.json`

### merge_glossary_auto (existing, extended)
- Non-interactive mode (already implemented via `interactive=False`)
- Manual entries in `glossary.json` always win
- New entries auto-merged without prompting

### qa_report (new function in engine.py)
Scans the output SRT (first N lines = `--qa-spotcheck-lines`) and prints:

```
[QA] E06_ger.srt: 0 errors, 2 warnings → PASS
  Warnings:
    - Line 12: "Vanguard" missing in German output (glossary entry)
    - Line 34: CPS=28 exceeds 25 limit
```

Checks:
1. **Glossary coverage**: For each glossary term present in English source, verify it appears in German output
2. **Name preservation**: Ensure names from names.json are not literally translated
3. **Line length**: Flag lines over 42 characters (optional, configurable)
4. **Reading speed**: Flag lines with CPS > 25 (optional, configurable)

### seconv integration (new)
- Detect `seconv`/`SubtitleEdit` CLI in PATH
- Call `seconv <file> --fix-common-errors` as subprocess
- With `--fix-aggressive`: run twice
- Non-fatal: if seconv not found, warn and skip

## Safety & Self-Improvement

- **Manual wins**: `merge_glossary_auto` never overwrites existing manual entries
- **Seconv**: Only modifies timing/formatting (not text content)
- **Non-breaking**: All flags default to False; existing commands work identically
- **Glacial improvement**: Each episode adds ~1-5 new glossary terms; over 35 episodes the glossary saturates, improving translation quality automatically

## Files to Modify

| File | Change |
|------|--------|
| `translator/engine.py` | Add `qa_report()` function; add `seconv_fix_timing()` function |
| `translator/__init__.py` | Export new functions |
| `subtranslate.py` | Add 5 new CLI flags; wire auto-glossary loop into main pipeline |

## Non-Goals

- No text-content changes from seconv (glossary/spelling handled by DeepSeek)
- No interactive prompts
- No new modes — only additive flags
- No changes to translate_fast/translate_polish internals
