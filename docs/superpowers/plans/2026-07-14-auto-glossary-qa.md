# Auto-Glossary + QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--auto-glossary`, `--fix-timing`, `--fix-aggressive`, `--qa-report`, `--qa-spotcheck-lines` flags to the subtitle pipeline so episodes E06–E40 translate fully automatically with self-improving glossary and post-translation QA.

**Architecture:** Five new CLI flags added to `subtranslate.py`, two new functions in `engine.py` (`seconv_fix_timing`, `qa_report`). Existing `generate_glossary` and `merge_glossary_auto` are re-used. The `--auto-glossary` flag inserts a glossary-extract+merge step before each file's translation. The `--fix-timing`/`--fix-aggressive` flags run seconv after translation. `--qa-report` prints a per-episode QA summary.

**Tech Stack:** Python 3.13, subprocess for seconv, pysrt for QA scanning, existing engine.py glossary functions.

## Global Constraints

- All new flags default to False — existing commands unchanged
- `--fix-aggressive` must call seconv as two separate subprocess runs (not repeated flags)
- CPS formula: `CPS = visible_chars / duration_seconds`; warn > 22, error > 25
- Glossary QA check must be case-insensitive
- Manual glossary entries always win in merge
- No text-content changes from seconv (timing/formatting only)

---

### Task 1: Add seconv_fix_timing() to engine.py

**Files:**
- Modify: `translator/engine.py` (after `qa_report` placeholder, before section 3)

**Interfaces:**
- Consumes: nothing from other tasks
- Produces: `seconv_fix_timing(srt_path: Path, aggressive: bool = False) -> bool`

- [ ] **Step 1: Add the function**

Insert after `merge_glossary_auto()` (line 1094), before the `# Name preservation` section:

```python
def seconv_fix_timing(srt_path: Path, aggressive: bool = False) -> bool:
    """Fix common subtitle timing errors via seconv (Subtitle Edit CLI).

    Returns True if seconv ran successfully, False if not found/failed.
    With aggressive=True, runs the fix pass twice to catch secondary issues.
    """
    seconv = shutil.which("seconv")
    if seconv is None:
        print(f"  [WARN] seconv not found in PATH. Skipping timing fix for {srt_path.name}")
        return False

    passes = 2 if aggressive else 1
    for i in range(passes):
        label = f"  seconv pass {i+1}/{passes}" if aggressive else "  seconv"
        print(f"{label} {srt_path.name}...", end=" ", flush=True)
        try:
            result = subprocess.run(
                [seconv, str(srt_path), "--fix-common-errors"],
                capture_output=True, text=True, timeout=60,
            )
            if result.returncode == 0:
                print("OK", flush=True)
            else:
                print(f"exit {result.returncode}: {result.stderr.strip()[:120]}", flush=True)
                return False
        except FileNotFoundError:
            print("seconv not found", flush=True)
            return False
        except subprocess.TimeoutExpired:
            print("timed out", flush=True)
            return False
    return True
```

- [ ] **Step 2: Add imports at top of engine.py**

Add `import subprocess, shutil` to the existing import line at line 14:
```python
import argparse, json, os, re, sys, time, warnings, subprocess, shutil
```

- [ ] **Step 3: Verify the function loads**

Run: `& "C:\Translator\venv\Scripts\python.exe" -c "from translator.engine import seconv_fix_timing; print('OK')"`
Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add translator/engine.py
git commit -m "feat: add seconv_fix_timing() helper"
```

---

### Task 2: Add qa_report() to engine.py

**Files:**
- Modify: `translator/engine.py` (after seconv_fix_timing)

**Interfaces:**
- Consumes: nothing from other tasks
- Produces: `qa_report(srt_path: Path, eng_path: Path, glossary: dict, names: dict, spotcheck_lines: int = 50) -> dict`

- [ ] **Step 1: Add the function**

Add after `seconv_fix_timing()`:

```python
def qa_report(
    srt_path: Path,
    eng_srt_path: Path,
    glossary: dict,
    names: dict,
    spotcheck_lines: int = 50,
) -> dict:
    """Scan the first spotcheck_lines of German output and flag issues.

    Checks performed:
    1. Glossary coverage — glossary terms present in EN source missing from DE output
    2. Name preservation — names from names.json that appear untranslated in DE
    3. Line length — flag lines over 42 visible characters
    4. Reading speed — flag lines with CPS > 25 (error) or > 22 (warning)

    Returns a dict: {"errors": int, "warnings": int, "status": "PASS"|"FAIL", "details": [...]}
    """
    try:
        ger_subs = pysrt.open(str(srt_path), encoding="utf-8")
    except Exception as e:
        return {"errors": 0, "warnings": 0, "status": "FAIL", "details": [f"Can't open {srt_path.name}: {e}"]}

    try:
        eng_subs = pysrt.open(str(eng_srt_path), encoding="utf-8")
    except Exception:
        eng_subs = None

    # Build lookup from glossary
    gloss_lookup: dict[str, str] = {}
    for k, v in glossary.items():
        default, acceptable = _resolve_glossary_entry(v)
        if default:
            gloss_lookup[k.lower()] = (default.lower(), [a.lower() for a in acceptable])

    name_set = set()
    for k, v in names.items():
        if isinstance(v, str):
            name_set.add(v.lower())
        elif isinstance(v, list):
            name_set.update(x.lower() for x in v)

    errors = 0
    warnings = 0
    details: list[str] = []

    max_lines = min(spotcheck_lines, len(ger_subs))
    for i in range(max_lines):
        ger_sub = ger_subs[i]
        ger_text = ger_sub.text.strip()
        ger_text_no_tags = re.sub(r'<[^>]+>', '', ger_text)
        visible_chars = len(ger_text_no_tags)

        # Duration in seconds
        start_ms = ger_sub.start.ordinal
        end_ms = ger_sub.end.ordinal
        duration_s = (end_ms - start_ms) / 1000.0

        # 1. Glossary coverage
        eng_text = ""
        if eng_subs and i < len(eng_subs):
            eng_text = eng_subs[i].text.strip().lower()
        if eng_text:
            for eng_term, (ger_term, acceptable) in gloss_lookup.items():
                if re.search(r'(?<!\w)' + re.escape(eng_term) + r'(?!\w)', eng_text):
                    all_ok = [ger_term] + acceptable
                    if not any(
                        re.search(r'(?<!\w)' + re.escape(w) + r'(?!\w)', ger_text.lower())
                        for w in all_ok
                    ):
                        errors += 1
                        details.append(
                            f"  Line {i+1}: '{eng_term}' → '{ger_term}' missing in German output"
                        )

        # 2. Name preservation
        ger_lower = ger_text.lower()
        for name in name_set:
            if re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', ger_lower):
                warnings += 1
                details.append(
                    f"  Line {i+1}: Name '{name}' appears untranslated in German"
                )

        # 3. Line length
        if visible_chars > 42:
            warnings += 1
            details.append(f"  Line {i+1}: {visible_chars} chars exceeds 42")

        # 4. Reading speed (CPS)
        if duration_s > 0:
            cps = visible_chars / duration_s
            if cps > 25:
                errors += 1
                details.append(f"  Line {i+1}: CPS={cps:.1f} exceeds 25 (error)")
            elif cps > 22:
                warnings += 1
                details.append(f"  Line {i+1}: CPS={cps:.1f} exceeds 22 (warning)")

    status = "FAIL" if errors > 0 else "PASS"
    return {"errors": errors, "warnings": warnings, "status": status, "details": details}
```

- [ ] **Step 2: Verify the function loads**

Run: `& "C:\Translator\venv\Scripts\python.exe" -c "from translator.engine import qa_report; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add translator/engine.py
git commit -m "feat: add qa_report() for automated post-translation QA"
```

---

### Task 3: Wire auto-glossary into the pipeline loop in subtranslate.py

**Files:**
- Modify: `subtranslate.py` (imports + main loop)
- Modify: `translator/__init__.py` (exports)

**Interfaces:**
- Consumes: `seconv_fix_timing`, `qa_report`, `generate_glossary`, `merge_glossary_auto` from engine.py

- [ ] **Step 1: Add 5 new CLI flags in subtranslate.py**

After line 64 (the `--dry-run` flag), add:

```python
    # Auto-glossary + QA flags
    p.add_argument("--auto-glossary", action="store_true",
                   help="Before each file: extract domain terms, auto-merge into glossary, "
                        "then translate with updated glossary.")
    p.add_argument("--fix-timing", action="store_true",
                   help="After translation, run seconv --fix-common-errors on output SRT.")
    p.add_argument("--fix-aggressive", action="store_true",
                   help="Run seconv fix pass twice to resolve stubborn overlap issues.")
    p.add_argument("--qa-report", action="store_true",
                   help="After translation, print per-episode QA summary (first N lines).")
    p.add_argument("--qa-spotcheck-lines", type=int, default=50,
                   help="Number of leading lines to scan for --qa-report (default: 50).")
```

- [ ] **Step 2: Update imports to include new functions**

Change line 12-19 to add the new imports:
```python
from translator import (
    __version__, Config, _auto_device,
    find_srt_files, output_path_for,
    translate_fast, translate_polish, translate_llm,
    run_test, run_benchmark, run_regression,
    _checkpoint_path,
    generate_glossary, merge_glossary_auto,
    seconv_fix_timing, qa_report,
    load_glossary, load_names,
)
```

- [ ] **Step 3: Add auto-glossary block before each translation**

Inside the `for f in files:` loop (line 139), add at the very beginning (before any mode-specific logic):

```python
        # --auto-glossary: extract, merge, then translate with updated glossary
        if a.auto_glossary:
            print(f"  [auto-glossary] Extracting terms from {f.name}...", flush=True)
            glossary_focus = a.glossary_focus or (
                "butchery trades and meat processing, "
                "matrilocal marriage customs (ruzhu), "
                "Qing-style military ranks and titles, "
                "traditional medicine and herbal dosages, "
                "court factions and rebellion terminology"
            )
            generate_glossary([f], cfg, focus_topics=glossary_focus)
            added = merge_glossary_auto(dry_run=False)
            if added:
                print(f"  [auto-glossary] {added} new term(s) merged")
```

- [ ] **Step 4: Add post-translation QA + timing fix**

After each mode block (after `translate_polish` calls), add a shared post-processing block. Insert this at the end of the `for f in files:` loop body (right before the loop's closing), after all mode-specific blocks:

```python
        # Post-processing: timing fix + QA report
        out = output_path_for(f)

        if a.fix_timing or a.fix_aggressive:
            seconv_fix_timing(out, aggressive=a.fix_aggressive)

        if a.qa_report:
            en_srt_path = f.absolute() if f.exists() else None
            result = qa_report(
                out, f,
                glossary=load_glossary(),
                names=load_names(),
                spotcheck_lines=a.qa_spotcheck_lines,
            )
            print(f"  [QA] {f.name} → {result['status']} "
                  f"({result['errors']} err, {result['warnings']} warn)")
            for d in result["details"]:
                print(d)
```

- [ ] **Step 5: Export new functions from `__init__.py`**

Add `seconv_fix_timing, qa_report,` to both the import and `__all__` list.

- [ ] **Step 6: Commit**

```bash
git add subtranslate.py translator/__init__.py
git commit -m "feat: wire auto-glossary, fix-timing, and qa-report into pipeline"
```

---

### Task 4: Verify everything compiles and runs

- [ ] **Step 1: Verify imports**

Run: `& "C:\Translator\venv\Scripts\python.exe" -c "from subtranslate import main; print('All imports OK')"`
Expected: `All imports OK`

- [ ] **Step 2: Verify new flags appear in help**

Run: `& "C:\Translator\venv\Scripts\python.exe" subtranslate.py --help`
Expected: `--auto-glossary`, `--fix-timing`, `--fix-aggressive`, `--qa-report`, `--qa-spotcheck-lines` in output

- [ ] **Step 3: Quick functional test — dry run with --qa-report on existing output**

Run:
```bash
& "C:\Translator\venv\Scripts\python.exe" subtranslate.py --mode fast --input-dir "C:\Users\real_\AppData\Local\Temp\opencode\batch5" --qa-report --fix-timing --force
```
Expected: Processes one file with QA report printed. If seconv is not installed, shows warning and continues.

- [ ] **Step 4: Commit final changes**

```bash
git add -A
git commit -m "chore: finalize auto-glossary + QA pipeline"
git push
```
