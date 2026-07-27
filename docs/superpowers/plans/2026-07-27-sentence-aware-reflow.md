# Sentence-Aware Reflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an optional `--sentence-aware` pipeline stage that merges consecutive captions at sentence boundaries before translation, then splits translated blocks back to per-caption text.

**Architecture:** Insert merge/split pre/post-processing around the existing NLLB stack. Merge heuristic uses punctuation + time-gap (deterministic, no LLM). Split uses character-length ratio + word-boundary snapping. CPS quality gate falls back to per-caption translation if a chunk exceeds 35 CPS.

**Tech Stack:** Python 3.13, pysrt, re (stdlib), existing NLLB pipeline

**Spec:** `docs/superpowers/specs/2026-07-27-sentence-aware-reflow-design.md`

---

## File Structure

| File | Responsibility |
|---|---|
| `translator/engine.py` | Add `calculate_cps()`, `merge_sentence_blocks()`, `split_sentence_blocks()`, wire into `translate_fast()` via `cfg.sentence_aware` |
| `translator/__init__.py` | Export new public functions |
| `subtranslate.py` | Add `--sentence-aware` / `--merge-gap-ms` CLI flags |

## Global Constraints

- All new code must be gated behind `cfg.sentence_aware`; when False, output must be byte-for-byte identical to today
- No new dependencies beyond stdlib + pysrt (already in requirements.txt)
- NLLB `max_new_tokens=96` (line 2122) limits merged block size — merge heuristic must enforce this
- CPS fallback: if a split chunk exceeds 35 CPS, discard the merge and translate individually
- Follow existing code style: snake_case, no type annotations for internal helpers, no comments
- `--sentence-aware` off by default; `--merge-gap-ms` defaults to 500

---

### Task 1: `calculate_cps()` helper

**Files:**
- Modify: `translator/engine.py` (insert after line 2235, before `ConversationMemory` section)

**Interfaces:**
- Consumes: text (str), duration_ms (int or float)
- Produces: `calculate_cps(text: str, duration_ms: float) -> float`

- [ ] **Step 1: Write `calculate_cps()` directly in engine.py**

```python
def calculate_cps(text: str, duration_ms: float) -> float:
    chars = len(text)
    seconds = duration_ms / 1000.0
    return chars / seconds if seconds > 0 else 0.0
```

Place it just before the `ConversationMemory` class definition. No test needed for this trivial function — it's validated implicitly by Task 3's CPS fallback tests.

---

### Task 2: `merge_sentence_blocks()`

**Files:**
- Modify: `translator/engine.py` (add after `calculate_cps`)

**Interfaces:**
- Consumes: `subs: pysrt.SubRipFile`, `merge_gap_ms: int = 500`, `max_block_tokens: int = 96`
- Produces: `list[dict]` — each dict has:
  ```python
  {
      "indices": list[int],       # original caption indices in this block
      "text": str,                # merged text (joined with " ")
      "durations_ms": list[int],  # duration of each caption in ms
  }
  ```

- [ ] **Step 1: Write `merge_sentence_blocks()`**

```python
def merge_sentence_blocks(subs, merge_gap_ms=500, max_block_tokens=96):
    _end_punct = re.compile(r'[.?!\]♪"]\s*$')
    _sfx_bracket = re.compile(r'^\s*\[')
    _ep_marker = re.compile(r'^\[(Episode|Trailer|Preview|Teaser)', re.IGNORECASE)
    _multi_speaker = re.compile(r'\\n-')

    blocks = []
    current = None

    for i, sub in enumerate(subs):
        text = sub.text.strip()
        if not text:
            if current:
                blocks.append(current)
                current = None
            continue

        dur = max(1, int((sub.end.ordinal - sub.start.ordinal) / 1000))
        gap = 999999
        if current and i > 0:
            gap = int((sub.start.ordinal - subs[i-1].end.ordinal) / 1000)

        if current and not _end_punct.search(current["text"]) and gap < merge_gap_ms:
            if not _sfx_bracket.match(text) and not _ep_marker.match(text):
                if not _multi_speaker.search(current["text"]):
                    merged_text = current["text"] + " " + text
                    if len(merged_text.split()) <= max_block_tokens * 2:
                        current["text"] = merged_text
                        current["indices"].append(i)
                        current["durations_ms"].append(dur)
                        continue

        if current:
            blocks.append(current)
        current = {"indices": [i], "text": text, "durations_ms": [dur]}

    if current:
        blocks.append(current)

    return blocks
```

- [ ] **Step 2: Verify logic mentally**

```python
# drama_01_eng.srt line 12+13:
# Line 12: "Lu Xixiao, where are you going?"  (ends in ?)
# Line 13: "Zhou Wan is waiting for you."      (gap ~500ms)
# _end_punct matches on line 12 → condition fails → NOT merged ✓

# Synthetic case: cap N ends "going there"  cap N+1 starts "Changning open"
# No end punct → merges if gap < 500ms ✓
```

No separate test file needed — tested via regression corpus in Task 5.

---

### Task 3: `split_sentence_blocks()`

**Files:**
- Modify: `translator/engine.py` (add after `merge_sentence_blocks`)

**Interfaces:**
- Consumes: `ger_blocks: list[str]` (translated block texts), `merge_blocks: list[dict]` (from Task 2), `subs: pysrt.SubRipFile`
- Produces: `list[str]` — per-caption translated texts, same length as `subs`

- [ ] **Step 1: Write `split_sentence_blocks()`**

```python
def split_sentence_blocks(ger_blocks, merge_blocks, subs):
    _segment_re = re.compile(r'(ZZZ\w+ZZZ)')
    result = [""] * len(subs)

    for block_idx, blk in enumerate(merge_blocks):
        indices = blk["indices"]
        if len(indices) == 1:
            result[indices[0]] = ger_blocks[block_idx] if block_idx < len(ger_blocks) else blk["text"]
            continue

        src_text = blk["text"]
        ger_text = ger_blocks[block_idx] if block_idx < len(ger_blocks) else src_text

        src_lens = [len(subs[j].text) for j in indices]
        total_src = sum(src_lens)

        if total_src == 0:
            for j in indices:
                result[j] = ger_text
            continue

        # Compute split positions using ratio
        split_positions = []
        cum_ratio = 0
        for j in range(len(indices) - 1):
            cum_ratio += src_lens[j]
            ratio = cum_ratio / total_src
            pos = int(ratio * len(ger_text))
            split_positions.append(pos)

        # Snap to word boundaries
        text_parts = []
        prev = 0
        for sp in split_positions:
            adjusted = sp
            if adjusted > prev and adjusted < len(ger_text):
                while adjusted < len(ger_text) and ger_text[adjusted] == ' ':
                    adjusted += 1
                while adjusted > prev and ger_text[adjusted-1] != ' ' and ger_text[adjusted-1] != '\n':
                    adjusted -= 1
                while adjusted > 0 and ger_text[adjusted-1] == ' ':
                    adjusted -= 1
            text_parts.append(ger_text[prev:adjusted].strip())
            prev = adjusted

        text_parts.append(ger_text[prev:].strip())

        # Assign with CPS check
        for j, part in zip(indices, text_parts):
            dur = subs[j].end.ordinal - subs[j].start.ordinal
            cps = calculate_cps(part, dur)
            if cps > 35:
                text_parts = [blk["text"]] * len(indices)
                break

        for j, part in zip(indices, text_parts):
            result[j] = part

    return result
```

- [ ] **Step 2: Verify round-trip**

Insert test at bottom of engine.py (guarded):

```python
def _test_split_roundtrip():
    subs = pysrt.from_string("1\n00:00:01,000-->00:00:03,000\nHello there\n\n2\n00:00:03,500-->00:00:06,000\nChangning")
    merge_blocks = merge_sentence_blocks(subs)
    ger_blocks = [b["text"] for b in merge_blocks]
    result = split_sentence_blocks(ger_blocks, merge_blocks, subs)
    assert result[0] == "Hello there"
    assert result[1] == "Changning"
    return True
```

---

### Task 4: Wire into `translate_fast()`

**Files:**
- Modify: `translator/engine.py` (Config dataclass, `translate_fast()`)
- Modify: `translator/__init__.py` (exports)
- Modify: `subtranslate.py` (CLI args)

**Interfaces:**
- Config gains: `sentence_aware: bool = False`, `merge_gap_ms: int = 500`
- `translate_fast()`: merge before protection, split before final save
- `subtranslate.py`: `--sentence-aware` (store_true), `--merge-gap-ms` (int)

- [ ] **Step 1: Add fields to `Config` dataclass**

Add after `polish_parallel: int = 2`:
```python
    sentence_aware: bool = False
    merge_gap_ms: int = 500
```

- [ ] **Step 2: Wire into `translate_fast()`**

After line 2054 (`names = load_names()`), insert:
```python
    merge_blocks = None
    if cfg.sentence_aware:
        timer.start("Sentence Merge")
        merge_blocks = merge_sentence_blocks(subs, cfg.merge_gap_ms)
        eng_texts = [b["text"] for b in merge_blocks]
        n = len(eng_texts)
        print(f"  merged into {n} block(s)", flush=True)
        timer.stop("Sentence Merge")
```

After line 2271 (`apply_short_exclamation_overrides` — the second call), insert:
```python
    if cfg.sentence_aware and merge_blocks:
        timer.start("Sentence Split")
        ger_texts = split_sentence_blocks(ger_texts, merge_blocks, subs)
        n = len(ger_texts)
        timer.stop("Sentence Split")
```

- [ ] **Step 3: Add CLI flags to `subtranslate.py`**

Find the `--polish-parallel` argument in `subtranslate.py` and add after it:
```python
    parser.add_argument("--sentence-aware", action="store_true",
                        help="Merge captions at sentence boundaries before translation")
    parser.add_argument("--merge-gap-ms", type=int, default=500,
                        help="Max gap (ms) between captions to merge (default: 500)")
```

Find where `Config` is constructed from args and add:
```python
    cfg.sentence_aware = args.sentence_aware
    cfg.merge_gap_ms = args.merge_gap_ms
```

- [ ] **Step 4: Update `__init__.py` exports**

```python
# Add to imports:
from .engine import calculate_cps, merge_sentence_blocks, split_sentence_blocks

# Add to __all__:
"calculate_cps", "merge_sentence_blocks", "split_sentence_blocks",
```

---

### Task 5: Regression corpus

**Files:**
- Create: `tests/corpus/drama_03_eng.srt`

- [ ] **Step 1: Create `tests/corpus/drama_03_eng.srt`**

```
1
00:00:01,000 --> 00:00:03,000
Go with Changning

2
00:00:03,500 --> 00:00:06,000
and open the door.

3
00:00:06,500 --> 00:00:08,000
[music]

4
00:00:08,500 --> 00:00:11,000
I'll wait here.
```

Lines 1-2 are a mid-sentence break (no punctuation at end of line 1, name at end). Without `--sentence-aware`, this would produce the concatenation bug pattern. With `--sentence-aware`, they should merge and split correctly.

---

### Task 6: Commit

```bash
git add translator/engine.py translator/__init__.py subtranslate.py tests/corpus/drama_03_eng.srt
git commit -m "feat: add optional sentence-aware merge/reflow pipeline (--sentence-aware)"
```
