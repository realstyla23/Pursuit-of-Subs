# Task 4: Wire into translate_fast(), Config, subtranslate.py, __init__.py

**Files:**
- Modify: `translator/engine.py` (Config dataclass + translate_fast)
- Modify: `translator/__init__.py` (exports)
- Modify: `subtranslate.py` (CLI args)

## Part A: Config dataclass

Find the `Config` dataclass in `translator/engine.py`. It has a field `polish_parallel: int = 2`. Add after that line:

```python
    sentence_aware: bool = False
    merge_gap_ms: int = 500
```

## Part B: Wire into translate_fast()

In `translate_fast()` (currently at line 2024 in `engine.py`):

**Merge step (after loading texts, before protection layers):**

After line 2054 (`names = load_names()`), insert:

```python
    merge_blocks = None
    if cfg.sentence_aware:
        merge_blocks = merge_sentence_blocks(subs, cfg.merge_gap_ms)
        eng_texts = [b["text"] for b in merge_blocks]
        n = len(eng_texts)
        print(f"  merged into {n} block(s)", flush=True)
```

**Split step (after all post-processing, before final save):**

Find the second `apply_short_exclamation_overrides` call (around line 2271 in the original file). After that line, insert:

```python
    if cfg.sentence_aware and merge_blocks:
        ger_texts = split_sentence_blocks(ger_texts, merge_blocks, subs)
        n = len(ger_texts)
```

## Part C: CLI flags in subtranslate.py

Find `subtranslate.py` in the project root. Look for `--polish-parallel` argument. After it, add:

```python
    parser.add_argument("--sentence-aware", action="store_true",
                        help="Merge captions at sentence boundaries before translation")
    parser.add_argument("--merge-gap-ms", type=int, default=500,
                        help="Max gap (ms) between captions to merge (default: 500)")
```

Then find where `Config` is constructed from `args` (look for `cfg.polish_parallel`). After that line, add:

```python
    cfg.sentence_aware = args.sentence_aware
    cfg.merge_gap_ms = args.merge_gap_ms
```

## Part D: Update __init__.py exports

In `translator/__init__.py`, find the imports from `engine` and add:

```python
from .engine import calculate_cps, merge_sentence_blocks, split_sentence_blocks
```

Also add these names to any `__all__` list if present.
