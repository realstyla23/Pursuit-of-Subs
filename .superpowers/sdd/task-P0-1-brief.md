# Task P0-1: Checkpoint validation — save/restore sentence_aware + merge_gap_ms

**Location:** `translator/engine.py`

**Goal:** The engine's `sentence_aware` and `merge_gap_ms` fields (Config defaults at lines 92-93) are persisted but never validated on restore. If a checkpoint was saved with different values than what the config specifies, the engine silently uses whatever was saved. This has caused subtle output drift.

## Implementation

1. **In `_save_checkpoint()`** (line 2034): Add `sentence_aware` and `merge_gap_ms` to the save dict. The function receives individual params, not a Config object, so add two new parameters `sentence_aware: bool` and `merge_gap_ms: int` to the function signature, and include them in the checkpoint dict.

2. **In `_load_checkpoint()`** (line 2057): After loading, validate that the loaded `sentence_aware` and `merge_gap_ms` match the current config values (`cfg.sentence_aware`, `cfg.merge_gap_ms`). If they differ, log a warning and use the config value (not the checkpoint value). Handle legacy checkpoints (missing keys) by logging an info message.

3. **At the call site** (line 2440, 2443, 2448): Update `_save_checkpoint` calls to pass `cfg.sentence_aware` and `cfg.merge_gap_ms`.

## Existing signatures

```python
def _save_checkpoint(fpath: Path, out: Path, completed_batches: list[int],
                     batch_size: int, total_lines: int, elapsed: float,
                     all_trans: dict | None = None):
    ...

def _load_checkpoint(fpath: Path, out: Path, cfg: Config
                     ) -> tuple[list[int], dict[int, str], float] | None:
    ...
```

## Validation logic

```python
# After the existing validation (line 2066-2068), add:
for key, cfg_val in [("sentence_aware", cfg.sentence_aware), ("merge_gap_ms", cfg.merge_gap_ms)]:
    loaded_val = ckp.get(key)
    if loaded_val is None:
        print(f"  [RESUME] Legacy checkpoint — setting {key}={cfg_val} from config", flush=True)
    elif loaded_val != cfg_val:
        print(f"  [RESUME] Checkpoint {key} mismatch: loaded={loaded_val}, config={cfg_val} — using config value", flush=True)
```

The key constraint: the validation must NOT abort loading — it always proceeds with config values. Only log warnings.
