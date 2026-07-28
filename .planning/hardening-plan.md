# Pipeline Hardening Plan

## Global Constraints

- No default value flips (P0-1 preserves existing behavior for legacy checkpoints)
- Every task ends with a Verify step — run relevant tests or commands
- Surgical changes: touch only what each task requires
- All commits go to `main` branch
- Tasks are ordered by priority tier: P0 first, then P1, then P2
- Inside P0: P0-1 and P0-2 are independent, P0-3 depends on neither
- Do not delete the HANDOFF.md file

## P0 — Crash & Integrity

### P0-1. Checkpoint validation — save/restore sentence_aware + merge_gap_ms

**Location:** engine.py (EngineSave dict, save_checkpoint, load_checkpoint, resume methods)

**Goal:** The engine's `sentence_aware` and `merge_gap_ms` fields are persisted but never validated on restore. If a checkpoint was saved with different values than what the config specifies, the engine silently uses whatever was saved. This has caused subtle output drift.

**Implementation:**

1. Add `sentence_aware` and `merge_gap_ms` to the EngineSave dict in `save_checkpoint()` so they're saved alongside the existing fields.
2. In `load_checkpoint()` (or wherever checkpoint restore happens), after loading the EngineSave, validate that the loaded `sentence_aware` and `merge_gap_ms` match the current config values.
3. If they differ, log a warning: `"Checkpoint sentence_aware mismatch: loaded=%s, config=%s — using config value"` and use the config value (not the checkpoint value).
4. If `sentence_aware` is not in the loaded checkpoint (legacy), log: `"Legacy checkpoint — setting sentence_aware=%s from config"`.
5. Repeat #3-4 for `merge_gap_ms`.
6. Verify: run `python translate.py` with a real checkpoint file to confirm round-trip works and warnings fire correctly when values differ.

**Surgical scope:** Only modify the save dict, the load/restore logic, and import any needed logging helpers. Do not touch the engine's core translation loop or any other feature.

---

### P0-2. Restore missing .gitmodules entry for proxy/ submodule

**Location:** repo root (create .gitmodules if missing)

**Goal:** `git submodule update --init --recursive` should work for `proxy/`.

The proxy/ directory is already a gitlink entry in the index (commit `49488fd2c6623cbcfdf11ef584722563df08aceb`), but `.gitmodules` is missing, so git doesn't know the URL. The origin URL is `https://github.com/bigdata2211it-web/opencode-free-proxy.git` (from proxy/README.md).

Do NOT use `git submodule add` — the path is already occupied. Instead, create `.gitmodules` manually.

**Implementation:**

Create `.gitmodules` with:

```
[submodule "proxy"]
	path = proxy
	url = https://github.com/bigdata2211it-web/opencode-free-proxy.git
```

Verify: `git submodule status` should no longer error, and `git config --file .gitmodules --list` should show the mapping.

---

### P0-3. Remove dead OpenRouter / NVIDIA provider code

**Location:** engine.py and subtranslate.py (search for openrouter, nvidia, open_router, nvida)

**Goal:** Dead config paths and provider-specific code for OpenRouter and NVIDIA have been accumulating. They're never used (the project only uses ollama, openai-compatible, and deepseek providers). Removing dead branches reduces cognitive load and eliminates the risk of someone accidentally configuring a dead provider.

**Approach: surgical grep-and-remove.** Do not refactor the provider dispatch logic — just remove dead provider branches.

**Implementation:**

1. Search engine.py for all references to `openrouter`, `nvidia` (case-insensitive).
2. Remove:
   - Any provider config defaults mentioning OpenRouter or NVIDIA.
   - Any conditional branches that dispatch to these providers (but not the shared dispatch infrastructure).
   - Any import-only lines that were only needed for these providers (check if removing the import creates a NameError).
3. Search subtranslate.py for the same terms.
4. Remove matching dead branches.
5. Check for any test files referencing these providers and remove/reference-srt them.
6. Verify: `python translate.py` starts and the help text no longer mentions dead providers.

---

## P1 — Confidence (Regression Oracle)

### P1-1. Wire e01_reference.srt as a regression oracle

**Location:** e01_reference.srt, translate.py (or a new test runner), any comparator logic

**Goal:** An `e01_reference.srt` file exists with the expected output for episode 1. Wire it into the pipeline so the user can run a single command to verify that a code change hasn't regressed the E01 output. The comparator is currently overmatching — flagging identical output as different. Fix the comparator.

**Implementation:**

1. Add a `--regression` flag (or a `verify()` entry point) to `translate.py` that:
   a. Runs the full pipeline for episode 01
   b. Loads `e01_reference.srt`
   c. Compares actual vs. reference
   d. Reports any differences with line numbers and context

2. Fix the comparator in the existing test infrastructure (`tests/test_translate.py` or wherever it lives). The bug: it compares output lists with something that makes it see differences where none exist. Likely cause: whitespace normalization, ordering of items, or encoding. Diagnose and fix.

3. Verify: the new `--regression` flag passes on the current code (green check) AND fails if you inject a known difference (red check). Run `python translate.py --regression` to confirm.

---

### P1-2. Feed e01 diffs into german_fixes.json via reference learn scan

**Location:** e01_reference.srt, german_fixes.json

**Goal:** Some entries in the E01 output still differ from the reference. Run the reference learn scan to absorb these into `german_fixes.json`, then re-verify P1-1 passes without error.

**Implementation:**

1. Run the pipeline's "reference learn" mode (or equivalent) that compares the engine's E01 output against `e01_reference.srt` and feeds any diffs through the fix extraction pipeline, adding entries to `german_fixes.json`.
2. After the scan, sort `german_fixes.json` keys alphabetically (if the file supports ordering).
3. Re-run `--regression` to confirm clean pass.
4. Run the full E01 pipeline to confirm output is now identical to reference.

---

## P2 — Polish Benchmarking

### P2-1. Benchmark polish_parallel 1 vs 2

**Goal:** Determine whether `polish_parallel` v2 (multi-line batches) is faster than v1 (single-line) on the full E01.

**Implementation:**

1. Run `python translate.py --polish-parallel 1` on E01 and time the polish phase.
2. Run `python translate.py --polish-parallel 2` on E01 and time the polish phase.
3. Record both times in the progress ledger or as a comment.
4. If v2 is not faster, note the suspicion that both are running the same path.
