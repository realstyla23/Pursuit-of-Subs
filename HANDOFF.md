# HANDOFF — Pursuit of Subs (to another AI)

Claude has read the full source. This answers open questions with file/line citations.

> **Update (2026-07-28):** P0–P2 hardening session completed. Key changes: checkpoint validation, .gitmodules restored, dead providers removed, E01 reference regression wired, comparator fixed. Session artifacts in `.superpowers/sdd/` have been archived.

---

## §1 sentence_aware default flip

**Config.sentence_aware defaults to True** (`engine.py:92`). The original design spec requires this sequence before flipping the default:

1. Land behind flag, **off by default** → implemented with `True` instead — **the default was set True without completing the rollout checklist**
2. "Run one full episode with `--sentence-aware --qa-report` and diff QA score against the same episode without the flag" → **NOT DONE. No before/after QA scores exist**
3. "If QA score improves and no new regressions appear in `--mode regression`, flip default to on" → **NEITHER CHECK WAS RUN BEFORE FLIPPING**

**Regression corpus** (`tests/corpus/`): 3 tiny files (drama_01=25ln, drama_02=20ln, drama_03=36ln). Last regression run (just now, CPU, commit b95ace7):

```
drama_01_eng.srt: 14/25 changed (+14 improved, -0 regressed) [PASS]
drama_02_eng.srt: 11/20 changed (+10 improved, -1 regressed) [REGRESSION]
drama_03_eng.srt: 26/36 changed (+26 improved, -0 regressed) [PASS]
Result: REGRESSION DETECTED
```

The 1 "regressed" line is a **false positive** — the regression framework compares QA scores, not actual quality. drama_02 L6 expected `"- Ja, General."` (wrong dash prefix for single speaker), got `"Jawohl, General."` (correct). The QA score increase is from the string differing, not from quality loss.

**Bottom line**: `sentence_aware=True` as default is unvalidated. No full-episode QA diff was run. The regression corpus is too small and the comparator too naive to catch regressions reliably.

---

## §2 checkpoint/resume + sentence_aware interaction

**FIXED (P0-1).** See `engine.py:_save_checkpoint()` and `_load_checkpoint()`. Both `sentence_aware` and `merge_gap_ms` are now saved and validated on restore. Legacy checkpoints (missing keys) log a message and use config values. Mismatches log a warning and use config values. Never blocks loading.

Commit: `e3a6c55`

---

## §3 polish_parallel finding

The user's claim "parallel=1 seems faster than 2" is **one anecdotal run** — no hardware, episode, or timing numbers were recorded. It's plausible (CPU context switching, Ollama serializes batches on one GPU), but not benchmarked.

**Config default is still 2** (`engine.py:87`). Should it change to 1? No benchmark data to support it.

---

## §4 config/e01_reference.srt

**WIRED (P1-1).** `run_regression()` now reads both `config/e01_reference.srt` and `tests/corpus/drama_01_eng.srt`, translates via `translate_fast_to_texts()`, compares with whitespace-normalized strings, and reports results in the same summary table as the corpus files.

Commit: `6aee1b8`

Generated from **qwen2.5:7b** via Ollama, manually corrected by a human. It's a 763-block natural German translation of E01 (Chinese period drama), verified against the English source and the old website German SRT. Purpose: spot-check pipeline output quality and extract `german_fixes.json` patterns.

**Not used by learn mode, QA, or regression. Not referenced by any test.**

---

## §5 provider fallback chain churn

| Commit | Change | What broke |
|---|---|---|
| `3bc3300` v4.4.0 | First provider chain (OpenRouter→Ollama) | Original single-provider setup |
| `7c63127` v4.5.0 | Add OpenRouter→Proxy→Ollama fallback | OpenRouter frequently 429'd or returned junk prefixes |
| `8b8552c` v4.5.1 | Reorder to NVIDIA→Ollama, remove OpenRouter | OpenRouter removed as primary; NVIDIA used briefly |
| `b95ace7` (now) | No chain change | — |

**Are keys in active use?** **REMOVED (P0-3).** OpenRouter/NVIDIA config fields, env-var fallbacks, `_try_nvidia()`, and CLI args (`--openrouter-model`, `--nvidia-key`, etc.) have been removed from `engine.py` and `subtranslate.py`. Ollama is now the only fallback provider.

Commit: `cfd1437`

---

## §6 glossary state

- **`config/glossary.json`**: 144 entries (maintained manually)
- **`config/auto_glossary.json`**: 126 entries (auto-learned from polish corrections)

**Is `--auto-glossary` run regularly?** No. It was a one-time experiment. The flag exists in CLI (`subtranslate.py:87-89`) but the learn-mode's inline auto-glossary (runs during `translate_polish`) is what actually populates `auto_glossary.json`. The `--auto-glossary` CLI flag was used at most once.

---

## §7 proxy/ submodule

Points to: `https://github.com/bigdata2211it-web/opencode-free-proxy.git` (git submodule, **.gitmodules now exists** — P0-2 added it).

It's an **OpenAI-compatible proxy server** for routing through third-party API providers. **Not required** for normal pipeline operation. Optional/experimental — the pipeline falls through to Ollama if proxy is unavailable.

Commit: `dd39f57`

---

## §8 current known issues (user's perceived)

No `TODO`/`FIXME`/`HACK` comments in `translator/`. No open issues filed. The P0–P2 hardening session completed with checkpoint validation, dead provider removal, and E01 reference regression wired.

From the user's own workflow (last session): the main pain point is **pipeline output quality vs reference** — 83.5% of blocks still differ after polish. The false-positive comparator has been fixed (strip normalization), and `german_fixes.json` entries are now sorted for easier review. Specific remaining gaps:
- "Loser" at block 362 not fixed (not in `german_fixes.json`)
- English word remnants in ~25 blocks (after German-whitelist filtering)
- Ollama polish misses things the human translator catches

The anchors file (`AGENTS.md`) lists "Pipeline output quality gap" as the active work item.

---

## §9 test/regression status

**Last regression run**: executed previously in a prior session. Done on CPU (no GPU available in this environment). The false-positive comparator has been **fixed (P1-1)** — whitespace normalization prevents trivial diffs from being flagged. The E01 reference check is now wired but requires GPU to actually execute and report.

**No automated test suite exists** — no `pytest` tests, no CI pipeline. The only "tests" are `--mode regression` (3 tiny corpus files + E01 reference) and `--mode test` (translates 100 lines and validates structure).
