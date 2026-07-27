# Sentence-Aware Reflow Pipeline — Design Spec

**Date:** 2026-07-27
**Status:** Approved
**Target:** Fix cross-caption sentence-boundary defects (incl. name-concatenation bug), without changing the model stack

## 0. Context — why this spec exists, and what it deliberately rejects

A friend recommended switching the translation backend to `gpt-oss-20b` via `llama.cpp`,
claiming ~10-20x speed over Ollama and better translation quality, and separately
recommended a "merge sentences → translate → reflow" architecture instead of translating
caption-by-caption.

Fact-checked before writing this spec (sources: 2026 benchmarks from AgenticWire,
Khired, InsiderLLM, Selfhostr, willitrunai, localllm.in, hardwarepedia):

| Claim | Verdict |
|---|---|
| llama.cpp is 10-20x faster than Ollama | **False.** On Windows/Linux, Ollama wraps llama.cpp directly; measured overhead is 5-15% (up to ~30% in edge cases), never 10x+. The friend's benchmark almost certainly had a GPU-offload or context/reasoning-effort misconfiguration in one of the two runs. |
| gpt-oss-20b fits the 4060 Ti 16GB | **True but marginal.** ~12-14GB VRAM incl. KV cache at Q4_K_M (8K context). Real throughput on a 4060 Ti is ~35-45 tok/s at Q4_K_M — slower than our current NLLB (~30 l/s) + Qwen2.5 7B polish (~1.2s/10-line batch) combo for batch subtitle work. |
| gpt-oss "refuses" to translate whole files and chunks despite instructions | **True, and not fixable by prompting.** gpt-oss is a chain-of-thought reasoning model; these are documented to override direct system-prompt instructions about length/format because the reasoning step re-derives its own plan (OpenAI's own harmony format docs, Caesar Creek security analysis). This is a structural property of reasoning models, not a prompt bug. |
| Sentence-aware merge → translate → reflow is better than per-caption translation | **True, and independent of model choice.** Matches professional subtitling practice: translate meaning at sentence boundaries, then reflow. The FBK submission to IWSLT 2026 ("The FBK Sentence-Aware Subtitling System") used identical two-stage architecture and showed consistent improvements across all language pairs. |

**Decision: do not migrate to gpt-oss-20b / llama.cpp.** Our current stack (NLLB-600M GPU
fast-pass + local Qwen2.5 7B polish) is already faster, already local, and doesn't fight
a reasoning model's own chain-of-thought for compliance. The one idea worth adopting is
sentence-aware processing — and it directly addresses our #1 open bug.

## 1. Root-cause hypothesis: name-concatenation bug is a sentence-boundary bug

Current defect (from project memory): character names glue onto the next word with no
space — `ChangningMach die Tür auf!` — across a large fraction of name references.

Looking at `protect_names()` / `restore_names()` / `_segment_line()` in `translator/engine.py`,
every protection and restoration pass operates **per caption line**, independently. NLLB
translates each caption's content in isolation. While `_segment_line` preserves spaces
between markers and content within a single caption, it has no cross-caption awareness.
When a name sits at the very end of one caption and the sentence it belongs to actually
continues into the next caption (a common SRT authoring pattern — subtitlers break
mid-sentence for timing, not grammar), there is no mechanism to ensure correct spacing
or sentence continuity across the boundary.

This spec's core claim: **treat sentence-boundary crossing as the mechanism**, not just a
symptom to patch with more regex. A dedicated sentence-merge stage would catch this class of
bug systematically rather than requiring one-off `german_fixes.json` entries per instance
(which is how it's been handled so far, per `config/german_fixes.json`'s 600+ entries).

## 2. Architecture

New optional stage inserted **before** NLLB translation and **before** protection layers,
gated behind a flag so it doesn't disturb the existing fast/polish/full/learn modes:

```
Source SRT (*.srt)
    │
    ▼
┌─────────────────────────────────────────┐
│ 0. SENTENCE MERGE (new, --sentence-aware)│
│    Group consecutive captions into       │
│    sentence blocks using punctuation +   │
│    time-gap heuristic. Record a mapping  │
│    of (block_id → [caption_indices])     │
│    so the split step can reverse it.     │
└──────────────────┬────────────────────────┘
                   ▼
   [existing protection layers: SFX, names,
    numbers, song/episode markers, etc. —
    now operate on the merged block text]
                   ▼
   [existing NLLB translate + polish passes
    — unchanged, just fed larger units]
                   ▼
┌─────────────────────────────────────────┐
│ N. SENTENCE SPLIT / REFLOW (new)          │
│    Split translated block back into      │
│    per-caption text using original       │
│    caption boundaries as target lengths, │
│    breaking at word boundaries nearest   │
│    the original split point. Re-run      │
│    cleanup_subtitles() + CPS check per   │
│    caption. Original timestamps          │
│    unchanged.                            │
└──────────────────┬────────────────────────┘
                   ▼
            Output SRT (*_ger.srt)
```

Key constraint: **this is additive**. If `--sentence-aware` is not passed, behavior is
byte-for-byte identical to today. This lets the regression corpus (`tests/corpus/`) validate
both code paths without needing new expected-output files immediately.

## 3. Merge heuristic (Step 0)

Reuse the merge condition already proven in `protect_multispeaker` / episode-marker logic
style (regex-first, deterministic, no LLM):

Merge caption `i` into the same block as caption `i+1` when **all** of:

1. Caption `i`'s text does not end in one of `. ? ! ♪ ] "` (i.e., sentence looks unfinished)
2. Caption `i+1`'s text does not start with a bracket/SFX marker or `[Episode N]` style marker
3. The time gap between caption `i`'s end and caption `i+1`'s start is below a threshold
   (default 500ms, configurable via `--merge-gap-ms`)
4. Caption `i` is not itself a multi-speaker line (contains `\n-`) — multi-speaker lines are
   already complete conversational units and should not be merged forward
5. The merged text does not exceed NLLB's `max_new_tokens=96` limit (enforced per block)

This mirrors the design doc's own recommended heuristic (sentence-boundary punctuation +
time-gap) and needs no new dependencies — it's pure regex/timestamp arithmetic against
`pysrt` objects already in use throughout `engine.py`.

## 4. Split/reflow heuristic (Step N)

After translation, each block must map back onto its original N captions. Approach:

1. Compute the character-length ratio of each original caption within its block
   (`len(caption_i) / len(block_source)`).
2. Apply the same ratios to the translated block text to get target split points.
3. Snap each target split point to the nearest word boundary (never split mid-word,
   never split a `ZZZ` marker — reuse `_SEGMENT_RE` to protect markers during this pass).
4. Assign each resulting chunk to its original caption's timestamps unchanged.

This is intentionally the *simple* ratio-based approach, not a translation-aware realignment
(no word-alignment model). It will not be perfect on heavy expansion/contraction, but it
directly targets the reported defect class: names and short connector words landing at block
boundaries with missing whitespace.

**CPS quality gate:** A helper function `calculate_cps(text, duration_ms)` must be added.
If a resulting caption chunk exceeds 35 CPS, fall back to **not merging that block** and
translate its captions individually as today — never make output worse than the current
per-caption baseline.

## 5. Explicit non-goals

- **No automatic retiming.** Changing caption start/end times is a separate, riskier
  feature (affects sync) and is out of scope. `--fix-timing` / `seconv` already exists for
  timing cleanup and is untouched by this spec.
- **No model/backend change.** NLLB-600M + Qwen2.5 7B stay as-is. This spec is purely a
  pre/post-processing layer around the existing translate calls.
- **No LLM-based sentence segmentation.** Punctuation + time-gap regex only, consistent with
  the project's existing "curated dictionaries/regex are more reliable" principle.

## 6. Files to modify

| File | Change |
|---|---|
| `translator/engine.py` | Add `merge_sentence_blocks()`, `split_sentence_blocks()`, `calculate_cps()`; wire into `translate_fast()` behind `cfg.sentence_aware`; block merging before protection layers; split after restoration/cleanup |
| `translator/__init__.py` | Export new functions |
| `subtranslate.py` | Add `--sentence-aware` and `--merge-gap-ms` flags |
| `config/german_fixes.json` | No change — but expect fewer new entries needed for concatenation-class bugs going forward |
| `tests/corpus/` | Add one corpus file with a deliberately mid-sentence caption break, to regression-test the merge/split round trip |

## 7. Implementation task list

### Task 1: `calculate_cps()` helper
- Takes text + duration_ms, returns CPS (characters per second, incl. spaces)
- Needs to handle multi-line text (count all chars, ignore newline for CPS count)
- Pure utility, no dependencies
- Verify: `assert calculate_cps("Hello world", 1000) == 11`

### Task 2: `merge_sentence_blocks()`
- Consumes: list of `(index, text, start_ms, end_ms)` tuples from an opened SRT
- Produces: list of blocks, each `{"caption_indices": [...], "merged_text": str}`
- Step 1: Implement merge heuristic from §3 (including NLLB token limit check)
- Step 2: Unit test on the corpus's `Zhou Wan Er wartet auf dich.` line (12+13 in
  `drama_01_eng.srt`) — verify it does NOT merge (line 12 already ends in `?`)
- Step 3: Add a synthetic corpus case where caption N ends without punctuation and N+1
  starts with a name — verify it DOES merge
- Verify: `pytest tests/ -k merge_sentence_blocks`

### Task 3: `split_sentence_blocks()`
- Consumes: translated block text + original caption texts
- Produces: list of per-caption translated strings, same length as `caption_indices`
- Step 1: Implement ratio-based split with word-boundary snapping (§4)
- Step 2: Protect `ZZZ` markers during split (reuse `_SEGMENT_RE`)
- Step 3: CPS fallback — reject merge result, translate individually, log a `[FALLBACK]` line
- Verify: round-trip test — merge then split with an identity "translation" (no-op) must
  reproduce the original captions exactly

### Task 4: Wire into `translate_fast()`
- Step 1: Add `sentence_aware: bool = False` and `merge_gap_ms: int = 500` to `Config`
- Step 2: When enabled, run merge after loading source texts but BEFORE protection layers;
  split after `_canonical_restore()`, before `apply_short_exclamation_overrides` final pass
- Step 3: Add `--sentence-aware` / `--merge-gap-ms` flags to `subtranslate.py`
- Verify: `python subtranslate.py --mode fast --sentence-aware --input-dir tests/corpus`
  produces output with no regressions vs `--mode regression` baseline

### Task 5: Regression corpus addition
- Step 1: Add `tests/corpus/drama_03_eng.srt` with a caption pair split mid-sentence,
  where the second caption starts with a name (reproduces the concatenation bug pattern)
- Step 2: Generate expected output via `--mode regression` on GPU
- Verify: `python subtranslate.py --mode regression` shows the new file passing with
  `--sentence-aware` and shows the *old* concatenation defect without it (documents the fix)

### Task 6: Commit
```bash
git add translator/engine.py translator/__init__.py subtranslate.py tests/corpus/
git commit -m "feat: add optional sentence-aware merge/reflow pipeline (--sentence-aware)"
```

## 8. Rollout plan

1. Land behind the flag, off by default.
2. Run one full episode with `--sentence-aware --qa-report` and diff QA score against the
   same episode without the flag.
3. If QA score improves and no new regressions appear in `--mode regression`, flip the
   default to `on` in a follow-up change; keep the flag as an escape hatch.

## 9. References

- Cettolo et al. (2026). "The FBK Sentence-Aware Subtitling System at the IWSLT 2026
  Subtitling Track." *Proceedings of the 23rd International Conference on Spoken Language
  Translation (IWSLT 2026)*, 68–77. — Validates the two-stage sentence-aware architecture
  with consistent quality improvements across all tested language pairs.
- Ollama vs llama.cpp benchmarks (2026): AgenticWire, Khired, InsiderLLM, Selfhostr —
  Confirm 5-15% overhead, not 10-20x.
- GPT-OSS VRAM/hardware data (2026): willitrunai, localllm.in, hardwarepedia — Confirm
  ~12-14GB Q4_K_M on RTX 4060 Ti, ~35-45 tok/s throughput.
- Subtitle expansion data: inter-contact.de, sublingo.cc — German ~6.5 chars/word vs
  English ~5.2 chars/word, 20-35% text expansion.
