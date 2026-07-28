# Pursuit of Subs

GPU-accelerated batch subtitle translation (EN → DE) using Opus-MT EN-DE (default) or Facebook's NLLB-600M distilled, with LLM polishing (local via Ollama, or any OpenAI-compatible proxy).

## Features

- **Multi-model** — Choose between Opus-MT EN-DE (default, 2.6x faster) or NLLB-600M via Web GUI dropdown
- **Polish** — LLM quality pass on suspicious lines (Qwen 2.5 locally via Ollama, or DeepSeek via proxy)
- **Full** — Translation + polish combined for highest quality
- **Learn** — Self-improving mode: full pipeline + automatic error detection + fix persistence to `german_fixes.json`. Each run makes the next run better. Re-runs skip expensive passes in ~0.3s.
- **Artifact scan** — Algorithmic NLLB hallucination detection (no LLM needed). Catches fake compounds, repeated words, wrong titles, Sie/du mixing, hallucinated addresses in milliseconds.
- **Learn verify** — Blacklist filter rejects fixes containing known hallucination terms (Schweinschlachter, Geistesgestörter, jinx) before they enter the fix database.
- **English filter** — Catches and translates English words the model missed, without false positives
- **Parallel batches** — Suspicious lines grouped into batches of 10, sent concurrently (2 by default)
- **Show context** — 27-character character database with relationship-aware formality rules (du/Sie) injected into the polish prompt
- **Smart protection** — SFX, numbers, names, song/episode markers, multi-speaker lines, short fragments (vocatives, interjections) survive translation correctly
- **Glossary** — Domain-specific terminology enforcement (140+ entries)
- **Glossary Automation** — Extract domain terms from source SRTs via DeepSeek, then merge into glossary with dry-run preview
- **Auto-Glossary mode** — `--auto-glossary` learns new terms from each episode and applies them immediately, improving translation quality over time without manual intervention
- **Post-translation QA** — `--qa-report` scans output for missing glossary terms, lost character names, line length violations, and reading speed (CPS) issues
- **Timing fix** — `--fix-timing` runs Subtitle Edit CLI (`seconv`) to fix common subtitle timing errors
- **Translation Memory** — Caches approved translations per line (opt-in, off by default)
- **QA scoring** — Detects untranslated lines, missing glossary/names, length anomalies, invented content
- **Checkpoints** — Resume interrupted translations without data loss

## Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA GTX 1060 6GB | RTX 3060+ 8GB+ |
| CUDA | 11.8 | 12.4+ |
| RAM | 8 GB | 16 GB |
| Disk | 4 GB free | 10 GB+ (for model cache) |
| Python | 3.10 | 3.13 |

## Quick Start

### 1. Install

```bash
git clone https://github.com/realstyla23/Pursuit-of-Subs.git
cd Pursuit-of-Subs

python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 2. Run

```bash
# Fast mode (Opus or NLLB) — ~16s per episode (Opus)
python subtranslate.py --mode fast --input-dir "path\to\subs"

# Full mode (translate + LLM polish) — ~62s per episode
python subtranslate.py --mode full --input-dir "path\to\subs"

# Full with Gemma 4 and 3 parallel batches
python subtranslate.py --mode full --polish-model gemma4:e4b --polish-parallel 3 --input-dir "path\to\subs"

# Polish-only on existing NLLB output with Gemma 4
python subtranslate.py --mode polish --polish-model gemma4:e4b --input-dir "path\to\subs"

# Force re-translate (no TM reads)
python subtranslate.py --mode fast --force --input-dir "path\to\subs"

# Launch the browser-based web GUI (model dropdown in browser settings)
python subtranslate.py --web-gui
# Or double-click launch_web_gui.bat
```

## CLI Reference

```bash
python subtranslate.py [OPTIONS]

Options:
  --mode MODE           fast, polish, full, test, benchmark, regression, llm, learn
  --input-dir DIR       Directory containing .srt files (default: .)
  --device DEV          cuda or cpu (default: cuda)
  --batch-size N        NLLB batch size (default: 64)
  --num-beams N         Beam search width (default: 4)
  --force               Re-translate existing output, skip TM reads
  --cache               Enable Translation Memory (default: off)
  --proxy-base-url URL  OpenCode proxy URL (default: http://127.0.0.1:6446)
  --proxy-api-key KEY   OpenCode API key (set via PROXY_API_KEY env var or this flag)
  --polish-model MODEL  Ollama model for polish (default: qwen2.5:7b, or gemma4:e4b)
  --polish-parallel N   Parallel polish batches (default: 2, max: 3)
  --resume              Resume from checkpoint
  --gui                 Launch PySide6 desktop GUI
  --web-gui             Launch browser-based GUI (Flask + SSE)
  --generate-glossary   Extract domain terms via DeepSeek → config/glossary_auto.json
  --merge-glossary      Auto-merge glossary_auto.json into glossary.json
  --glossary-focus TOPICS  Comma-separated domain topics (overrides default)
  --interactive         Prompt per new entry when merging
  --dry-run             With merge: show diff, no write
  --auto-glossary       Per-file: extract terms → auto-merge → translate (self-improving)
  --fix-timing          Post-translation: run seconv --fix-common-errors
  --fix-aggressive      Run seconv twice for stubborn overlaps
  --qa-report           Post-translation: print QA summary (glossary, names, CPS, line length)
  --qa-spotcheck-lines N  Leading lines to scan for QA (default: 50)
  --test                Run internal test suite
  --benchmark           Measure performance
```

Examples:

```bash
# Fast pass (Opus-MT default)
python subtranslate.py --mode fast --input-dir "D:\Shows\Season 1"

# Full pipeline: NLLB + local Gemma 4 polish
python subtranslate.py --mode full --force --polish-model gemma4:e4b

# Polish NLLB output with Qwen 2.5 (default)
python subtranslate.py --mode polish --polish-model qwen2.5:7b

# Fastest polish: Gemma 4 with 3 parallel batches
python subtranslate.py --mode polish --polish-model gemma4:e4b --polish-parallel 3

# Learn mode: full pipeline + auto-error-detection + fix persistence
python subtranslate.py --mode learn --input-dir "D:\Shows\E06"

# Extract domain glossary from source SRTs
python subtranslate.py --generate-glossary --input-dir "D:\Shows\Season 1"

# Preview what would be merged
python subtranslate.py --merge-glossary --dry-run

# Merge into glossary.json
python subtranslate.py --merge-glossary

# Full auto-pipeline: glossary learning + NLLB + polish + QA
python subtranslate.py --mode full --auto-glossary --qa-report --input-dir "D:\Shows\E06"

# Fast pipeline with timing fix and QA
python subtranslate.py --mode fast --fix-timing --qa-report --input-dir "D:\Shows\E06"

# Benchmark
python subtranslate.py --mode benchmark

# Regression test
python subtranslate.py --mode regression
```

## Input/Output Naming

Input files (`.srt`) are translated to `*_ger.srt` in the same directory:

```
Pursuit.of.Jade.E01.srt  →  Pursuit.of.Jade.E01_ger.srt
```

## Project Structure

```
├── subtranslate.py              CLI entry point
├── launch_gui.bat               PySide6 GUI launcher (double-click)
├── launch_web_gui.bat           Web GUI launcher (double-click)
├── try_polish.bat               Polish-only launcher
├── translator/
│   ├── engine.py                Core translation pipeline
│   ├── gui.py                   PySide6 GUI
│   └── __init__.py              Public API exports
├── web_gui/
│   ├── server.py                Flask app, API, SSE streaming
│   └── static/
│       └── index.html           Browser UI (HTML+CSS+JS)
├── config/
│   ├── glossary.json            Domain terminology map
│   ├── german_fixes.json        Known fix patterns
│   ├── short_fragments.json     Fragments NLLB hallucinates on
│   ├── names.json               Character name list
│   ├── titles.json              Known title translations
│   ├── auto_glossary.json       Auto-learned terms from polish corrections
│   ├── glossary_auto.json       DeepSeek-extracted domain terms
│   ├── learned_episodes.json    Episodes processed by learn mode
│   ├── e01_reference.srt        Human-verified E01 reference translation
│   └── show_contexts/           Character context databases per episode
├── assets/
│   └── screenshot_*.png         README screenshots
├── tests/
│   └── corpus/                  Regression test corpus
├── e01/
│   └── E01_eng.srt              English source for episode 1
├── tm/
│   └── .gitkeep                 Translation memory directory
├── requirements.txt
└── pyproject.toml
```

## Pipeline Architecture

```
Source SRT (*.srt)
    │
    ▼
┌─────────────────────────────────────────────┐
│ 1. PROTECT                                   │
│    SFX markers          protect_sfx()       │
│    Short exclamations   protect_short_excl()│
│    Song markers         protect_song()      │
│    Episode markers      protect_ep()        │
│    Character names      protect_names()     │
│    Numbers              protect_numbers()   │
│    Multi-speaker (\n-)  protect_multispkr() │
│    Short fragments      protect_short_frag()│
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 2. TRANSLATE (Opus-MT EN-DE or NLLB-600M)   │
│    Batch inference on GPU (batch_size=64)   │
│    Content extracted from ZZZ placeholders  │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 3. RESTORE + POST-PROCESS                   │
│    Canonical ZZZ restore  (3 passes)        │
│    Song markers re-apply                    │
│    Translation Memory lookup (opt-in)       │
│    Glossary correction                      │
│    Name preservation                        │
│    German fix patterns                      │
│    Title corrections                        │
│    Punctuation/spacing cleanup              │
│    Conversation memory (context window)     │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 4. ARTIFACT SCAN (algorithmic, no LLM)       │
│    Repeated words (triple, comma-sep)       │
│    Hallucinated addresses (Geistesgestörter)│
│    Fake compounds (Schweinschlachter)       │
│    Wrong titles (Mylord→mein Herr)          │
│    Sie/du mixing in same line               │
│    Bracketed alternatives, pipe artifacts   │
│    Duplicated lines                         │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 5. QA SCORING                                │
│    Suspicious line detection (score_line)   │
│    Missing glossary/names check             │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 6. POLISH (LLM — local or proxy) [optional] │
│    Only suspicious lines sent to LLM        │
│    Parallel batches (10 lines, 2 workers)   │
│    Model: Qwen2.5 7B / Gemma4 8B via Ollama │
│    Known-pattern instructions in prompt     │
│    Hallucination safeguard (rejects content)│
│    Re-glossary after correction             │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 7. LEARN (auto-fix persistence) [optional]  │
│    Suspicious line detection (Pass 2)       │
│    LLM error scan + learn_verify filter     │
│    NLLB artifact scan (candidates→fixes)    │
│    Fixes persisted to german_fixes.json     │
│    Re-runs skip expensive passes (~0.3s)    │
└──────────────────────┬──────────────────────┘
                       ▼
               Output SRT (*_ger.srt)
```

## Protection Layers

Before NLLB translation, the pipeline applies these protection passes (in order):

| Layer | ZZZ Prefix | What it protects |
|---|---|---|
| SFX | `ZZZSFX` | Bracket markers like `[Music]`, `[Xigu Lane]` |
| Short Exclamations | `ZZZEXCL` | Known short utterances: `Bravo!`, `Yes, sir.`, `Great!` |
| Song Markers | `ZZZSONG` | `♪` characters |
| Episode Markers | `ZZZEP` | `=Episode N=` markers |
| Names | `ZZZNM` | Character names from `names.json` |
| Numbers | `ZZZNU` | Time, date, money, percentages |
| Multi-speaker | `ZZZMULTI` | `\n-` between speakers (preventing NLLB from merging) |
| Short Fragments | `ZZZSHORT` | Known hallucination-prone fragments: `Mom.`, `But...`, `Shh.`, `Sir,` |

After NLLB, `_canonical_restore()` performs tolerant restoration handling case-mutated, space-split, or token-split ZZZ markers.

## Translation Memory

TM is stored in `tm/translation_memory.db`. Default: **off**. Enable with `--cache`.

```bash
# Enable TM caching
python subtranslate.py --mode full --cache

# Force fresh translation (skip TM reads)
python subtranslate.py --mode full --force
```

## Configuration

### Glossary (`config/glossary.json`)
Domain-specific term mappings applied after translation:

```json
{
  "Sir": "mein Herr",
  "Sect Leader": "Sektenführer",
  "Captain": { "default": "Hauptmann", "acceptable": ["Kapitän"] }
}
```

### Glossary Automation

**Two-step extraction (`--generate-glossary` / `--merge-glossary`):**
1. **`--generate-glossary`** — Scans all SRTs in `--input-dir`, sends text chunks to DeepSeek with a narrow domain prompt (butchery, marriage customs, military ranks, medicine, court/rebels), saves to `config/glossary_auto.json`. Character names and common nouns are excluded.
2. **`--merge-glossary`** — Batches new entries from `glossary_auto.json` into `glossary.json`. Manual entries always win. Use `--dry-run` to preview. Use `--interactive` to approve per entry.

**Self-improving mode (`--auto-glossary`):**
Before each file, the pipeline automatically:
1. Extracts domain terms from the source SRT via DeepSeek
2. Auto-merges new terms into `glossary.json` (manual entries always win)
3. Translates with the now-updated glossary

Each episode makes the glossary slightly better. Over a full season, the glossary saturates and translation quality improves without any manual intervention.

### Post-Translation QA (`--qa-report`)

Scans the first N lines (default 50) of each output file and flags:
- **Glossary coverage** — glossary terms present in English source but missing from German output
- **Name preservation** — character names from `names.json` that were lost during translation
- **Line length** — lines exceeding 42 visible characters (warning)
- **Reading speed** — CPS (characters per second) > 22 (warning) or > 25 (error)

### Timing Fix (`--fix-timing` / `--fix-aggressive`)

Runs `seconv --fix-common-errors` (Subtitle Edit CLI) on the output SRT to fix common subtitle timing issues. `--fix-aggressive` runs the fix pass twice to catch secondary issues. Gracefully warns if `seconv` is not installed.

### Short Fragments (`config/short_fragments.json`)
Fragments that NLLB hallucinates on. Protected before NLLB and replaced with correct German:

```json
{
  "shh.": "Pst.",
  "mom.": "Mutter.",
  "but...": "Aber...",
  "miss .": "Fräulein <NAME>.",
  "miss ,": "Fräulein <NAME>,"
}
```

The `<NAME>` placeholder is dynamically substituted with the actual character name marker from the current run.

## Performance

Measured on RTX 4060 Ti 16GB (CUDA 12.4), Ryzen 7 5800X, 16GB RAM:

| Metric | NLLB-600M | Opus-MT EN-DE |
|---|---|---|
| Model load | ~8.5s | ~2.7s |
| Translate throughput | 18–19 l/s | 45–50 l/s |
| 763-line episode (fast) | ~41s | ~16s |
| Full pipeline (+ Qwen polish) | ~53s | ~62s |
| VRAM usage | ~2–3 GB | ~1.5 GB |
| Batch size | 64 | 64 |
| Beam width | 4 | 4 |

| Other | Value |
|---|---|
| Proxy polisher batch (5 lines) | ~13s per batch |
| Qwen 2.5 7B polish (10 lines, parallel=2) | ~1.2s per batch |
| Gemma4 8B polish (10 lines, parallel=2) | ~2.5s per batch |
| Artifact scan (763 lines) | ~0.01s per scan |
| Ollama models | qwen2.5:7b (4.7 GB), gemma4:e4b (9.6 GB) |

## Known Limitations

- **EN → DE only** — Hardcoded language pair (NLLB supports 200+ languages, easily configurable)
- **NLLB short-line hallucination** — Very short lines like "Mom." or "But..." can produce wrong output. Mitigated via `config/short_fragments.json` dictionary
- **Proxy latency** — Remote proxy polish has ~2-10s overhead per batch regardless of batch size. For speed, use local Ollama (qwen2.5:7b or gemma4:e4b).
- **NLLB artifact patterns** — NLLB invents fake German compounds (Schweinschlachter, matrilokaler) and hallucinates titles/addresses. Mitigated by the artifact scan + LLM polish with pattern-aware prompts.
- **SRT only** — No ASS/SSA/VTT support

## Tests

```bash
# Run all tests
python -m pytest tests/ -v

# CLI test mode
python subtranslate.py --mode test
```

## License

MIT
