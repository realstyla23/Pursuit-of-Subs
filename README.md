# Subtitle Translator v4

GPU-accelerated batch subtitle translation (EN → DE) using Facebook's NLLB-600M distilled, with optional LLM polishing (local via Ollama or proxy via DeepSeek).

## Features

- **Fast** — Batch NLLB-600M translation at ~30 lines/second on an RTX 4060 Ti
- **Polish** — LLM quality pass on suspicious lines (Qwen 2.5 locally via Ollama, or DeepSeek via proxy)
- **Full** — Both passes combined for highest quality
- **Learn** — Self-improving mode: full pipeline + automatic error detection + fix persistence to `german_fixes.json`. Each run makes the next run better. Takes ~15min/episode but requires zero manual intervention.
- **English filter** — Catches and translates English words that NLLB missed, without false positives
- **Parallel batches** — Suspicious lines grouped into batches of 10, sent concurrently (2 by default)
- **Smart protection** — SFX, numbers, names, song/episode markers, multi-speaker lines, short fragments (vocatives, interjections) survive translation correctly
- **Glossary** — Domain-specific terminology enforcement
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
git clone <repo>
cd SubtitleTranslator

python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/macOS

pip install -r requirements.txt
```

### 2. Run

```bash
# Fast mode (NLLB only) — ~28s per episode
python subtranslate.py --mode fast --input-dir "path\to\subs"

# Full mode (NLLB + LLM polish) — ~80s per episode
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
  --proxy-api-key KEY   OpenCode API key (default: oc-efb2bc22...)
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
# Fast NLLB-only pass
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
├── translator/
│   ├── engine.py                Core translation pipeline (~2900 lines)
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
│   └── titles.json              Known title translations
├── assets/
│   └── screenshot_*.png         README screenshots
├── tests/
│   └── corpus/                  Regression test corpus
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
│ 2. TRANSLATE (NLLB-600M distilled)          │
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
│ 4. QA SCORING                                │
│    Suspicious line detection                │
│    Missing glossary/names check             │
└──────────────────────┬──────────────────────┘
                       ▼
┌─────────────────────────────────────────────┐
│ 5. POLISH (LLM — local or proxy) [optional]│
│    Only suspicious lines sent to LLM        │
│    Parallel batches (10 lines, 2 workers)   │
│    Local: Qwen2.5 7B via Ollama             │
│    Proxy: DeepSeek via proxy                │
│    Hallucination safeguard filters bad      │
│    English word filter catches leakage      │
│    Re-glossary after correction             │
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

| Metric | Value |
|---|---|
| NLLB throughput | 27–32 l/s (num_beams=4) |
| 763-line episode (fast) | ~27s |
| DeepSeek proxy batch (5 lines) | ~13s per batch |
| Qwen 2.5 7B polish (10 lines, parallel=2) | ~1.2s per batch |
| Full pipeline (fast + DeepSeek polish) | ~72s–2min total |
| Full pipeline (fast + Qwen polish) | ~35–45s total |
| NLLB model | NLLB-200-distilled-600M |
| Ollama model | qwen2.5:7b (4.7 GB) |
| NLLB batch size | 64 |
| Beam width | 4 |
| NLLB VRAM usage | ~2–3 GB |
| Ollama VRAM | ~5 GB |

## Known Limitations

- **EN → DE only** — Hardcoded language pair (NLLB supports 200+ languages, easily configurable)
- **NLLB short-line hallucination** — Very short lines like "Mom." or "But..." can produce wrong output. Mitigated via `config/short_fragments.json` dictionary
- **Proxy polish latency** — Each DeepSeek proxy request has ~10-15s overhead regardless of batch size
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
