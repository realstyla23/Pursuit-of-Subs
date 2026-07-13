# Subtitle Translator v4

GPU-accelerated batch subtitle translation (EN → DE) using Facebook's NLLB-600M distilled, with optional Deepseek AI polishing via OpenCode proxy.

## Features

- **Fast** — Batch NLLB-600M translation at ~30 lines/second on an RTX 4060 Ti
- **Polish** — Deepseek-powered quality pass on suspicious lines via OpenCode proxy
- **Full** — Both passes combined for highest quality
- **Smart protection** — SFX, numbers, names, song/episode markers, multi-speaker lines, short fragments (vocatives, interjections) survive translation correctly
- **Glossary** — Domain-specific terminology enforcement
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

# Full mode (NLLB + Deepseek proxy polish) — ~80s per episode
python subtranslate.py --mode full --input-dir "path\to\subs"

# Force re-translate (no TM reads)
python subtranslate.py --mode fast --force --input-dir "path\to\subs"
```

## CLI Reference

```bash
python subtranslate.py [OPTIONS]

Options:
  --mode MODE           fast, polish, full, test, benchmark, regression, llm
  --input-dir DIR       Directory containing .srt files (default: .)
  --device DEV          cuda or cpu (default: cuda)
  --batch-size N        NLLB batch size (default: 64)
  --num-beams N         Beam search width (default: 4)
  --force               Re-translate existing output, skip TM reads
  --cache               Enable Translation Memory (default: off)
  --proxy-base-url URL  OpenCode proxy URL (default: http://127.0.0.1:6446)
  --proxy-api-key KEY   OpenCode API key (default: oc-efb2bc22...)
  --resume              Resume from checkpoint
  --gui                 Launch GUI (experimental)
  --test                Run internal test suite
  --benchmark           Measure performance
```

Examples:

```bash
# Fast NLLB-only pass
python subtranslate.py --mode fast --input-dir "D:\Shows\Season 1"

# Full pipeline: NLLB + Deepseek proxy polish
python subtranslate.py --mode full --force

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
├── translator/
│   ├── engine.py                Core translation pipeline (~2900 lines)
│   ├── gui.py                   PySide6 GUI
│   └── __init__.py              Public API exports
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
│ 5. POLISH (Deepseek proxy) [optional]       │
│    Only suspicious lines sent to LLM        │
│    Hallucination safeguard filters bad      │
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
| Deepseek proxy batch | ~13s per 5-line batch |
| Full pipeline (fast + polish) | ~72s–2min total |
| Model | NLLB-200-distilled-600M |
| Batch size | 64 |
| Beam width | 4 |
| VRAM usage | ~2–3 GB |

## Known Limitations

- **EN → DE only** — Hardcoded language pair (NLLB supports 200+ languages, easily configurable)
- **NLLB short-line hallucination** — Very short lines like "Mom." or "But..." can produce wrong output. Mitigated via `config/short_fragments.json` dictionary
- **Proxy polish latency** — Each Deepseek proxy request has ~10-15s overhead regardless of batch size
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
