"""subtranslate v4.3 — LLM-powered subtitle translator with quality pipeline.

Modes:
  --fast       NLLB + SFX/glossary/validation       ~30s for 4000 lines
  --polish     Ollama polish of suspicious lines     ~seconds
  --full       Both passes
  --llm        TowerInstruct batch translate + qwen2.5:7b polish  ~3-5 min
  --test       Translate 100 lines, validate output
  --benchmark  Measure performance metrics
"""

__version__ = "4.3.0"

import argparse, json, os, re, sys, time, warnings, subprocess, shutil, threading
from dataclasses import dataclass
from pathlib import Path
import pysrt
import requests
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import transformers

transformers.logging.set_verbosity_error()

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
except (AttributeError, OSError):
    pass


def _auto_device(preferred: str) -> str:
    """Auto-detect device. Falls back to CPU if CUDA requested but unavailable."""
    if preferred == "cuda" and not torch.cuda.is_available():
        print("  [WARN] CUDA not available, falling back to CPU", flush=True)
        return "cpu"
    return preferred


def safe_open_srt(path: Path) -> pysrt.SubRipFile:
    """Open SRT file, trying UTF-8 first, then common fallback encodings."""
    for enc in ["utf-8", "utf-8-sig", "latin-1", "cp1252"]:
        try:
            return pysrt.open(str(path), encoding=enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
    return pysrt.open(str(path), encoding="utf-8")


def atomic_save(subs, path: Path, encoding: str = "utf-8"):
    """Write SRT atomically via .tmp file, then replace."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        subs.save(str(tmp), encoding=encoding)
        raw = tmp.read_bytes()
        if raw.startswith(b"\xef\xbb\xbf"):
            tmp.write_bytes(raw[3:])
        os.replace(str(tmp), str(path))
    except Exception:
        if tmp.exists():
            tmp.unlink()
        raise


@dataclass
class Config:
    src_lang: str = "eng_Latn"
    tgt_lang: str = "deu_Latn"
    device: str = "cuda"
    batch_size: int = 64
    num_beams: int = 4
    no_repeat_ngram: int = 3
    ollama_model: str = "subtitle-translator"
    ollama_host: str = "http://127.0.0.1:11434"
    input_dir: str = "."
    force: bool = False
    resume: bool = False
    use_tm: bool = False
    mode: str = "fast"
    llm_model: str = "thinkverse/towerinstruct:latest"
    polish_model: str = "qwen2.5:7b"
    llm_batch_size: int = 30
    proxy_base_url: str = ""
    proxy_api_key: str = ""
    polish_parallel: int = 2

    def __post_init__(self):
        if not self.proxy_base_url:
            self.proxy_base_url = os.environ.get("PROXY_BASE_URL", "")
        if not self.proxy_api_key:
            self.proxy_api_key = os.environ.get("PROXY_API_KEY", "")

# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------

def find_srt_files(d: str) -> list[Path]:
    seen = set()
    for pat in ["*_eng.srt", "*_en.srt", "*.srt"]:
        for f in sorted(Path(d).glob(pat)):
            if "_ger.srt" not in f.name and f.name not in seen:
                seen.add(f)
    return sorted(seen, key=lambda p: p.name)

def output_path_for(p: Path) -> Path:
    n = p.name
    for s in ["_eng.srt", "_en.srt"]:
        if n.endswith(s):
            return p.with_name(n.replace(s, "_ger.srt"))
    return p.with_name(p.stem + "_ger.srt")

_CONFIG_CACHE: dict[str, object] = {}
if getattr(sys, 'frozen', False):
    CONFIG_DIR = Path(sys._MEIPASS).resolve() / "config"
else:
    CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"

def _config_path(name: str) -> Path:
    """Resolve a config filename relative to the project config/ directory."""
    return CONFIG_DIR / name

def load_json(name: str) -> dict | list:
    if name in _CONFIG_CACHE:
        return _CONFIG_CACHE[name]
    p = _config_path(name)
    if not p.exists():
        p = Path(name)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            result = json.load(f)
            _CONFIG_CACHE[name] = result
            return result
    result = {} if name.endswith("glossary.json") else []
    _CONFIG_CACHE[name] = result
    return result

# ---------------------------------------------------------------------------
# XML helpers (used by polish pass)
# ---------------------------------------------------------------------------

LINE_TAG = re.compile(r'<LINE id="(\d+)">(.*?)</LINE>')
CONTEXT_WINDOW = 2

def _xml_escape(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

def build_xml(texts: list[str], start_id: int) -> str:
    return "\n".join(
        f'<LINE id="{start_id + i}">{t.replace(chr(10), " ").strip()}</LINE>'
        for i, t in enumerate(texts)
    )

def parse_xml(text: str) -> dict[int, str]:
    result = {}
    for m in LINE_TAG.finditer(text):
        result[int(m.group(1))] = m.group(2).strip()
    if result:
        return result
    # Subtitle-translator model echoes CONTEXT blocks then outputs corrected ones.
    # Find the LAST <current><de> for each CONTEXT id (the corrected version).
    context_blocks = re.finditer(
        r'<CONTEXT id="(\d+)">.*?</CONTEXT>', text, re.DOTALL
    )
    corrected = {}
    for m in context_blocks:
        cid = int(m.group(1))
        corrected[cid] = m.group(0)
    for cid, block in corrected.items():
        de_m = re.search(r'<current[^>]*><en>.*?</en><de>(.*?)</de></current[^>]*>', block, re.DOTALL)
        if de_m:
            result[cid] = de_m.group(1).strip()
    return result

def build_contextual_xml(batch_ids: list[int], eng_texts: list[str],
                         ger_texts: list[str], window: int = CONTEXT_WINDOW) -> str:
    """Build XML with EN+DE context for each line being polished.
    
    Wraps each target line with previous/next N subtitles so Ollama
    can produce more consistent output. Response format unchanged.
    """
    blocks = []
    for idx in batch_ids:
        start = max(0, idx - window)
        end = min(len(eng_texts), idx + window + 1)
        ctx_en = eng_texts[start:end]
        ctx_de = ger_texts[start:end]
        local_pos = idx - start

        xml = f'<CONTEXT id="{idx + 1}">\n'
        for j in range(len(ctx_en)):
            if j < local_pos:
                tag = "previous"
            elif j == local_pos:
                en_speakers = ctx_en[j].count("\n") + 1
                de_speakers = ctx_de[j].count("\n") + 1
                tag = f'current speakers="{en_speakers}"'
            else:
                tag = "next"
            en_esc = _xml_escape(ctx_en[j].replace("\n", " // "))
            de_esc = _xml_escape(ctx_de[j].replace("\n", " // "))
            xml += f"  <{tag}><en>{en_esc}</en><de>{de_esc}</de></{tag}>\n"
        xml += "</CONTEXT>"
        blocks.append(xml)
    return "\n".join(blocks)

# ---------------------------------------------------------------------------
# Pipeline profiling timer
# ---------------------------------------------------------------------------

class _Timer:
    """Simple timer for profiling pipeline stages."""
    def __init__(self):
        self._starts: dict[str, float] = {}
        self._elapsed: dict[str, float] = {}
        self._order: list[str] = []

    def start(self, stage: str):
        if stage not in self._starts:
            self._order.append(stage)
        self._starts[stage] = time.time()

    def stop(self, stage: str):
        if stage in self._starts:
            elapsed = time.time() - self._starts[stage]
            self._elapsed[stage] = self._elapsed.get(stage, 0) + elapsed
            del self._starts[stage]

    def report(self) -> str:
        lines = []
        total = sum(self._elapsed.values())
        for stage in self._order:
            e = self._elapsed.get(stage, 0)
            if e > 0:
                pct = e / total * 100 if total > 0 else 0
                lines.append(f"    {stage+':':24s} {e:.1f}s  ({pct:.0f}%)")
        if total > 0:
            lines.append(f"    {'─'*36}")
            lines.append(f"    {'TOTAL:':24s} {total:.1f}s")
        return "\n".join(lines)

    def get_time(self, stage: str) -> float:
        return self._elapsed.get(stage, 0)

    def reset(self):
        self._starts.clear()
        self._elapsed.clear()
        self._order.clear()

# ---------------------------------------------------------------------------
# 1. SFX protection layer
# ---------------------------------------------------------------------------

SFX_RE = re.compile(r'\[([^\]]+)\]')

# Placeholder validation — catches any placeholder that leaks after restoration
_PLACEHOLDER_RE = re.compile(r'ZZZ\s*\w+\s*\d+\s*ZZZ', re.IGNORECASE)


def validate_no_placeholders(texts: list[str], label: str = "output") -> int:
    """Check that no ZZZ…ZZZ placeholders remain after restoration."""
    count = 0
    for i, t in enumerate(texts):
        m = _PLACEHOLDER_RE.search(t)
        if m:
            count += 1
            print(f"  [ERROR] Placeholder leak in {label} line {i+1}: {m.group()} in {t!r}",
                  flush=True)
    if count:
        print(f"  [ERROR] {count} placeholder(s) leaked into {label}", flush=True)
    return count


# Canonical placeholder restore — tolerant of all mutation forms
_CANONICAL_ZZZ_RE = re.compile(r'ZZZ\s*(\w+?)\s*(\d+)\s*ZZZ', re.IGNORECASE)


def _canonical_restore(texts: list[str], *maps: dict[str, str]) -> list[str]:
    """Restore ALL placeholder variants using type+id then id-only fallback.

    Handles:
      ZZZSONG004ZZZ   — standard (exact type+id match)
      ZZZsONG004ZZZ   — case mutation (case-insensitive type+id match)
      ZZZ SONG016ZZZ  — space insertion (id-only fallback)
      ZZZFFX010ZZZ    — token-split type mutation (id-only fallback)

    Lookup order:
      1. Type prefix (case-insensitive) + numeric ID against each map
      2. Numeric ID alone against each map (handles type mutations)
    Multiple maps are searched in order; the first match wins.
    """
    combined_map: dict[str, str] = {}
    id_map: dict[str, str] = {}

    for m in maps:
        for key, val in m.items():
            k_match = re.fullmatch(r'ZZZ(\w+?)(\d+)ZZZ', key, re.IGNORECASE)
            if k_match:
                type_id = k_match.group(1).upper() + k_match.group(2)
                if type_id not in combined_map:
                    combined_map[type_id] = val
                nid = k_match.group(2)
                if nid not in id_map:
                    id_map[nid] = val

    restored = []
    for t in texts:
        def _replacer(m: re.Match) -> str:
            raw_prefix = m.group(1)
            num = m.group(2)
            candidate = raw_prefix.upper() + num
            if candidate in combined_map:
                return combined_map[candidate]
            if num in id_map:
                return id_map[num]
            return m.group(0)
        restored.append(_CANONICAL_ZZZ_RE.sub(_replacer, t))
    return restored


def protect_sfx(texts: list[str]) -> tuple[list[str], dict[str, str]]:
    """Replace bracket content with ZZZ-delimited placeholders (survive NLLB).
    
    Only protects actual SFX/labels (short, no sentence punctuation).
    Translatable dialogue in brackets (sentences with ., ?, or !) passes through.
    """
    sfx_map = {}
    counter = 0
    protected = []
    for t in texts:
        def repl(m):
            inner = m.group(1)
            if re.search(r'[?!]|\.(?!\d)', inner) or '\n' in inner:
                return m.group(0)
            nonlocal counter
            counter += 1
            key = f"ZZZSFX{counter:03d}ZZZ"
            sfx_map[key] = m.group(0)
            return key
        protected.append(SFX_RE.sub(repl, t))
    return protected, sfx_map


MULTI_RE = re.compile(r"\n-")


def protect_multispeaker(texts: list[str]
                         ) -> tuple[list[str], dict[str, str]]:
    """Protect \\n- between speakers with ZZZ so NLLB doesn't merge them."""
    multi_map = {}
    counter = 0
    protected = []
    for t in texts:
        def repl(m):
            nonlocal counter
            counter += 1
            key = f"ZZZMULTI{counter:03d}ZZZ"
            multi_map[key] = "\n-"
            return key
        protected.append(MULTI_RE.sub(repl, t))
    return protected, multi_map


def restore_sfx(texts: list[str], sfx_map: dict[str, str]) -> list[str]:
    """Restore bracket placeholders after NLLB (case-insensitive)."""
    restored = []
    for t in texts:
        for key, val in sfx_map.items():
            t = re.sub(re.escape(key), val, t, flags=re.IGNORECASE)
        restored.append(t)
    return restored

# ---------------------------------------------------------------------------
# 1a. Short exclamation protection layer
# ---------------------------------------------------------------------------

_SHORT_EXCL_DICT = {
    "bravo!": "Bravo!",
    "great.": "Prima.",
    "great!": "Prima!",
    "alright.": "In Ordnung.",
    "alright!": "In Ordnung!",
    "okay.": "Okay.",
    "okay!": "Okay!",
    "yes.": "Ja.",
    "yes!": "Ja!",
    "no.": "Nein.",
    "no!": "Nein!",
    "no one.": "Niemand.",
    "shoo, shoo, shoo.": "Husch, husch, husch.",
    "sorry.": "Es tut mir leid.",
    "come on.": "Komm schon.",
    "come on!": "Komm schon!",
    "hurry up!": "Beeil dich!",
    "go.": "Los.",
    "no way.": "Unmöglich.",
    "nonsense.": "Unsinn.",
    "fine.": "Gut.",
    "ma'am,": "Meine Dame,",
    "ma'am.": "Meine Dame.",
    "madam,": "Meine Dame,",
    "madam.": "Meine Dame.",
    "miss,": "Fräulein,",
    "miss.": "Fräulein.",
}

# Pattern: (Yes|No), <address> → NLLB hallucinates by adding invented content
# and dash-splitting single-speaker lines. Catch generically before NLLB.
_AFFIRM_ADDRESS_RE = re.compile(
    r'^\s*(-\s+)?(Yes|Yeah|Yep|No|Nope),\s*(sir|ma\'?am|madam|miss)\s*[.!]?\s*$',
    re.IGNORECASE,
)
_ADDRESS_TRANS: dict[str, str] = {
    "sir": "mein Herr",
    "ma'am": "meine Dame",
    "madam": "meine Dame",
    "miss": "Fräulein",
}


def _short_excl_replacement(part: str) -> str | None:
    stripped = part.strip()
    text = stripped[1:].strip() if stripped.startswith("-") else stripped
    # Exact dict match first
    result = _SHORT_EXCL_DICT.get(text.lower())
    if result is not None:
        return result
    # Generic fallback: affirmation + address pattern that NLLB hallucinates
    m = _AFFIRM_ADDRESS_RE.match(stripped)
    if m:
        affirmation = m.group(2).lower()
        address = m.group(3).lower()
        de_affirmation = "Ja" if affirmation in ("yes", "yeah", "yep") else "Nein"
        de_address = _ADDRESS_TRANS.get(address, address)
        # Preserve trailing punctuation from source
        punct = ""
        raw = stripped.rstrip()
        if raw.endswith("!"):
            punct = "!"
        elif raw.endswith("."):
            punct = "."
        return f"{de_affirmation}, {de_address}{punct}"
    return None


def _format_short_excl_part(source_part: str, replacement: str) -> str:
    return ("-" if source_part.strip().startswith("-") else "") + replacement


def apply_short_exclamation_overrides(eng_texts: list[str], ger_texts: list[str]) -> int:
    """Force static translations for known standalone short speaker lines."""
    count = 0
    for i in range(min(len(eng_texts), len(ger_texts))):
        en_parts = eng_texts[i].split('\n')
        de_parts = ger_texts[i].split('\n')
        if len(en_parts) != len(de_parts):
            continue
        changed = False
        for j, en_part in enumerate(en_parts):
            replacement = _short_excl_replacement(en_part)
            if replacement is None:
                continue
            new_part = _format_short_excl_part(en_part, replacement)
            if de_parts[j].strip() != new_part.strip():
                de_parts[j] = new_part
                changed = True
        if changed:
            ger_texts[i] = '\n'.join(de_parts)
            count += 1
    return count


def _is_all_short_exclamations(text: str) -> bool:
    parts = [p for p in text.split('\n') if p.strip()]
    return bool(parts) and all(_short_excl_replacement(p) is not None for p in parts)

def protect_short_exclamations(texts: list[str]) -> tuple[list[str], dict[str, str]]:
    """Protect known short exclamations in any context (single or multi-speaker).
    
    Splits each line by \n, checks each segment individually, and protects
    matched segments with ZZZEXCL markers. This lets mixed multi-speaker lines
    like "-Great.\\n-Alright." have only "Alright." protected while "-Great."
    passes through NLLB normally.
    """
    excl_map = {}
    counter = 0
    protected = []
    for t in texts:
        parts = t.split('\n')
        new_parts = []
        changed = False
        for part in parts:
            replacement = _short_excl_replacement(part)
            if replacement is not None:
                counter += 1
                key = f"ZZZEXCL{counter:03d}ZZZ"
                excl_map[key] = replacement
                new_parts.append(_format_short_excl_part(part, key))
                changed = True
            else:
                new_parts.append(part)
        if changed:
            protected.append('\n'.join(new_parts))
        else:
            protected.append(t)
    return protected, excl_map


def protect_short_fragments(texts: list[str], glossary: dict,
                            short_fragments: dict[str, str]
                            ) -> tuple[list[str], dict[str, str]]:
    """Protect short fragments that NLLB hallucinates.

    Two checks (checked in order):
      1. Single-word vocative in glossary (e.g., "Sir," → "mein Herr,")
      2. Curated short_fragments dictionary matched against ZZZ-stripped
         visible text (e.g., "Miss Fan." → "Fräulein <NAME>.")

    The <NAME> placeholder is substituted with actual ZZZ markers captured
    from the current run's text, avoiding marker-numbering-drift across runs.
    """
    short_map = {}
    counter = 0
    gloss_flat = {}
    for k, v in glossary.items():
        default, _ = _resolve_glossary_entry(v)
        if default:
            gloss_flat[k.lower()] = default

    protected = []
    for t in texts:
        stripped = t.strip()
        if not stripped or '\n' in stripped:
            protected.append(t)
            continue

        # Capture actual ZZZ markers from this run's text (dynamic IDs)
        zzz_markers = re.findall(r'ZZZ\w+ZZZ', stripped)

        # Check 1: Single-word vocative in glossary (no ZZZ in line)
        if not zzz_markers:
            m = re.match(r'^(\w+),$', stripped)
            if m and m.group(1).lower() in gloss_flat:
                counter += 1
                key = f"ZZZSHORT{counter:03d}ZZZ"
                de_term = gloss_flat[m.group(1).lower()]
                short_map[key] = f"{de_term},"
                protected.append(key)
                continue

        # Check 2: Short fragments dictionary (match against ZZZ-stripped text)
        content_visible = re.sub(r'ZZZ\w+ZZZ', '', stripped).strip()
        if not content_visible:
            protected.append(t)
            continue

        content_normalized = re.sub(r'\s+', ' ', content_visible)
        key_fragment = content_normalized.lower()

        if key_fragment in short_fragments:
            counter += 1
            key = f"ZZZSHORT{counter:03d}ZZZ"
            de_result = short_fragments[key_fragment]
            for zzz in zzz_markers:
                de_result = de_result.replace('<NAME>', zzz, 1)
            short_map[key] = de_result
            protected.append(key)
        else:
            protected.append(t)

    return protected, short_map


# ---------------------------------------------------------------------------
# 1b. Song marker protection layer
# ---------------------------------------------------------------------------

SONG_MARKER = '\u266a'

def protect_song_markers(texts: list[str]) -> tuple[list[str], dict[str, str]]:
    """Replace \u266a characters with ZZZ-delimited placeholders."""
    song_map = {}
    counter = 0
    protected = []
    for t in texts:
        new_t = t
        while SONG_MARKER in new_t:
            counter += 1
            key = f"ZZZSONG{counter:03d}ZZZ"
            song_map[key] = SONG_MARKER
            new_t = new_t.replace(SONG_MARKER, key, 1)
        protected.append(new_t)
    return protected, song_map

def restore_song_markers(texts: list[str], song_map: dict[str, str]) -> list[str]:
    """Restore song marker placeholders after NLLB (case-insensitive)."""
    restored = []
    for t in texts:
        for key, val in song_map.items():
            t = re.sub(re.escape(key), val, t, flags=re.IGNORECASE)
        restored.append(t)
    return restored

def apply_song_markers(ger_texts: list[str], eng_texts: list[str]) -> int:
    """Post-hoc: copy \u266a markers from English source to German output.
    NLLB often drops the placeholders, so this is a reliable fallback.
    """
    count = 0
    for i, (en, de) in enumerate(zip(eng_texts, ger_texts)):
        if SONG_MARKER not in en:
            continue
        if SONG_MARKER in de:
            continue
        new_de = de
        if en.startswith(SONG_MARKER) and not new_de.startswith(SONG_MARKER):
            new_de = SONG_MARKER + new_de.lstrip()
        if en.endswith(SONG_MARKER) and not new_de.endswith(SONG_MARKER):
            new_de = new_de.rstrip() + SONG_MARKER
        if new_de != de:
            ger_texts[i] = new_de
            count += 1
    return count

# ---------------------------------------------------------------------------
# 1a2. Episode/series marker protection (=Episode N=, =Staffel N=, etc.)
# ---------------------------------------------------------------------------

EPISODE_EQ_RE = re.compile(r'=(Episode|Staffel|Season|Folge)\s*\d*\s*=', re.IGNORECASE)

def protect_episode_markers(texts: list[str]) -> tuple[list[str], dict[str, str]]:
    """Replace =Episode N= markers with ZZZ-delimited placeholders."""
    ep_map = {}
    counter = 0
    protected = []
    for t in texts:
        def repl(m):
            nonlocal counter
            counter += 1
            key = f"ZZZEP{counter:03d}ZZZ"
            ep_map[key] = m.group(0)
            return key
        protected.append(EPISODE_EQ_RE.sub(repl, t))
    return protected, ep_map

def restore_episode_markers(texts: list[str], ep_map: dict[str, str]) -> list[str]:
    """Restore episode marker placeholders after NLLB (case-insensitive)."""
    restored = []
    for t in texts:
        for key, val in ep_map.items():
            t = re.sub(re.escape(key), val, t, flags=re.IGNORECASE)
        restored.append(t)
    return restored

# ---------------------------------------------------------------------------
# 1b. Name protection layer
# ---------------------------------------------------------------------------

def protect_names(texts: list[str], names: list[str]) -> tuple[list[str], dict[str, str]]:
    """Replace character names with ZZZ-delimited placeholders (survive NLLB).
    Multi-word names matched first (descending length) to avoid partial overlap.
    Possessive "'s" is consumed so NLLB sees a clean noun phrase.
    After individual name replacement, title+name combinations (e.g. "Miss Fan")
    are merged into a single ZZZNM marker so NLLB can't split them.
    """
    name_map = {}
    counter = 0
    sorted_names = sorted(names, key=lambda n: -len(n))
    protected = []
    for t in texts:
        def repl(m):
            nonlocal counter
            counter += 1
            key = f"ZZZNM{counter:03d}ZZZ"
            name_map[key] = m.group(1)
            return key
        for name in sorted_names:
            pattern = re.compile(
                r'(?<!\w)(' + re.escape(name) + r")(?:'s)?(?!\w)", re.IGNORECASE
            )
            t = pattern.sub(repl, t)
        protected.append(t)

    # Merge "Title + ZZZNM\d+ZZZ" into a single combined marker
    # so NLLB never sees the title word adjacent to a placeholder.
    _TITLE_WORDS = [
        "miss", "mrs", "mr", "ms", "sir", "lady", "lord",
        "king", "queen", "prince", "princess", "master",
    ]
    _TITLE_PAT = re.compile(
        r'(?<!\w)(' + '|'.join(re.escape(w) + r'\.?' for w in _TITLE_WORDS)
        + r')\s+(ZZZNM\d+ZZZ)',
        re.IGNORECASE,
    )
    for i, t in enumerate(protected):
        new_t = t
        while True:
            m = _TITLE_PAT.search(new_t)
            if not m:
                break
            counter += 1
            key = f"ZZZNM{counter:03d}ZZZ"
            title_word = m.group(1)
            name_marker = m.group(2)
            actual_name = name_map.get(name_marker, "")
            name_map[key] = f"{title_word} {actual_name}"
            del name_map[name_marker]
            new_t = new_t[:m.start()] + key + new_t[m.end():]
        protected[i] = new_t

    return protected, name_map


def restore_names(texts: list[str], name_map: dict[str, str]) -> list[str]:
    """Restore name placeholders after NLLB (case-insensitive)."""
    restored = []
    for t in texts:
        for key, val in name_map.items():
            t = re.sub(re.escape(key), val, t, flags=re.IGNORECASE)
        restored.append(t)
    return restored

# ---------------------------------------------------------------------------
# 1c. Literal number protection layer
# ---------------------------------------------------------------------------

# Patterns for literal number values that should survive NLLB
_NUM_PATTERNS = [
    # Time: 10:30, 7:00 PM, 2:00
    (re.compile(r'\b\d{1,2}:\d{2}(?:\s*[APap][Mm])?\b'), None),
    # Year: 4-digit years (1900-2099)
    (re.compile(r'\b(?:19|20)\d{2}\b'), None),
    # Money with currency symbol
    (re.compile(r'[$€£¥]\s*\d+(?:[.,]\d+)*\b'), None),
    # Money with currency word
    (re.compile(r'\b\d+(?:[.,]\d+)*\s*(?:yuan|dollars?|euros?|USD|EUR)\b', re.IGNORECASE), None),
    # Percentage
    (re.compile(r'\b\d+(?:[.,]\d+)?%'), None),
    # Phone-like: 123-456-7890 or 123.456.7890
    (re.compile(r'\b\d{3}[-.]\d{3,4}[-.]\d{3,4}\b'), None),
    # Thousands separator: 1,000 or 10,000 (not years)
    (re.compile(r'\b\d{1,3}(?:,\d{3})+\b'), None),
    # Room/Chapter/Grade/Class + number
    (re.compile(r'\b(Room|Chapter|Section|Grade|Class|No\.?)\s+\d+\b', re.IGNORECASE), None),
    # Episode + number (catch any not caught by title preservation)
    (re.compile(r'\b(Episode|Ep\.)\s+\d+\b', re.IGNORECASE), None),
]

def protect_numbers(texts: list[str]) -> tuple[list[str], dict[str, str]]:
    """Replace literal number patterns with ZZZ-delimited placeholders."""
    num_map = {}
    counter = 0
    protected = []
    for t in texts:
        def repl(m):
            nonlocal counter
            counter += 1
            key = f"ZZZNU{counter:03d}ZZZ"
            num_map[key] = m.group(0)
            return key
        for pattern, _ in _NUM_PATTERNS:
            t = pattern.sub(repl, t)
        protected.append(t)
    return protected, num_map


def restore_numbers(texts: list[str], num_map: dict[str, str]) -> list[str]:
    """Restore number placeholders after NLLB (case-insensitive)."""
    restored = []
    for t in texts:
        for key, val in num_map.items():
            t = re.sub(re.escape(key), val, t, flags=re.IGNORECASE)
        restored.append(t)
    return restored

# ---------------------------------------------------------------------------
# 2. Glossary loading & correction
# ---------------------------------------------------------------------------

def load_glossary() -> dict:
    return load_json("glossary.json")

def _resolve_glossary_entry(v) -> tuple[str, list[str]]:
    """Extract (default, acceptable_list) from old or new glossary format."""
    if isinstance(v, dict):
        default = v.get("default", "")
        acceptable = v.get("acceptable", [])
        return default, acceptable
    return v, []


# Auto-glossary — learns term corrections from Ollama polish over time
_AUTO_GLOSSARY_PATH = _config_path("auto_glossary.json")
_OLLAMA_CACHE_PATH = _config_path("ollama_cache.json")


def _load_auto_glossary() -> dict[str, str]:
    if _AUTO_GLOSSARY_PATH.exists():
        try:
            with open(_AUTO_GLOSSARY_PATH, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_auto_glossary(ag: dict[str, str]):
    try:
        _AUTO_GLOSSARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_AUTO_GLOSSARY_PATH, "w", encoding="utf-8") as f:
            json.dump(ag, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"  [WARN] Could not save auto-glossary: {e}", flush=True)


def _learn_from_fixes(eng_texts: list[str], ger_texts: list[str],
                      fixes: dict[int, str], auto_glossary: dict[str, str]) -> int:
    """Extract term-level corrections from Ollama fixes and store in auto-glossary.

    Heuristic: if an English content word appears literally (untranslated) in the
    NLLB output but is replaced in the Ollama correction, learn the replacement
    by matching word positions between NLLB and corrected text.
    """
    _SKIP = {"the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
             "of", "with", "by", "from", "is", "are", "was", "were", "be", "been",
             "has", "have", "had", "do", "does", "did", "will", "would", "could",
             "should", "may", "might", "shall", "can", "not", "no", "yes", "it",
             "its", "this", "that", "these", "those", "i", "you", "he", "she",
             "we", "they", "my", "your", "his", "her", "our", "their", "me",
             "him", "us", "them", "what", "who", "which", "where", "when", "why",
             "how", "all", "each", "every", "some", "any", "many", "much",
             "more", "most", "few", "less", "very", "too", "so", "just", "also",
             "only", "now", "then", "here", "there", "up", "down", "out", "off",
             "over", "under", "again", "further", "once", "than", "as", "if",
             "because", "while", "though", "until", "about", "after", "before",
             "between", "through", "during", "without", "along", "around",
             "near", "among", "across", "behind", "beyond", "inside", "outside",
             "onto", "upon"}

    added = 0
    for idx, corr in fixes.items():
        en = eng_texts[idx]
        nllb = ger_texts[idx]
        if corr == nllb:
            continue
        nllb_words = nllb.split()
        corr_words = corr.split()
        for word in en.split():
            w = word.strip(",.!?;:'\"()[]♪-").lower()
            if len(w) < 3 or w in _SKIP:
                continue
            if w in nllb.lower() and w not in corr.lower():
                # English word was literally in NLLB output; find replacement
                # by matching word positions
                for i in range(min(len(nllb_words), len(corr_words))):
                    if nllb_words[i].lower().strip(",.!?") == w:
                        repl = corr_words[i].strip(",.!?")
                        if repl.lower() != w and w not in auto_glossary:
                            auto_glossary[w] = repl
                            added += 1
                        break

    if added:
        _save_auto_glossary(auto_glossary)
    return added


def apply_glossary(eng_texts: list[str], ger_texts: list[str], glossary: dict) -> int:
    """Source-aware glossary correction. Check EN input against DE output.
    Supports both old (str) and new (dict with default/acceptable) formats.
    Uses word-boundary matching, avoids double commas, corrects capitalization.
    """
    gloss_parsed = {}
    for k, v in glossary.items():
        default, acceptable = _resolve_glossary_entry(v)
        if default:
            gloss_parsed[k.lower()] = (default, [d.lower() for d in acceptable])

    gloss_sorted = sorted(gloss_parsed.items(), key=lambda x: len(x[0]), reverse=True)

    count = 0
    for i in range(min(len(eng_texts), len(ger_texts))):
        en, de = eng_texts[i], ger_texts[i]
        en_lower = en.lower()
        de_lower = de.lower()

        for eng_term, (ger_term, acceptable) in gloss_sorted:
            if not re.search(r'(?<!\w)' + re.escape(eng_term) + r"(?!'s)(?!\w)", en_lower):
                continue

            all_check = [ger_term.lower()] + acceptable
            already_correct = any(
                re.search(r'(?<!\w)' + re.escape(w) + r"(?!'s)(?!\w)", de_lower)
                for w in all_check
            )
            if already_correct:
                continue

            new_de = de

            for wrong_form in [eng_term.title(), eng_term, eng_term.upper()]:
                pattern = re.compile(r'(?<!\w)' + re.escape(wrong_form) + r"(?!'s)(?!\w)")
                if pattern.search(de):
                    new_de = pattern.sub(ger_term, de)
                    break

            if new_de != de:
                ger_texts[i] = new_de
                count += 1
                continue

            en_stripped = en.strip()
            en_stripped_lower = en_stripped.lower()
            en_clean = en_stripped_lower.rstrip(" \t\n\r.!?,;:")
            en_matches_start = (en_stripped_lower.startswith(eng_term + " ")
                                or en_stripped_lower.startswith(eng_term + ","))
            en_matches_end = en_clean.endswith(" " + eng_term) or en_clean == eng_term
            en_matches_anywhere = (
                re.search(r'(?<!\w)' + re.escape(eng_term) + r"(?!'s)(?!\w)", en_stripped_lower)
                is not None
            )
            if en_matches_start or en_matches_end or en_matches_anywhere:
                if not re.search(r'(?<!\w)' + re.escape(ger_term) + r"(?!'s)(?!\w)", de_lower):
                    en_core = en_stripped.strip(" \t\n\r.!?,;:-")
                    if en_core.lower() == eng_term:
                        trailing = en_stripped[-1] if en_stripped[-1:] in '.!?' else ''
                        ger_texts[i] = ger_term + trailing
                        count += 1
                        continue
                    if new_de:
                        leading = new_de[0].lower() + new_de[1:]
                        leading = re.sub(r'^-\s*', '', leading).strip()
                        ger_texts[i] = f"{ger_term}, {leading}"
                    else:
                        ger_texts[i] = f"{ger_term}."
                    count += 1

    return count


_GLOSSARY_EXTRACTION_PROMPT = """\
You are a terminology extraction specialist for Chinese period dramas.
Extract recurring DOMAIN-SPECIFIC terms from the following subtitles.

FOCUS TOPICS: {focus_topics}

CRITICAL RULES:
1. IGNORE all character names, place names, and proper nouns.
2. IGNORE common nouns (e.g. "table", "house", "money") unless they carry specific cultural or technical meaning in this context.
3. Output ONLY a valid JSON object: {{"English term": "German translation"}}.
4. If no domain terms are found in this chunk, return {{}}.
5. Do not include explanations or markdown.
"""


def generate_glossary(
    srt_paths: list[Path],
    cfg: Config,
    focus_topics: str = (
        "butchery trades and meat processing, "
        "matrilocal marriage customs (ruzhu), "
        "Qing-style military ranks and titles, "
        "traditional medicine and herbal dosages, "
        "court factions and rebellion terminology"
    ),
) -> dict[str, str]:
    """Scan SRT files and extract domain-specific glossary terms via DeepSeek.

    Returns a flat dict of {english_term: german_translation} from all chunks.
    Saves the result to config/glossary_auto.json as a side effect.
    """
    polisher = load_polisher(cfg)
    if polisher is None:
        print("  [ERROR] Could not load polisher. Check proxy settings.")
        return {}
    session, chat_url, model_name, api_key = polisher

    # Extract plain text from all SRTs
    all_texts: list[str] = []
    for p in srt_paths:
        try:
            subs = safe_open_srt(p)
            all_texts.extend(sub.text for sub in subs)
        except Exception as e:
            print(f"  [WARN] Skipping {p.name}: {e}")

    if not all_texts:
        print("  [ERROR] No text found in SRT files.")
        return {}

    full_text = "\n".join(all_texts)
    chunk_size = 12000
    chunks = [full_text[i:i + chunk_size] for i in range(0, len(full_text), chunk_size)]
    print(f"  Extracted {len(full_text)} chars, {len(chunks)} chunk(s)", flush=True)

    combined: dict[str, str] = {}
    system_prompt = _GLOSSARY_EXTRACTION_PROMPT.format(focus_topics=focus_topics)

    for idx, chunk in enumerate(chunks):
        print(f"  Chunk {idx + 1}/{len(chunks)}...", end=" ", flush=True)
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Extract terms from this subtitle text:\n\n{chunk}"},
            ],
            "temperature": 0.1,
            "max_tokens": 2048,
            "response_format": {"type": "json_object"},
        }
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        result = _ollama_chat(session, chat_url, payload, timeout=120, headers=headers)
        if result is None:
            print("FAILED", flush=True)
            continue
        try:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "{}")
            chunk_glossary = json.loads(content)
            if isinstance(chunk_glossary, dict):
                combined.update(chunk_glossary)
                print(f"{len(chunk_glossary)} terms", flush=True)
            else:
                print(f"unexpected type: {type(chunk_glossary).__name__}", flush=True)
        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"parse error: {e}", flush=True)

    # Save to glossary_auto.json
    out_path = _config_path("glossary_auto.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(combined, f, ensure_ascii=False, indent=2)
    print(f"  Saved {len(combined)} terms to {out_path}", flush=True)
    return combined


def merge_glossary_auto(
    auto_path: Path | None = None,
    interactive: bool = False,
    dry_run: bool = False,
) -> int:
    """Merge glossary_auto.json into glossary.json. Manual entries always win.

    Args:
        auto_path: Path to auto-generated glossary. Defaults to config/glossary_auto.json.
        interactive: If True, prompt per new entry before adding.
        dry_run: If True, only print diff without modifying.

    Returns: Number of new terms added (or would be added in dry-run mode).
    """
    if auto_path is None:
        auto_path = _config_path("glossary_auto.json")
    manual_path = _config_path("glossary.json")

    if not auto_path.exists():
        print(f"  [ERROR] {auto_path} not found. Run --generate-glossary first.")
        return 0

    with open(auto_path, encoding="utf-8") as f:
        auto_glossary: dict = json.load(f)
    with open(manual_path, encoding="utf-8") as f:
        manual_glossary: dict = json.load(f)

    # Find new keys (in auto but not in manual)
    new_terms = {k: v for k, v in auto_glossary.items() if k not in manual_glossary}
    if not new_terms:
        print("  No new terms to merge.")
        return 0

    print(f"\n  Found {len(new_terms)} new term(s):")
    for k, v in sorted(new_terms.items()):
        print(f"    {k:40s} → {v}")

    if dry_run:
        print(f"\n  Dry-run: {len(new_terms)} term(s) would be added. No changes written.")
        return len(new_terms)

    if interactive:
        accepted: dict[str, str] = {}
        print()
        for k, v in sorted(new_terms.items()):
            answer = input(f"  Add '{k}' → '{v}'? [Y/n/e(dit)]: ").strip().lower()
            if answer in ("", "y", "yes"):
                accepted[k] = v
            elif answer.startswith("e"):
                new_v = input(f"    Edit translation for '{k}': ").strip()
                if new_v:
                    accepted[k] = new_v
            # n/no → skip
        new_terms = accepted

    if not new_terms:
        print("  No terms accepted.")
        return 0

    # Merge (manual wins, but these are all new so no conflicts)
    merged = dict(manual_glossary)
    merged.update(new_terms)
    with open(manual_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)
    print(f"  Added {len(new_terms)} term(s) to {manual_path}")
    return len(new_terms)


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


def qa_report(
    srt_path: Path,
    eng_srt_path: Path,
    glossary: dict,
    names: list[str],
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

    gloss_lookup: dict[str, str] = {}
    for k, v in glossary.items():
        default, acceptable = _resolve_glossary_entry(v)
        if default:
            gloss_lookup[k.lower()] = (default.lower(), [a.lower() for a in acceptable])

    name_set = set(n.lower() for n in names)

    errors = 0
    warnings = 0
    details: list[str] = []

    max_lines = min(spotcheck_lines, len(ger_subs))
    for i in range(max_lines):
        ger_sub = ger_subs[i]
        ger_text = ger_sub.text.strip()
        ger_text_no_tags = re.sub(r'<[^>]+>', '', ger_text)
        visible_chars = len(ger_text_no_tags)

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
                            f"  Line {i+1}: '{eng_term}' -> '{ger_term}' missing in German output"
                        )

        # 2. Name preservation — names should survive translation unchanged
        ger_lower = ger_text.lower()
        if eng_text:
            for name in name_set:
                name_in_en = re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', eng_text)
                name_in_de = re.search(r'(?<!\w)' + re.escape(name) + r'(?!\w)', ger_lower)
                if name_in_en and not name_in_de:
                    errors += 1
                    details.append(
                        f"  Line {i+1}: Name '{name}' present in English but missing from German"
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


# ---------------------------------------------------------------------------
# 3. Name preservation
# ---------------------------------------------------------------------------

def load_names() -> list[str]:
    return load_json("names.json")

def preserve_names(eng_texts: list[str] | None, ger_texts: list[str],
                   names: list[str], glossary: dict | None = None) -> int:
    """Restore character names that NLLB may have translated.
    With eng_texts, can detect names dropped entirely by NLLB and prepend them.
    Uses pre-compiled regex patterns for performance.
    """
    count = 0
    patterns = []
    for name in names:
        parts = name.split()
        compiled_parts = [
            re.compile(r'(?<!\w)' + re.escape(part) + r'(?!\w)')
            for part in parts
        ]
        patterns.append((name, compiled_parts))

    for i, t in enumerate(ger_texts):
        new_t = t
        for name, compiled_parts in patterns:
            for part, pat in zip(name.split(), compiled_parts):
                new_t = pat.sub(part, new_t)
        if new_t != t:
            ger_texts[i] = new_t
            count += 1

    # Build glossary lookup: en_term_lower -> set of german_lower terms
    gloss_en_to_de: dict[str, set[str]] = {}
    if glossary:
        for k, v in glossary.items():
            default, acceptable = _resolve_glossary_entry(v)
            terms: set[str] = set()
            if default:
                terms.add(default.lower())
            terms.update(a.lower() for a in acceptable)
            gloss_en_to_de[k.lower()] = terms

    # Detect names that were dropped entirely by NLLB (no parts remain)
    if eng_texts is not None:
        for i, (en, de) in enumerate(zip(eng_texts, ger_texts)):
            de_lower = de.lower()
            for name in names:
                en_has = all(p.lower() in en.lower() for p in name.split())
                if not en_has:
                    continue
                de_has = all(p.lower() in de_lower for p in name.split())
                if de_has:
                    continue
                # Check if glossary translated a name part (e.g. "Mr."→"Herr")
                parts = name.split()
                gloss_covered = True
                for p in parts:
                    p_lower = p.lower()
                    if p_lower in de_lower:
                        continue
                    if p_lower in gloss_en_to_de:
                        if any(gterm in de_lower for gterm in gloss_en_to_de[p_lower]):
                            continue
                    gloss_covered = False
                    break
                if gloss_covered:
                    continue
                if de:
                    ger_texts[i] = f"{name}, {de[0].lower() + de[1:]}"
                else:
                    ger_texts[i] = name
                count += 1

    return count

# ---------------------------------------------------------------------------
# 4a. German subtitle phrase fixes
# ---------------------------------------------------------------------------

def load_german_fixes() -> list[dict]:
    return load_json("german_fixes.json")

def load_short_fragments() -> dict[str, str]:
    return load_json("short_fragments.json")

def apply_german_fixes(ger_texts: list[str], fixes: list[dict]) -> int:
    """Fix known awkward German translations."""
    count = 0
    for i, t in enumerate(ger_texts):
        for fix in fixes:
            f = fix["find"]
            r = fix["replace"]
            if f.lower() in t.lower():
                new_t = t.replace(f, r)
                if new_t != t:
                    ger_texts[i] = new_t
                    count += 1
    return count


_ENG_MAP = {
    "escort": "Eskorte",
    "escorts": "Eskorten",
}


def _filter_english_words(eng_texts: list[str], ger_texts: list[str]) -> int:
    """Replace common English words that NLLB failed to translate.
    Only replaces when the English source actually contains the trigger word,
    to avoid false positives on shared vocabulary.
    """
    count = 0
    for i, (en, de) in enumerate(zip(eng_texts, ger_texts)):
        en_lower = en.lower()
        de_lower = de.lower()
        for eng_word, ger_word in _ENG_MAP.items():
            if eng_word in de_lower and eng_word in en_lower:
                ger_texts[i] = re.sub(
                    r'(?<!\w)' + re.escape(eng_word) + r'(?!\w)',
                    ger_word, ger_texts[i], flags=re.IGNORECASE
                )
                count += 1
    if count:
        print(f"    english filter: {count} fix(es)", flush=True)
    return count


def cleanup_subtitles(ger_texts: list[str]) -> int:
    """Normalize subtitle typography (space before punct, duplicates, etc.)."""
    count = 0
    for i, t in enumerate(ger_texts):
        new_t = t
        # 1. Space before punctuation → remove space
        new_t = re.sub(r'\s+([.,!?;:])', r'\1', new_t)
        # 2. Duplicate spaces → single space
        new_t = re.sub(r'  +', ' ', new_t)
        # 3. Repeated punctuation (3+ same punctuation chars → 2)
        new_t = re.sub(r'(\.\.\.)(\.+)', r'\1', new_t)  # .... → ...
        new_t = re.sub(r'([!?,;:]){3,}', r'\1\1', new_t)  # ??!? etc.
        # 4. Leading/trailing whitespace
        new_t = new_t.strip()
        if new_t != t:
            ger_texts[i] = new_t
            count += 1
    return count

# ---------------------------------------------------------------------------
# 4b. Title preservation
# ---------------------------------------------------------------------------

EPISODE_MARKER = re.compile(r"\[(Episode|Trailer|Preview|Teaser)\s*\d*\]", re.IGNORECASE)

def load_titles() -> list[str]:
    return load_json("titles.json")

def preserve_titles(eng_texts: list[str], ger_texts: list[str], titles: list[str]) -> int:
    """Restore episode markers and known titles."""
    count = 0
    for i, (en, de) in enumerate(zip(eng_texts, ger_texts)):
        en_s = en.strip()
        # Restore episode markers
        if EPISODE_MARKER.match(en_s):
            if en_s != de.strip():
                ger_texts[i] = en_s
                count += 1
        # Restore known titles
        elif titles:
            for title in titles:
                if title.lower() in en_s.lower() and title.lower() not in de.lower():
                    ger_texts[i] = en_s
                    count += 1
                    break
    return count

# ---------------------------------------------------------------------------
# 5. Validation
# ---------------------------------------------------------------------------

def _normalize_line(text: str) -> str:
    """Strip punctuation, collapse whitespace, lowercase."""
    return re.sub(r'[^\w\s]', '', text).strip().lower()

def validate(eng_subs, ger_subs, fpath: str):
    issues = []
    n_eng, n_ger = len(eng_subs), len(ger_subs)
    if n_eng != n_ger:
        issues.append(f"count mismatch: {n_eng} EN vs {n_ger} DE")

    for i in range(min(n_eng, n_ger)):
        es, gs = eng_subs[i], ger_subs[i]
        # Timestamp check
        if str(es.start) != str(gs.start) or str(es.end) != str(gs.end):
            issues.append(f"line {i+1}: timestamp changed")
        # Empty translation
        if not gs.text.strip():
            issues.append(f"line {i+1}: empty translation")
        # Broken UTF-8 (invalid sequences)
        try:
            gs.text.encode("utf-8").decode("utf-8")
        except:
            issues.append(f"line {i+1}: broken UTF-8")

    # Duplicate blocks (3+ consecutive identical German subs)
    # Only flag when the English source is NOT also repetitive,
    # to avoid false positives on dramatic repetition in the source material.
    for i in range(n_ger - 2):
        if not (ger_subs[i].text == ger_subs[i+1].text == ger_subs[i+2].text
                and ger_subs[i].text.strip()):
            continue
        # Check if English source is also repetitive (same normalized form)
        en_norms = {_normalize_line(eng_subs[j].text) for j in range(i, i + 3)}
        if len(en_norms) == 1 and all(en_norms):
            continue  # EN is equally repetitive — expected duplication
        issues.append(f"lines {i+1}-{i+3}: duplicated text")

    if issues:
        print(f"  VALIDATION ({fpath}):")
        for iss in issues[:10]:
            print(f"    {iss}")
        if len(issues) > 10:
            print(f"    ... and {len(issues)-10} more")
    else:
        print(f"  VALIDATION ({fpath}): OK")

    # Check for placeholder leaks (imperative — no ZZZ…ZZZ should survive)
    ger_texts = [gs.text for gs in ger_subs]
    leaked = validate_no_placeholders(ger_texts, label=fpath)
    if leaked:
        issues.append(f"{leaked} placeholder(s) leaked")
    return len(issues) == 0

# ---------------------------------------------------------------------------
# 5b. QA scoring
# ---------------------------------------------------------------------------

_END_PUNCT = re.compile(r'[.!?]+$')


def score_line(en_text: str, de_text: str, glossary: dict, names: list[str],
               titles: list[str]) -> dict:
    """Score a single translated line. Higher = more issues."""
    score = 0
    reasons = []
    en_s = en_text.strip()
    de_s = de_text.strip()

    # +10 empty translation
    if not de_s:
        score += 10
        reasons.append("empty")

    # +5 glossary missing: EN has glossary term but DE lacks correct translation
    glossary_defaults = {}
    for k, v in glossary.items():
        if isinstance(v, dict):
            d = v.get("default", "")
            acceptable = [d] + v.get("acceptable", [])
            glossary_defaults[k] = acceptable
        elif isinstance(v, str) and v:
            glossary_defaults[k] = [v]

    for eng_term, de_terms in glossary_defaults.items():
        if not re.search(r'(?<!\w)' + re.escape(eng_term) + r'(?!\w)', en_s, re.IGNORECASE):
            continue
        has_any = any(
            re.search(r'(?<!\w)' + re.escape(t) + r'(?!\w)', de_s, re.IGNORECASE)
            for t in de_terms
        )
        if not has_any:
            score += 5
            reasons.append(f"glossary:{eng_term}")
            break

    # +5 name changed: name in EN missing from DE
    for name in names:
        name_lower = name.lower()
        en_has = all(p in en_s for p in name.split())
        de_has = all(p in de_s for p in name.split())
        if en_has and not de_has:
            score += 5
            reasons.append(f"name:{name}")
            break

    # +3 untranslated English
    if en_s and en_s == de_s and not en_s.startswith("["):
        score += 3
        reasons.append("untranslated")

    # +3 SFX missing: EN has [...], DE doesn't
    if re.search(r'\[([^\]]+)\]', en_s) and not re.search(r'\[([^\]]+)\]', de_s):
        score += 3
        reasons.append("sfx_missing")

    # +2 title mismatch
    for title in titles:
        if title.lower() in en_s.lower():
            if title.lower() not in de_s.lower():
                score += 2
                reasons.append(f"title:{title}")
                break

    # +2 song lyrics: NLLB often mishandles poetic/lyric lines
    if "♪" in en_s:
        score += 2
        reasons.append("song_lyrics")

    # +1 punctuation mismatch (end-of-line punctuation differs)
    en_punct = _END_PUNCT.search(en_s)
    de_punct = _END_PUNCT.search(de_s)
    if en_punct and de_punct and en_punct.group() != de_punct.group():
        score += 1
        reasons.append("punct_mismatch")

    # +3 song marker missing: EN has ♪ but DE doesn't
    if SONG_MARKER in en_s and SONG_MARKER not in de_s:
        score += 3
        reasons.append("song_marker_missing")

    # +4 length anomaly: DE >> EN on short EN (hallucination indicator)
    if len(en_s) < 15 and len(de_s) > len(en_s) * 2.5:
        score += 4
        reasons.append("length_anomaly")

    # +3 parenthetical content dropped: EN has (...) but DE doesn't
    if re.search(r'\([^)]+\)', en_s) and not re.search(r'\([^)]+\)', de_s):
        score += 3
        reasons.append("parenthetical_dropped")

    # +2 invented dash prefix: DE starts with "- " but EN doesn't
    if de_s.startswith("- ") and not en_s.startswith("- "):
        score += 2
        reasons.append("invented_dash")

    # +N English words remain in DE output (partial translation detection)
    # Flags lines where NLLB translated some words but left others in English.
    # Skips known loanwords, glossary DE terms, name parts, and short words.
    _ENG_LOANWORDS = {"okay", "ok", "hi", "bye", "sorry", "wow", "cool",
                      "yeah", "yep", "nope", "hey", "huh", "oops", "whoa"}
    gloss_de_words = set()
    for de_terms in glossary_defaults.values():
        for t in de_terms:
            if t:
                gloss_de_words.add(t.lower())
    name_parts = set()
    for name in names:
        for part in name.split():
            name_parts.add(part.lower())
    en_words = set(re.findall(r"[a-z]+", en_s.lower()))
    de_words = re.findall(r"[a-z]+", de_s.lower())
    eng_remain = []
    seen = set()
    for w in de_words:
        if len(w) < 3 or w in seen:
            continue
        seen.add(w)
        if w in _ENG_LOANWORDS or w in gloss_de_words or w in name_parts:
            continue
        if w in en_words:
            eng_remain.append(w)
            if len(eng_remain) >= 3:
                break
    if eng_remain:
        n = len(eng_remain)
        score += 6 if n >= 3 else 3
        reasons.append(f"eng_remain:{' '.join(eng_remain)}")

    return {"score": score, "reasons": reasons}


def _build_term_map(glossary: dict) -> dict[str, list[str]]:
    """Build {english_term: [german_options]} lookup."""
    m = {}
    for k, v in glossary.items():
        if isinstance(v, dict):
            m[k] = [v.get("default", "")] + v.get("acceptable", [])
        elif isinstance(v, str) and v:
            m[k] = [v]
    return m


def check_context_consistency(eng_texts: list[str], ger_texts: list[str],
                               glossary: dict, names: list[str],
                               window: int = 10) -> list[dict]:
    """Detect inconsistent translations within a sliding context window.
    
    Returns list of issues with score contributions.
    """
    issues: list[dict] = []
    n = len(eng_texts)
    term_map = _build_term_map(glossary)

    for i in range(n):
        en = eng_texts[i]
        de = ger_texts[i]

        for term, de_options in term_map.items():
            if not re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', en, re.IGNORECASE):
                continue
            # Which DE option is used in this line?
            used = None
            for opt in de_options:
                if opt and re.search(r'(?<!\w)' + re.escape(opt) + r'(?!\w)', de, re.IGNORECASE):
                    used = opt
                    break
            if used is None:
                continue  # no DE option found (handled by existing score_line)

            # Check nearby lines for same EN term with different DE
            start = max(0, i - window)
            end = min(n, i + window + 1)
            for j in range(start, end):
                if j == i:
                    continue
                if not re.search(r'(?<!\w)' + re.escape(term) + r'(?!\w)', eng_texts[j], re.IGNORECASE):
                    continue
                other_de = ger_texts[j]
                other_used = None
                for opt in de_options:
                    if opt and re.search(r'(?<!\w)' + re.escape(opt) + r'(?!\w)', other_de, re.IGNORECASE):
                        other_used = opt
                        break
                if other_used is not None and other_used != used:
                    issues.append({
                        "line": i + 1,
                        "type": "inconsistent_glossary",
                        "term": term,
                        "used_in_line": used,
                        "used_nearby": other_used,
                        "nearby_line": j + 1,
                        "score": 2,
                    })

    return issues


def generate_qa_report(eng_texts: list[str], ger_texts: list[str],
                       glossary: dict, names: list[str], titles: list[str],
                       output_path: str | None = None):
    """Generate qa_report.json with per-line scores.
    If output_path is None, skip file writing and return report dict only.
    """
    report = {
        "total_lines": len(ger_texts),
        "total_score": 0,
        "lines": [],
    }
    for i, (en, de) in enumerate(zip(eng_texts, ger_texts)):
        result = score_line(en, de, glossary, names, titles)
        report["lines"].append({
            "line": i + 1,
            "score": result["score"],
            "reasons": result["reasons"],
        })
        report["total_score"] += result["score"]

    # Context consistency check (additive)
    ctx_issues = check_context_consistency(eng_texts, ger_texts, glossary, names)
    if ctx_issues:
        report["context_consistency_issues"] = ctx_issues
        for issue in ctx_issues:
            report["total_score"] += issue["score"]
            line_idx = issue["line"] - 1
            if 0 <= line_idx < len(report["lines"]):
                report["lines"][line_idx]["score"] += issue["score"]
                reason = (
                    f"inconsistent:{issue['term']}:"
                    f"{issue['used_in_line']}≠{issue['used_nearby']}(L{issue['nearby_line']})"
                )
                report["lines"][line_idx]["reasons"].append(reason)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"  QA report: {output_path}  (total score: {report['total_score']})", flush=True)
    return report

# ---------------------------------------------------------------------------
# Resume/checkpoint support
# ---------------------------------------------------------------------------

def _checkpoint_path(out: Path) -> Path:
    return out.with_name(out.stem + ".resume.json")

def _save_checkpoint(fpath: Path, out: Path, completed_batches: list[int],
                     batch_size: int, total_lines: int, elapsed: float,
                     all_trans: dict | None = None):
    """Save translation progress checkpoint with optional translated text."""
    ckp: dict = {
        "input": str(fpath),
        "output": str(out),
        "version": __version__,
        "completed_batches": completed_batches,
        "batch_size": batch_size,
        "total_lines": total_lines,
        "translated_count": len(completed_batches) * batch_size,
        "elapsed_seconds": round(elapsed, 1),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if all_trans is not None:
        ckp["translated"] = {str(k): v for k, v in all_trans.items()}
    ckp_path = _checkpoint_path(out)
    tmp = ckp_path.with_suffix(".resume.tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckp, f, ensure_ascii=False)
    os.replace(str(tmp), str(ckp_path))

def _load_checkpoint(fpath: Path, out: Path, cfg: Config
                     ) -> tuple[list[int], dict[int, str], float] | None:
    """Load checkpoint if valid. Returns (completed_batches, all_trans, elapsed)."""
    ckp_path = _checkpoint_path(out)
    if not ckp_path.exists():
        return None
    try:
        with open(ckp_path, encoding="utf-8") as f:
            ckp = json.load(f)
        if (ckp.get("input") == str(fpath)
                and ckp.get("batch_size") == cfg.batch_size
                and ckp.get("version") == __version__):
            n = ckp.get("translated_count", 0)
            print(f"  [RESUME] checkpoint: {n}/{ckp['total_lines']} lines", flush=True)
            batches = ckp["completed_batches"]
            raw = ckp.get("translated", {})
            trans = {int(k): v for k, v in raw.items()}
            return (batches, trans, ckp.get("elapsed_seconds", 0))
    except (json.JSONDecodeError, KeyError, OSError):
        pass
    return None

def _remove_checkpoint(out: Path):
    """Remove checkpoint file after successful completion."""
    ckp_path = _checkpoint_path(out)
    try:
        ckp_path.unlink(missing_ok=True)
    except OSError:
        pass

# ---------------------------------------------------------------------------
# 6a. Translation Memory (SQLite, exact-match only)
# ---------------------------------------------------------------------------

import sqlite3

_TM_CONN: sqlite3.Connection | None = None

def _tm_db_path() -> str:
    if getattr(sys, 'frozen', False):
        base = Path(os.environ.get('LOCALAPPDATA', Path.home())) / "SubtitleTranslator"
    else:
        base = CONFIG_DIR.parent
    tm_dir = base / "tm"
    tm_dir.mkdir(parents=True, exist_ok=True)
    return str(tm_dir / "translation_memory.db")

def _get_tm() -> sqlite3.Connection:
    global _TM_CONN
    if _TM_CONN is None:
        _TM_CONN = sqlite3.connect(_tm_db_path(), check_same_thread=False)
        _TM_CONN.execute(
            "CREATE TABLE IF NOT EXISTS tm ("
            "normalized_en TEXT PRIMARY KEY,"
            "approved_de TEXT NOT NULL,"
            "usage_count INTEGER DEFAULT 1,"
            "last_used TEXT DEFAULT (datetime('now'))"
            ")"
        )
        _TM_CONN.commit()
    return _TM_CONN

def _normalize_en(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r'[^\w\s\']', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def tm_lookup(en_text: str, commit: bool = True) -> str | None:
    conn = _get_tm()
    norm = _normalize_en(en_text)
    if not norm:
        return None
    cursor = conn.execute("SELECT approved_de FROM tm WHERE normalized_en = ?", (norm,))
    row = cursor.fetchone()
    if row:
        conn.execute(
            "UPDATE tm SET usage_count = usage_count + 1, last_used = datetime('now') "
            "WHERE normalized_en = ?",
            (norm,),
        )
        if commit:
            conn.commit()
        return row[0]
    return None


def tm_store(en_text: str, de_text: str, commit: bool = True):
    norm = _normalize_en(en_text)
    if not norm or not de_text:
        return
    conn = _get_tm()
    conn.execute(
        "INSERT OR IGNORE INTO tm (normalized_en, approved_de) VALUES (?, ?)",
        (norm, de_text),
    )
    if commit:
        conn.commit()

def tm_stats() -> dict:
    conn = _get_tm()
    total = conn.execute("SELECT COUNT(*) FROM tm").fetchone()[0]
    most_used = conn.execute(
        "SELECT normalized_en, approved_de, usage_count FROM tm ORDER BY usage_count DESC LIMIT 10"
    ).fetchall()
    return {"total_entries": total, "most_used": most_used}

# ---------------------------------------------------------------------------
# NLLB fast translation
# ---------------------------------------------------------------------------

_MODEL = None
_TOKENIZER = None

def load_nllb(cfg: Config):
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _TOKENIZER, _MODEL
    print("  Loading NLLB-600M model...", end=" ", flush=True)
    t0 = time.time()
    _TOKENIZER = AutoTokenizer.from_pretrained(
        "facebook/nllb-200-distilled-600M", src_lang=cfg.src_lang
    )
    _MODEL = AutoModelForSeq2SeqLM.from_pretrained(
        "facebook/nllb-200-distilled-600M"
    ).to(cfg.device)
    _MODEL.eval()
    print(f"  Done ({time.time()-t0:.1f}s)", flush=True)
    return _TOKENIZER, _MODEL


# ---------------------------------------------------------------------------
# 6b. Segment-based placeholder protection (never sends ZZZ through NLLB)
# ---------------------------------------------------------------------------

_SEGMENT_RE = re.compile(r'(ZZZ\w+ZZZ)')
_LITERAL_RE = re.compile(r'^[^\w]*$')  # purely non-alphanumeric


def _segment_line(text: str, lookup: dict[str, str] | None = None
                  ) -> tuple[list[str], list]:
    """Split protected text into translatable content segments + layout.

    The layout preserves whitespace and marker positions. Only non-whitespace
    content is sent to NLLB; whitespace, markers, and punctuation-only
    fragments are reinserted literally.

    Returns:
        content: list of content strings to translate (indexed by layout)
        layout: list of ('content', idx) / ('literal', text) / ('space', text) / ('marker', text)
    """
    parts = _SEGMENT_RE.split(text)
    content = []
    layout = []

    def add_text_part(part: str):
        before = re.match(r'^\s*', part).group()
        after = re.search(r'\s*$', part).group()
        mid = part.strip()
        if mid:
            if _LITERAL_RE.match(mid) and len(mid) < 5:
                if before:
                    layout.append(('space', before))
                layout.append(('literal', mid))
                if after:
                    layout.append(('space', after))
            else:
                # Strip leading non-word characters (punctuation) from content
                # so NLLB never sees structural punctuation (comma after name, etc.)
                punc_match = re.match(r'^([^\w\s]+)(\s*)(.*)', mid, re.DOTALL)
                if punc_match:
                    punct, spaces, rest = punc_match.groups()
                    if rest and len(punct) < 5:
                        if before:
                            layout.append(('space', before))
                        layout.append(('literal', punct))
                        if spaces:
                            layout.append(('space', spaces))
                        layout.append(('content', len(content)))
                        content.append(rest)
                        if after:
                            layout.append(('space', after))
                        return
                if before:
                    layout.append(('space', before))
                layout.append(('content', len(content)))
                content.append(mid)
                if after:
                    layout.append(('space', after))
        else:
            layout.append(('space', part))

    for i, part in enumerate(parts):
        if i % 2 == 0:
            if not part:
                continue
            for line_part in re.split(r'(\n)', part):
                if line_part:
                    add_text_part(line_part)
        else:
            original = lookup.get(part, part) if lookup else part
            layout.append(('marker', original))
    return content, layout


def _segment_texts(protected_texts: list[str],
                   *maps: dict[str, str]) -> tuple[list[str], list[list]]:
    """Segment all protected texts into flat content list and per-line layouts."""
    lookup = {}
    for m in maps:
        lookup.update(m)
    all_content = []
    line_layouts = []
    for text in protected_texts:
        c, lay = _segment_line(text, lookup)
        all_content.extend(c)
        line_layouts.append(lay)
    return all_content, line_layouts


def _reconstruct_line(layout: list, translated: list[str]) -> str:
    """Reconstruct one line from layout and translated content segments."""
    parts = []
    seg_ptr = 0
    for typ, val in layout:
        if typ == 'content':
            parts.append(translated[seg_ptr])
            seg_ptr += 1
        elif typ in ('space', 'literal'):
            parts.append(val)
        elif typ == 'marker':
            parts.append(val)
    return ''.join(parts)


@torch.inference_mode()
def translate_fast(fpath: Path, cfg: Config,
                   progress_callback: callable = None,
                   output_path: Path | None = None) -> bool:
    """Translate an SRT file. Calls progress_callback(done, total) after each batch."""
    timer = _Timer()
    out = output_path or output_path_for(fpath)
    try:
        subs = safe_open_srt(fpath)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

    eng_texts = [sub.text for sub in subs]
    n = len(eng_texts)
    print(f"  {fpath.name}  {n} lines", flush=True)
    if progress_callback:
        progress_callback(0, n)

    # Load glossary & auto-glossary (learned from Ollama fixes)
    glossary = load_glossary()
    short_fragments = load_short_fragments()
    auto_glossary = _load_auto_glossary()
    if auto_glossary:
        # Merge auto-glossary into main glossary (lowercase keys for matching)
        merged = dict(glossary)
        for k, v in auto_glossary.items():
            if k not in merged:
                merged[k] = v
        glossary = merged
    names = load_names()

    # Protection layers (order: SFX → short excl → song markers → episode markers → names → numbers → multispeaker)
    timer.start("Protect")
    protected, sfx_map = protect_sfx(eng_texts)
    protected, excl_map = protect_short_exclamations(protected)
    protected, song_map = protect_song_markers(protected)
    protected, ep_map = protect_episode_markers(protected)
    protected, name_map = protect_names(protected, names)
    protected, num_map = protect_numbers(protected)
    protected, multi_map = protect_multispeaker(protected)
    protected, short_map = protect_short_fragments(protected, glossary, short_fragments)
    timer.stop("Protect")

    # Segment protected texts: extract content (no ZZZ markers) for NLLB
    timer.start("Segment")
    all_content, line_layouts = _segment_texts(protected, sfx_map, excl_map, song_map, ep_map, name_map, num_map, multi_map, short_map)
    _cum_content = [0]
    for lay in line_layouts:
        _cum_content.append(_cum_content[-1] + sum(1 for t, _ in lay if t == 'content'))
    timer.stop("Segment")

    # Check for resume checkpoint
    completed_batches: list[int] = []
    all_trans: dict[int, str] = {}
    t_start = time.time()
    batch_size = cfg.batch_size

    if cfg.resume:
        checkpoint = _load_checkpoint(fpath, out, cfg)
        if checkpoint:
            completed_batches, all_trans, prev_elapsed = checkpoint
            t_start = time.time()
            print(f"  [RESUME] {len(completed_batches)} batches done, {n - len(all_trans)} remaining",
                  flush=True)

    timer.start("NLLB Load")
    tok, model = load_nllb(cfg)
    timer.stop("NLLB Load")
    forced_bos = tok.convert_tokens_to_ids(cfg.tgt_lang)

    try:
        timer.start("NLLB Translate")
        for s in range(0, n, batch_size):
            batch_idx = s // batch_size
            if batch_idx in completed_batches:
                done = min(s + batch_size, n)
                elapsed = time.time() - t_start
                rate = done / elapsed if elapsed > 0 else 0
                print(f"  [{done}/{n}  {elapsed:.0f}s  {rate:.0f} l/s  (restored from checkpoint)]",
                       flush=True)
                if progress_callback:
                    progress_callback(done, n)
                continue

            # Translate only text content (no ZZZ placeholders reach NLLB)
            batch_end = min(s + batch_size, n)
            c_start = _cum_content[s]
            c_end = _cum_content[batch_end]
            batch_content = all_content[c_start:c_end]

            if batch_content:
                inputs = tok(batch_content, src_lang=cfg.src_lang, return_tensors="pt",
                             padding=True, truncation=True).to(cfg.device)

                outputs = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos,
                    max_new_tokens=96,
                    num_beams=cfg.num_beams,
                    no_repeat_ngram_size=cfg.no_repeat_ngram,
                )
                translated = tok.batch_decode(outputs, skip_special_tokens=True)
            else:
                translated = []

            # Reconstruct each line from its translated content segments
            seg_ptr = 0
            for li in range(s, batch_end):
                n_c = _cum_content[li + 1] - _cum_content[li]
                line_segs = translated[seg_ptr:seg_ptr + n_c] if n_c else []
                seg_ptr += n_c
                all_trans[li + 1] = _reconstruct_line(line_layouts[li], line_segs)

            completed_batches.append(batch_idx)

            elapsed = time.time() - t_start
            done = min(s + batch_size, n)
            rate = done / elapsed if elapsed > 0 else 0
            eta = (n - done) / rate if rate > 0 else 0
            print(f"  [{done}/{n}  {elapsed:.0f}s  {rate:.0f} l/s  ETA {eta:.0f}s]", flush=True)
            if progress_callback:
                progress_callback(done, n)

            _save_checkpoint(fpath, out, completed_batches, batch_size, n, elapsed, all_trans)

    except KeyboardInterrupt:
        _save_checkpoint(fpath, out, completed_batches, batch_size, n,
                         time.time() - t_start, all_trans)
        print(f"\n  [INTERRUPT] Checkpoint saved. Use --resume to continue.", flush=True)
        return False
    except Exception:
        _save_checkpoint(fpath, out, completed_batches, batch_size, n,
                         time.time() - t_start, all_trans)
        print(f"\n  [CRASH] Checkpoint saved. Use --resume to continue.", flush=True)
        raise

    timer.stop("NLLB Translate")

    # Reconstruct ger_texts then canonically restore any ZZZ placeholders
    timer.start("Restore")
    ger_texts = [all_trans.get(i + 1, "") for i in range(n)]
    ger_texts = _canonical_restore(
        ger_texts, num_map, name_map, ep_map, song_map, sfx_map, excl_map, multi_map, short_map,
    )

    # Post-hoc song marker restoration (handles markers in EN that aren't protected placeholders)
    smc = apply_song_markers(ger_texts, eng_texts)
    if smc:
        print(f"  song markers: {smc} restored", flush=True)

    # Validate no placeholders leaked into output
    timer.start("Validation")
    leaked = validate_no_placeholders(ger_texts, label=out.name)
    timer.stop("Validation")
    timer.stop("Restore")

    # Translation Memory lookup (exact match; glossary overrides below)
    # NOTE: --force skips TM reads to avoid stale cached data poisoning re-runs
    timer.start("TM")
    tm_hits = 0
    if cfg.use_tm and not cfg.force:
        _get_tm().execute("BEGIN")
        for i in range(n):
            cached = tm_lookup(eng_texts[i], commit=False)
            if cached is not None:
                ger_texts[i] = cached
                tm_hits += 1
        _get_tm().commit()
    timer.stop("TM")

    # Apply glossary corrections
    timer.start("Glossary")
    gc = apply_glossary(eng_texts, ger_texts, glossary)
    timer.stop("Glossary")

    # Preserve names
    timer.start("Names")
    nc = preserve_names(eng_texts, ger_texts, names, glossary)
    timer.stop("Names")

    # Apply German phrase fixes
    timer.start("Fixes")
    ger_fixes = load_german_fixes()
    pfc = apply_german_fixes(ger_texts, ger_fixes)
    timer.stop("Fixes")

    # Preserve titles & episode markers
    timer.start("Titles")
    titles = load_titles()
    tc = preserve_titles(eng_texts, ger_texts, titles)
    timer.stop("Titles")

    # Cleanup subtitle typography
    timer.start("Cleanup")
    cc = cleanup_subtitles(ger_texts)
    timer.stop("Cleanup")

    timer.start("Short Exclamations")
    apply_short_exclamation_overrides(eng_texts, ger_texts)
    timer.stop("Short Exclamations")

    # Feed conversation memory (rolling context for Ollama polish)
    timer.start("Conversation Memory")
    glossary_keys = list(glossary.keys())
    for en, de in zip(eng_texts, ger_texts):
        ConversationMemory.feed(en, de, glossary_keys, names)
    timer.stop("Conversation Memory")

    # Generate QA report (in-memory only — used for TM Store filtering)
    timer.start("QA")
    qa_report = generate_qa_report(eng_texts, ger_texts, glossary, names, titles, None)
    timer.stop("QA")

    # Store non-suspicious translations in Translation Memory
    # (restore first to avoid caching leaky ZZZ entries)
    timer.start("TM Store")
    stored = 0
    ger_texts = _canonical_restore(
        ger_texts, num_map, name_map, ep_map, song_map, sfx_map, excl_map, multi_map, short_map,
    )
    if qa_report and cfg.use_tm:
        _get_tm().execute("BEGIN")
        for i in range(n):
            line_info = qa_report["lines"][i]
            if line_info["score"] < SUSPICIOUS_THRESHOLD and ger_texts[i]:
                tm_store(eng_texts[i], ger_texts[i], commit=False)
                stored += 1
        _get_tm().commit()
    timer.stop("TM Store")

    # Canonical restore — final safety pass (catches any ZZZ re-introduced by
    # TM lookup, glossary overrides, or other post-processing)
    ger_texts = _canonical_restore(
        ger_texts, num_map, name_map, ep_map, song_map, sfx_map, excl_map, multi_map, short_map,
    )
    for i in range(n):
        m = re.match(r'^\s*\[([^\]]+)\]\s*$', eng_texts[i])
        if m and re.search(r'[.?!\n]', m.group(1)):
            if not re.match(r'^\s*\[', ger_texts[i]):
                ger_texts[i] = '[' + ger_texts[i].strip() + ']'

    apply_short_exclamation_overrides(eng_texts, ger_texts)

    # Write output (atomic) — assign AFTER final restore
    for i, sub in enumerate(subs):
        sub.text = ger_texts[i]

    # Hard fail on placeholder leak — never ship a ZZZ…ZZZ

    # Hard fail on placeholder leak — never ship a ZZZ…ZZZ
    for i, t in enumerate(ger_texts):
        if _PLACEHOLDER_RE.search(t):
            raise RuntimeError(
                f"Placeholder leak in {out.name} line {i+1}: "
                f"{_PLACEHOLDER_RE.search(t).group()} — aborting save"
            )

    try:
        timer.start("Write")
        atomic_save(subs, out)
        timer.stop("Write")
        _remove_checkpoint(out)
        elapsed = time.time() - t_start
        print(f"  DONE: {out.name}  ({elapsed:.0f}s  {n/elapsed:.1f} l/s)", flush=True)
        # Re-read English SRT for validation (subs was mutated above)
        eng_subs = safe_open_srt(fpath)
        timer.start("Validation")
        validate(eng_subs, safe_open_srt(out), out.name)
        timer.stop("Validation")
        print(f"")
        print(f"  Pipeline Timing")
        print(timer.report())
        print(flush=True)
        return True
    except Exception as e:
        print(f"  [ERROR] Write: {e}")
        return False

# ---------------------------------------------------------------------------
# 5a. Lightweight conversation memory (in-memory rolling context)
# ---------------------------------------------------------------------------

class ConversationMemory:
    """Rolling in-memory context of active characters and terms.
    
    Tracks which glossary terms and character names appear in recent
    subtitles. Used during Ollama polish to provide scene context.
    Discarded automatically as subtitles advance (max 20 entries).
    Thread-safe via class-level lock.
    """
    _lines: list[dict] = []
    _MAX = 20
    _lock: threading.Lock = threading.Lock()

    @classmethod
    def feed(cls, en_text: str, de_text: str,
             glossary_keys: list[str], known_names: list[str]):
        entry = {"en": en_text, "de": de_text}
        entry["active_terms"] = [
            k for k in glossary_keys
            if re.search(r'(?<!\w)' + re.escape(k) + r'(?!\w)', en_text, re.IGNORECASE)
        ]
        entry["active_names"] = [
            n for n in known_names
            if all(p.lower() in en_text.lower() for p in n.split())
        ]
        with cls._lock:
            cls._lines.append(entry)
            if len(cls._lines) > cls._MAX:
                cls._lines.pop(0)

    @classmethod
    def get_context(cls) -> dict:
        names: set[str] = set()
        terms: set[str] = set()
        with cls._lock:
            lines_copy = list(cls._lines)
        for e in lines_copy:
            names.update(e["active_names"])
            terms.update(e["active_terms"])
        return {"names": sorted(names), "terms": sorted(terms)}

    @classmethod
    def get_context_text(cls) -> str:
        ctx = cls.get_context()
        parts = []
        if ctx["names"]:
            parts.append("Active characters: " + ", ".join(ctx["names"]))
        if ctx["terms"]:
            parts.append("Active terms: " + ", ".join(ctx["terms"]))
        return "\n".join(parts) if parts else ""

    @classmethod
    def reset(cls):
        with cls._lock:
            cls._lines.clear()

    @classmethod
    def build_from_file(cls, eng_texts: list[str], ger_texts: list[str]):
        cls.reset()
        glossary_keys = list(load_glossary().keys())
        known_names = load_names()
        for en, de in zip(eng_texts, ger_texts):
            cls.feed(en, de, glossary_keys, known_names)

# ---------------------------------------------------------------------------
# 5b+. LLM-only base translation (TowerInstruct, batched with context)
# ---------------------------------------------------------------------------

LLM_BATCH_SIZE = 30
LLM_CONTEXT_WINDOW = 2
LLM_LINE_RE = re.compile(r"^(\d+)\s*:\s*(.+)", re.MULTILINE | re.UNICODE)


def _make_glossary_text(glossary: dict) -> str:
    lines = []
    for k, v in glossary.items():
        if isinstance(v, dict):
            v = v.get("default", str(v))
        lines.append(f"  {k} → {v}")
    return "\n".join(lines)


def build_llm_prompt(batch_ids: list[int], eng_texts: list[str],
                     glossary: dict, names: list[str]) -> str:
    first, last = batch_ids[0], batch_ids[-1]
    ctx_start = max(0, first - LLM_CONTEXT_WINDOW)
    ctx_end = min(len(eng_texts), last + LLM_CONTEXT_WINDOW + 1)
    batch_set = set(batch_ids)

    parts = [
        "Translate each English subtitle to German.",
        "Output each line as: NUMBER: German text",
        "",
        "RULES:",
        "- Natural spoken German matching character register",
        "- Preserve [SFX] brackets — translate content inside",
        "- Multi-speaker lines with dashes: preserve the dash format",
        "- Keep song markers (♪), translate lyrics",
        '- Short solo exclamations like "Bravo!", "Ah!": keep as-is',
        '- "Yes, sir." → "Jawohl" (military) or "Ja, mein Herr" (polite)',
        "",
    ]
    if glossary:
        parts.append("Use these glossary terms where applicable:")
        parts.append(_make_glossary_text(glossary))
        parts.append("")
    if names:
        parts.append("Proper nouns (keep as-is): " + ", ".join(names))
        parts.append("")

    ctx_lines = []
    for i in range(ctx_start, ctx_end):
        if i not in batch_set:
            tag = "P" if i < first else "N"
            t = eng_texts[i].replace("\n", " // ")
            ctx_lines.append(f"{tag}{i+1}: {t}")
    if ctx_lines:
        parts.append("[CONTEXT]")
        parts.extend(ctx_lines)
        parts.append("")

    parts.append("[TRANSLATE]")
    for idx in batch_ids:
        parts.append(f"{idx+1}: {eng_texts[idx].replace(chr(10), ' // ')}")

    return "\n".join(parts)


def translate_llm(fpath: Path, cfg: Config,
                  output_path: Path | None = None) -> bool:
    timer = _Timer()
    out = output_path or output_path_for(fpath)
    try:
        subs = safe_open_srt(fpath)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False

    eng_texts = [sub.text for sub in subs]
    n = len(eng_texts)
    print(f"  {fpath.name}  {n} lines", flush=True)

    glossary = load_glossary()
    names = load_names()

    session = requests.Session()
    chat_url = f"{cfg.ollama_host}/api/chat"

    timer.start("LLM Warmup")
    print(f"  Loading {cfg.llm_model}...", end=" ", flush=True)
    warmup = {
        "model": cfg.llm_model,
        "messages": [{"role": "user", "content": "Translate: Hello → German."}],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 10, "keep_alive": "30m"},
    }
    try:
        r = session.post(chat_url, json=warmup, timeout=120)
        r.raise_for_status()
    except Exception as e:
        print(f"FAILED: {e}")
        return False
    timer.stop("LLM Warmup")

    batch_size = cfg.llm_batch_size
    translations: dict[int, str] = {}
    batch_starts = list(range(0, n, batch_size))
    total = len(batch_starts)
    t_start = time.time()

    timer.start("LLM Translate")
    for bi, s in enumerate(batch_starts):
        ids = list(range(s, min(s + batch_size, n)))
        prompt = build_llm_prompt(ids, eng_texts, glossary, names)
        payload = {
            "model": cfg.llm_model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_ctx": 4096,
                        "num_predict": len(ids) * 120, "keep_alive": "30m"},
        }
        found = 0
        expected = len(ids)
        for attempt in range(2):
            try:
                resp = session.post(chat_url, json=payload, timeout=180)
                resp.raise_for_status()
                content = resp.json().get("message", {}).get("content", "")
                for m in LLM_LINE_RE.finditer(content):
                    num = int(m.group(1))
                    if num in {i + 1 for i in ids}:
                        text = m.group(2).strip()
                        if text:
                            translations[num] = text
                            found += 1
                break
            except Exception as e:
                if attempt == 0:
                    print(f"\n    [RETRY] batch {bi+1}: {e}", flush=True)
                    time.sleep(5)
                else:
                    print(f"\n    [FAIL] batch {bi+1}: {e}", flush=True)

        elapsed = time.time() - t_start
        done = min(s + batch_size, n)
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else 0
        print(f"  [{bi+1:2d}/{total}] {s+1:4d}-{ids[-1]+1:4d}  {found:2d}/{expected}  "
              f"[{done}/{n}  {elapsed:.0f}s  {rate:.0f} l/s  ETA {eta:.0f}s]",
              flush=True)

    timer.stop("LLM Translate")

    # Reconstruct ger_texts from translations
    timer.start("Reconstruct")
    ger_texts = [translations.get(i + 1, "") for i in range(n)]
    missing = n - len([t for t in ger_texts if t])
    if missing:
        print(f"  [WARN] {missing} lines untranslated", flush=True)
    timer.stop("Reconstruct")

    # Apply same post-processing as translate_fast
    timer.start("Glossary")
    gc = apply_glossary(eng_texts, ger_texts, glossary)
    timer.stop("Glossary")

    timer.start("Names")
    nc = preserve_names(eng_texts, ger_texts, names, glossary)
    timer.stop("Names")

    timer.start("Fixes")
    ger_fixes = load_german_fixes()
    pfc = apply_german_fixes(ger_texts, ger_fixes)
    timer.stop("Fixes")

    timer.start("Titles")
    titles = load_titles()
    tc = preserve_titles(eng_texts, ger_texts, titles)
    timer.stop("Titles")

    timer.start("Cleanup")
    cc = cleanup_subtitles(ger_texts)
    timer.stop("Cleanup")

    timer.start("Short Exclamations")
    apply_short_exclamation_overrides(eng_texts, ger_texts)
    timer.stop("Short Exclamations")

    timer.start("QA")
    qa_report = generate_qa_report(eng_texts, ger_texts, glossary, names, titles, None)
    timer.stop("QA")

    timer.start("TM Store")
    stored = 0
    if cfg.use_tm:
        _get_tm().execute("BEGIN")
        for i in range(n):
            line_info = qa_report["lines"][i]
            if line_info["score"] < SUSPICIOUS_THRESHOLD and ger_texts[i]:
                tm_store(eng_texts[i], ger_texts[i], commit=False)
                stored += 1
        _get_tm().commit()
    timer.stop("TM Store")

    # Write output
    for i, sub in enumerate(subs):
        sub.text = ger_texts[i]

    try:
        timer.start("Write")
        atomic_save(subs, out)
        timer.stop("Write")
        elapsed = time.time() - t_start
        print(f"  DONE: {out.name}  ({elapsed:.0f}s  {n/elapsed:.1f} l/s)", flush=True)
        eng_subs = safe_open_srt(fpath)
        timer.start("Validation")
        validate(eng_subs, safe_open_srt(out), out.name)
        timer.stop("Validation")
        print(f"")
        print(f"  Pipeline Timing")
        print(timer.report())
        print(flush=True)
        return True
    except Exception as e:
        print(f"  [ERROR] Write: {e}")
        return False

POLISH_PATTERNS = re.compile(
    r"\b(General|Colonel|Major|Captain|Lieutenant|Commander|Admiral|Sir|"
    r"Young Master|Elder Brother|Elder Sister|Young Lady|"
    r"Sect Leader|Sect|Master|Disciple|Cultivation|Immortal|Demon)\b",
    re.IGNORECASE,
)

def _ollama_chat(session, url: str, payload: dict, timeout: int = 120, headers: dict | None = None) -> dict | None:
    """Call LLM API with retry logic. Returns parsed JSON or None."""
    for attempt in range(2):
        try:
            resp = session.post(url, json=payload, headers=headers or {}, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except (requests.ConnectionError, requests.Timeout,
                requests.HTTPError) as e:
            if attempt == 0:
                print(f"  [RETRY] {e}", flush=True)
                time.sleep(5)
            else:
                print(f"  [ERROR] LLM call failed: {e}", flush=True)
                return None
    return None


def load_polisher(cfg: Config, model_override: str | None = None):
    session = requests.Session()
    model_name = model_override or cfg.ollama_model

    if cfg.proxy_base_url and model_name.startswith("deepseek"):
        chat_url = cfg.proxy_base_url.rstrip("/") + "/v1/chat/completions"
        model = model_name
        api_key = cfg.proxy_api_key
        print(f"  Loading proxy polisher ({model})...", end=" ", flush=True)
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "test"}],
            "temperature": 0,
            "max_tokens": 4,
        }
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        try:
            resp = session.post(chat_url, json=payload, headers=headers, timeout=30)
            resp.raise_for_status()
            print("OK", flush=True)
            return session, chat_url, model, api_key
        except Exception as e:
            print(f"FAILED: {e}", flush=True)
            return None

    chat_url = f"{cfg.ollama_host}/api/chat"
    model = model_name
    print(f"  Loading Ollama polisher ({model})...", end=" ", flush=True)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": '<LINE id="1">Hello.</LINE>'}],
        "stream": False,
        "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 16, "keep_alive": "30m"},
    }
    result = _ollama_chat(session, chat_url, payload, timeout=120)
    if result is None:
        print("FAILED", flush=True)
        return None
    print("OK", flush=True)
    return session, chat_url, model, None

SUSPICIOUS_THRESHOLD = 4


def find_suspicious_lines(eng_texts: list[str], ger_texts: list[str],
                          glossary: dict, names: list[str],
                          titles: list[str] | None = None) -> list[int]:
    """Find lines needing Ollama polish using QA scoring.
    Only flags lines above SUSPICIOUS_THRESHOLD for precise targeting.
    """
    if titles is None:
        titles = load_titles()

    suspicious = set()
    for i, (en, de) in enumerate(zip(eng_texts, ger_texts)):
        if _is_all_short_exclamations(en):
            continue
        result = score_line(en, de, glossary, names, titles)
        if result["score"] >= SUSPICIOUS_THRESHOLD:
            suspicious.add(i)

    return sorted(suspicious)


def _rejects_content_addition(en_text: str, nllb_text: str, corr_text: str) -> bool:
    """Reject Ollama correction if it adds content absent from English source.

    Detects two patterns of invented content:
    1. Parentheticals added where English source has none
       (e.g. "Mom!" → "- Mutter! (Ich dachte, du wärst tot.)")
    2. Short English phrase expanded into a completely different sentence
       (e.g. "118." → "- Das ist der Preis.")
    
    Only rejects when both EN is short (≤20 chars) and the correction
    adds substantial new content beyond what NLLB produced.
    """
    en_stripped = en_text.strip()

    # Bracketed source content must not disappear during polish, regardless of length.
    if re.search(r'\[[^\]]+\]', en_stripped) and not re.search(r'\[[^\]]+\]', corr_text):
        return True

    if len(en_stripped) > 20:
        return False

    # Pattern 1: parentheses in correction but not in EN source
    if "(" in corr_text and "(" not in en_stripped:
        return True

    # Pattern 2: word count explosion vs NLLB when EN is short
    en_words = len(en_stripped.split())
    nllb_words = len(nllb_text.strip().split())
    corr_words = len(corr_text.strip().split())
    if en_words <= 3 and corr_words >= nllb_words * 1.5 and corr_words >= en_words * 3:
        return True

    return False


def translate_polish(fpath: Path, cfg: Config,
                     nllb_path: Path | None = None,
                     polish_model: str | None = None) -> bool:
    timer = _Timer()
    out = nllb_path or output_path_for(fpath)
    if not out.exists():
        print(f"  [SKIP] No NLLB output found: {out.name}")
        return False

    timer.start("Load")
    eng = safe_open_srt(fpath)
    ger = safe_open_srt(out)
    eng_texts = [sub.text for sub in eng]
    ger_texts = [sub.text for sub in ger]
    n = len(ger_texts)

    glossary = load_glossary()
    auto_glossary = _load_auto_glossary()
    if auto_glossary:
        merged = dict(glossary)
        for k, v in auto_glossary.items():
            if k not in merged:
                merged[k] = v
        glossary = merged
    names = load_names()
    timer.stop("Load")

    # Apply glossary & names first (fast, no LLM)
    timer.start("Pre-Polish")
    gc = apply_glossary(eng_texts, ger_texts, glossary)
    nc = preserve_names(eng_texts, ger_texts, names, glossary)
    apply_short_exclamation_overrides(eng_texts, ger_texts)

    # Build conversation memory from file for context-aware polishing
    ConversationMemory.build_from_file(eng_texts, ger_texts)
    timer.stop("Pre-Polish")

    timer.start("QA")
    suspicious = find_suspicious_lines(eng_texts, ger_texts, glossary, names)
    timer.stop("QA")
    if not suspicious:
        for i, sub in enumerate(ger):
            sub.text = ger_texts[i]
        for i, t in enumerate(ger_texts):
            if _PLACEHOLDER_RE.search(t):
                raise RuntimeError(
                    f"Placeholder leak in {out.name} line {i+1}: "
                    f"{_PLACEHOLDER_RE.search(t).group()} — aborting save"
                )
        atomic_save(ger, out)
        print(f"  {fpath.name}  no suspicious lines, {gc} glossary + {nc} name fixes")
        return True

    print(f"  {fpath.name}  {n} lines, {len(suspicious)} suspicious", flush=True)

    timer.start("Ollama Start")
    polisher = load_polisher(cfg, polish_model)
    timer.stop("Ollama Start")
    if polisher is None:
        print(f"  [WARN] Polisher unavailable, skipping polish for {fpath.name}")
        for i, sub in enumerate(ger):
            sub.text = ger_texts[i]
        atomic_save(ger, out)
        print(f"  SKIPPED: {out.name}  ({gc} glossary + {nc} name fixes)")
        return False

    session, chat_url, polish_model_name, proxy_api_key = polisher
    t_start = time.time()

    batch_size = 10
    ollama_fixes = {}
    n_rejected = 0
    n_cache_hits = 0
    # Load persistent Ollama cache from disk
    _ollama_cache: dict[tuple[str, str], str] = {}
    if _OLLAMA_CACHE_PATH.exists():
        try:
            with open(_OLLAMA_CACHE_PATH, encoding="utf-8") as f:
                raw = json.load(f)
            for eng, de_map in raw.items():
                for de_text, corr in de_map.items():
                    _ollama_cache[(eng, de_text)] = corr
        except Exception as e:
            print(f"  [WARN] Failed to load Ollama cache: {e}", flush=True)
    timer.start("Ollama Batches")
    total_batches = (len(suspicious) + batch_size - 1) // batch_size
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    _cache_lock = threading.Lock()
    _fixes_lock = threading.Lock()

    def _submit_batch(s):
        """Send one batch to the LLM, return (fixes, rejected, cache_hits, all_cached) or (None, ...)."""
        batch_ids = suspicious[s : s + batch_size]
        local_fixes = []
        local_rejected = 0
        local_cache_hits = 0

        cached_results = {}
        uncached_ids = []
        for idx in batch_ids:
            with _cache_lock:
                key = (eng_texts[idx], ger_texts[idx])
                if key in _ollama_cache:
                    cached_results[idx] = _ollama_cache[key]
                    local_cache_hits += 1
                else:
                    uncached_ids.append(idx)

        if not uncached_ids:
            for idx, corr in cached_results.items():
                if corr != ger_texts[idx]:
                    if _rejects_content_addition(eng_texts[idx], ger_texts[idx], corr):
                        local_rejected += 1
                    else:
                        local_fixes.append((idx, corr))
            return local_fixes, local_rejected, local_cache_hits, True

        xml_input = build_contextual_xml(uncached_ids, eng_texts, ger_texts)
        mem_ctx = ConversationMemory.get_context_text()
        user_msg = (
            "Improve each German <de> translation inside <current> using context.\n"
            "Rules:\n"
            "- ONLY rewrite the <de> text within <current>, never <previous> or <next>\n"
            "- Use surrounding context to ensure consistent character voices\n"
            "- Preserve all [SFX] brackets exactly\n"
            "- Match EN speaker count: if EN has multiple speaker lines (//), DE must have same count, each prefixed with dash\n"
            "- Never merge two speakers into one line\n"
            "- Translate ALL remaining English words to natural German. NO English words allowed in output.\n"
            "- Produce natural spoken German, as if written by a native speaker\n"
            "- Keep subtitle length appropriate\n"
            + (f"\nContext from recent scenes:\n{mem_ctx}\n" if mem_ctx else "")
            + "\n" + xml_input
        )

        if proxy_api_key is not None:
            payload = {
                "model": polish_model_name,
                "messages": [{"role": "user", "content": user_msg}],
                "temperature": 0,
                "max_tokens": len(batch_ids) * 300,
            }
            headers = {"Authorization": f"Bearer {proxy_api_key}"}
            result = _ollama_chat(session, chat_url, payload, timeout=120, headers=headers)
        else:
            payload = {
                "model": polish_model_name,
                "messages": [{"role": "user", "content": user_msg}],
                "stream": False,
                "options": {"temperature": 0, "num_ctx": 8192,
                            "num_predict": len(batch_ids) * 300, "keep_alive": "30m"},
            }
            result = _ollama_chat(session, chat_url, payload, timeout=120)

        if result is None:
            return None, local_rejected, local_cache_hits, False

        if proxy_api_key is not None:
            content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        else:
            content = result.get("message", {}).get("content", "")
        parsed = parse_xml(content)

        for idx in uncached_ids:
            sid = idx + 1
            if sid in parsed and parsed[sid]:
                corr = parsed[sid]
                with _cache_lock:
                    _ollama_cache[(eng_texts[idx], ger_texts[idx])] = corr
                if corr != ger_texts[idx]:
                    if _rejects_content_addition(eng_texts[idx], ger_texts[idx], corr):
                        local_rejected += 1
                    else:
                        local_fixes.append((idx, corr))

        for idx, corr in cached_results.items():
            if corr != ger_texts[idx]:
                if _rejects_content_addition(eng_texts[idx], ger_texts[idx], corr):
                    local_rejected += 1
                else:
                    local_fixes.append((idx, corr))

        return local_fixes, local_rejected, local_cache_hits, False

    batch_starts = list(range(0, len(suspicious), batch_size))
    parallel = getattr(cfg, "polish_parallel", 2) if total_batches > 1 else 1
    with ThreadPoolExecutor(max_workers=parallel) as executor:
        futures = {executor.submit(_submit_batch, s): s for s in batch_starts}
        for future in as_completed(futures):
            s = futures[future]
            try:
                local_fixes, local_rejected, local_cache_hits, all_cached = future.result()
            except Exception as exc:
                print(f"  [WARN] Batch at line {s+1} failed: {exc}", flush=True)
                continue
            if local_fixes is None:
                print(f"  [WARN] Skipping batch at line {s+1}", flush=True)
                continue
            with _fixes_lock:
                for idx, corr in local_fixes:
                    ollama_fixes[idx] = corr
                n_rejected += local_rejected
                n_cache_hits += local_cache_hits
            done = min(s + batch_size, len(suspicious))
            elapsed = time.time() - t_start
            tag = "  (all cached)" if all_cached else ""
            print(f"  [{done}/{len(suspicious)}  {elapsed:.0f}s{tag}]", flush=True)

    timer.stop("Ollama Batches")
    for idx, new_text in ollama_fixes.items():
        ger_texts[idx] = new_text.replace(" // ", "\n")

    # Learn term corrections from Ollama fixes (auto-glossary)
    timer.start("Auto-Learn")
    auto_glossary = _load_auto_glossary()
    learned = _learn_from_fixes(eng_texts, ger_texts, ollama_fixes, auto_glossary)
    if learned:
        print(f"  auto-glossary: {learned} new term(s) learned", flush=True)
    timer.stop("Auto-Learn")

    # Re-apply glossary after Ollama may have modified lines
    timer.start("Re-glossary")
    _ = apply_glossary(eng_texts, ger_texts, glossary)
    timer.stop("Re-glossary")

    timer.start("Short Exclamations")
    apply_short_exclamation_overrides(eng_texts, ger_texts)
    timer.stop("Short Exclamations")

    # Re-apply German phrase fixes (Ollama may have reverted them)
    timer.start("German Fixes")
    gfc = apply_german_fixes(ger_texts, load_german_fixes())
    timer.stop("German Fixes")

    # Catch remaining English words that NLLB missed
    timer.start("English Filter")
    _eng_fixes = _filter_english_words(eng_texts, ger_texts)
    timer.stop("English Filter")

    # Save persistent Ollama cache to disk
    try:
        _OLLAMA_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        serialized: dict[str, dict[str, str]] = {}
        for (eng, de), corr in _ollama_cache.items():
            serialized.setdefault(eng, {})[de] = corr
        with open(_OLLAMA_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(serialized, f, ensure_ascii=False, indent=1)
    except Exception as e:
        print(f"  [WARN] Failed to save Ollama cache: {e}", flush=True)

    timer.start("Save")
    for i, sub in enumerate(ger):
        sub.text = ger_texts[i]

    # Hard fail on placeholder leak
    for i, t in enumerate(ger_texts):
        if _PLACEHOLDER_RE.search(t):
            raise RuntimeError(
                f"Placeholder leak in {out.name} line {i+1}: "
                f"{_PLACEHOLDER_RE.search(t).group()} — aborting save"
            )

    try:
        atomic_save(ger, out)
        timer.stop("Save")
        elapsed = time.time() - t_start

        # ---- Stats collection for summary ----
        timer.start("Summary")
        conn = _get_tm()
        tm_total = conn.execute("SELECT COUNT(*) FROM tm").fetchone()[0]
        if cfg.use_tm:
            tm_hits = sum(1 for en in eng_texts if tm_lookup(en) is not None)
            tm_misses = n - tm_hits
            tm_rate = (tm_hits / n * 100) if n > 0 else 0
        else:
            tm_hits = 0
            tm_misses = n
            tm_rate = 0

        # Final QA score after all polish + re-glossary
        titles = load_titles()
        final_qa = generate_qa_report(eng_texts, ger_texts, glossary, names,
                                      titles)
        final_score = sum(l["score"] for l in final_qa["lines"]) if final_qa else 0

        n_unchanged = len(suspicious) - len(ollama_fixes)
        n_corr = gc + nc + len(ollama_fixes)
        timer.stop("Summary")

        print(f"  POLISHED: {out.name}  ({elapsed:.0f}s  {n_corr} total fixes)", flush=True)
        print(f"")
        print(f"  Translation Summary")
        print(f"  {'─'*50}")
        print(f"  QA:")
        print(f"    Suspicious: {len(suspicious)}")
        print(f"  Ollama:")
        print(f"    Sent: {len(suspicious)}")
        print(f"    Modified: {len(ollama_fixes)}")
        print(f"    Rejected: {n_rejected}")
        print(f"    Unchanged: {n_unchanged - n_rejected}")
        if n_cache_hits:
            print(f"    Cache hits: {n_cache_hits}")
        print(f"    Time: {elapsed:.1f}s")
        print(f"  TM:")
        print(f"    Entries: {tm_total}")
        print(f"    Hits: {tm_hits}")
        print(f"    Misses: {tm_misses}")
        print(f"    Hit Rate: {tm_rate:.1f}%")
        print(f"  Final QA Score: {final_score}")
        print(f"  {'─'*50}")
        print(f"")
        print(f"  Polish Timing")
        print(timer.report())
        print(flush=True)
        return True
    except Exception as e:
        print(f"  [ERROR] Write: {e}")
        return False

# ---------------------------------------------------------------------------
# 6. Test mode
# ---------------------------------------------------------------------------

TEST_LINES = [
    (1, "General, prepare the troops."),
    (2, "You are my elder brother."),
    (3, "The young master has arrived."),
    (4, "Captain, the enemy is approaching."),
    (5, "[music]"),
    (6, "Lu Xixiao, where are you going?"),
    (7, "I said, don't ever let me see you again."),
    (8, "Sir, yes sir!"),
    (9, "This is Commander Wang."),
    (10, "Forgive me, sect leader."),
]

EXPECTED = [
    (1, "General"),
    (2, "älterer Bruder"),
    (3, "Junger Herr"),
    (4, "Hauptmann"),
    (5, "[music]"),
    (6, "Lu Xixiao"),
    (7, None),
    (8, "Sir"),
    (9, "Kommandant"),
    (10, "Sektenführer"),
]

def run_test(cfg: Config):
    """Translate smoke-test lines + 90 real lines; validate output."""
    files = find_srt_files(cfg.input_dir)
    if not files:
        print("No test files found.")
        return

    # Build 100 lines: 10 smoke-test + first 90 from file
    fpath = files[0]
    subs = safe_open_srt(fpath)
    file_texts = [sub.text for sub in subs[:90]]
    test_texts = [t for _, t in TEST_LINES]
    all_texts = test_texts + file_texts
    n = len(all_texts)
    print(f"TEST: {n} lines (10 smoke + 90 from {fpath.name})\n", flush=True)

    tok, model = load_nllb(cfg)
    forced_bos = tok.convert_tokens_to_ids(cfg.tgt_lang)
    glossary = load_glossary()
    short_fragments = load_short_fragments()
    names = load_names()

    protected, sfx_map = protect_sfx(all_texts)
    protected, excl_map = protect_short_exclamations(protected)
    protected, song_map = protect_song_markers(protected)
    protected, ep_map = protect_episode_markers(protected)
    protected, name_map = protect_names(protected, names)
    protected, num_map = protect_numbers(protected)
    protected, multi_map = protect_multispeaker(protected)
    protected, short_map = protect_short_fragments(protected, glossary, short_fragments)

    t0 = time.time()
    inputs = tok(protected, src_lang=cfg.src_lang, return_tensors="pt",
                 padding=True, truncation=True).to(cfg.device)
    outputs = model.generate(
        **inputs, forced_bos_token_id=forced_bos,
        max_new_tokens=96, num_beams=cfg.num_beams,
        no_repeat_ngram_size=cfg.no_repeat_ngram,
    )
    results = tok.batch_decode(outputs, skip_special_tokens=True)
    t1 = time.time()
    ger_texts = _canonical_restore(
        results, num_map, name_map, ep_map, song_map, sfx_map, excl_map, multi_map, short_map,
    )
    apply_song_markers(ger_texts, all_texts)
    apply_glossary(all_texts, ger_texts, glossary)
    preserve_names(all_texts, ger_texts, names, glossary)
    ger_fixes = load_german_fixes()
    apply_german_fixes(ger_texts, ger_fixes)
    titles = load_titles()
    preserve_titles(all_texts, ger_texts, titles)
    cleanup_subtitles(ger_texts)

    elapsed = t1 - t0
    print(f"NLLB: {n} lines in {elapsed:.1f}s ({n/elapsed:.0f} l/s)\n", flush=True)

    # 2. Print sample: first 10 (smoke test) + lines 91-100 (file)
    print("SMOKE TEST (first 10):")
    smoke_ok = 0
    for lid, en_text in TEST_LINES:
        idx = lid - 1
        de_text = ger_texts[idx]
        expected_word = None
        for elid, ew in EXPECTED:
            if elid == lid:
                expected_word = ew
                break
        if expected_word and expected_word.lower() in de_text.lower():
            print(f"  [PASS] line {lid}: '{expected_word}' in '{de_text[:60]}'", flush=True)
            smoke_ok += 1
        elif expected_word:
            print(f"  [FAIL] line {lid}: want '{expected_word}' got '{de_text[:60]}'", flush=True)
        else:
            print(f"  [....] line {lid}: {de_text[:60]}", flush=True)
    print(f"  SMOKE: {smoke_ok}/10 passed", flush=True)

    print("\nFILE SAMPLE (last 10):")
    for i in range(90, 100):
        idx = i
        print(f"  {i+1:3d}| EN: {all_texts[idx][:60]}", flush=True)
        print(f"     | DE: {ger_texts[idx][:60]}", flush=True)

    # 3. Validation
    issues = []
    for i in range(n):
        if not ger_texts[i].strip():
            issues.append(f"line {i+1}: empty")
    if issues:
        print(f"\nISSUES ({len(issues)}):")
        for iss in issues[:5]:
            print(f"  {iss}", flush=True)
    else:
        print(f"\nVALIDATION: no empty lines", flush=True)

    # 4. Write test_output.srt
    out = fpath.with_name("test_output.srt")
    out_subs = pysrt.SubRipFile()
    for i in range(n):
        item = pysrt.SubRipItem(index=i+1, start="00:00:00,000", end="00:00:01,000",
                                 text=ger_texts[i])
        out_subs.append(item)
    atomic_save(out_subs, out)
    print(f"\nOutput: {out.name} ({n} lines)", flush=True)

# ---------------------------------------------------------------------------
# Benchmark mode
# ---------------------------------------------------------------------------

def run_benchmark(cfg: Config):
    """Measure translation performance and output benchmark.json."""
    import gc, torch.cuda as cuda

    # Find a test file
    files = find_srt_files(cfg.input_dir)
    if not files:
        print("No SRT files found for benchmark.")
        return

    # Load model
    t0 = time.time()
    tok, model = load_nllb(cfg)
    model_load_ms = (time.time() - t0) * 1000

    # Check GPU memory
    gpu_mem_mb = 0
    if cfg.device == "cuda" and torch.cuda.is_available():
        gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)

    # Translate first 100 lines
    fpath = files[0]
    subs = safe_open_srt(fpath)
    test_texts = [sub.text for sub in subs[:100]]
    n = len(test_texts)

    protected, sfx_map = protect_sfx(test_texts)
    forced_bos = tok.convert_tokens_to_ids(cfg.tgt_lang)

    t0 = time.time()
    inputs = tok(protected, src_lang=cfg.src_lang, return_tensors="pt",
                 padding=True, truncation=True).to(cfg.device)
    outputs = model.generate(
        **inputs, forced_bos_token_id=forced_bos,
        max_new_tokens=96, num_beams=cfg.num_beams,
        no_repeat_ngram_size=cfg.no_repeat_ngram,
    )
    results = tok.batch_decode(outputs, skip_special_tokens=True)
    translate_ms = (time.time() - t0) * 1000

    # Ollama latency (optional)
    ollama_warm_ms = 0
    ollama_5_lines_ms = 0
    if cfg.ollama_host:
        try:
            t0 = time.time()
            polisher = load_polisher(cfg)
            if polisher:
                ollama_warm_ms = (time.time() - t0) * 1000
                session, chat_url, polish_model_name, _ = polisher
                t0 = time.time()
                payload = {
                    "model": polish_model_name,
                    "messages": [{"role": "user", "content": "Translate to German.\n<LINE id=\"1\">Hello.</LINE>"}],
                    "stream": False,
                    "options": {"temperature": 0, "num_ctx": 4096, "num_predict": 16},
                }
                session.post(chat_url, json=payload, timeout=30)
                ollama_5_lines_ms = (time.time() - t0) * 1000
        except Exception:
            pass

    # CPU memory (rough estimate via process)
    import psutil
    cpu_mem_mb = psutil.Process().memory_info().rss / (1024 * 1024)

    benchmark = {
        "version": __version__,
        "device": cfg.device,
        "cuda_available": torch.cuda.is_available(),
        "batch_size": cfg.batch_size,
        "nllb_model": "facebook/nllb-200-distilled-600M",
        "python_version": sys.version.split()[0],
        "nllb_load_ms": round(model_load_ms, 1),
        "nllb_translate_100_lines_ms": round(translate_ms, 1),
        "nllb_lines_per_sec": round(n / (translate_ms / 1000), 1),
        "ollama_warm_ms": round(ollama_warm_ms, 1),
        "ollama_5_lines_ms": round(ollama_5_lines_ms, 1),
        "gpu_memory_mb": round(gpu_mem_mb, 1),
        "peak_cpu_memory_mb": round(cpu_mem_mb, 1),
    }

    with open("benchmark.json", "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)
    print("\nBenchmark results:")
    for k, v in benchmark.items():
        print(f"  {k}: {v}")


# ---------------------------------------------------------------------------
# Regression testing
# ---------------------------------------------------------------------------

def run_regression(cfg: Config):
    """Translate a corpus of small SRT files and compare against expected output."""
    corpus_dir = Path("tests") / "corpus"
    if not corpus_dir.exists():
        print(f"  [ERROR] Corpus directory not found: {corpus_dir}")
        return

    eng_files = sorted(corpus_dir.glob("*_eng.srt"))
    if not eng_files:
        print(f"  [ERROR] No *_eng.srt files found in {corpus_dir}")
        return

    print(f"Regression corpus: {corpus_dir}/", flush=True)

    total_changed = 0
    total_improved = 0
    total_regressed = 0
    all_results = []

    for fpath in eng_files:
        expected_path = fpath.with_name(fpath.name.replace("_eng.srt", "_expected.srt"))
        if not expected_path.exists():
            print(f"  [SKIP] No expected output for {fpath.name}", flush=True)
            continue

        # Skip placeholder expected files (need real NLLB output generated first)
        expected_subs_check = safe_open_srt(expected_path)
        if expected_subs_check and expected_subs_check[0].text.strip().startswith("# PLACEHOLDER"):
            print(f"  [SKIP] {fpath.name} — placeholder expected, run on GPU to generate", flush=True)
            continue

        print(f"\n  {fpath.name}", flush=True)
        out = translate_fast_to_texts(fpath, cfg)
        if out is None:
            print(f"  [ERROR] Translation failed for {fpath.name}", flush=True)
            continue

        ger_texts = out
        expected_subs = safe_open_srt(expected_path)
        expected_texts = [sub.text for sub in expected_subs]

        n = min(len(ger_texts), len(expected_texts))
        if len(ger_texts) != len(expected_texts):
            print(f"  [WARN] Line count mismatch: got {len(ger_texts)}, expected {len(expected_texts)}")
            n = min(len(ger_texts), len(expected_texts))

        changed = 0
        improved = 0
        regressed = 0
        diffs = []

        en_subs = safe_open_srt(fpath)
        en_texts = [sub.text for sub in en_subs]

        glossary = load_glossary()
        names = load_names()

        for i in range(n):
            if ger_texts[i] != expected_texts[i]:
                changed += 1
                # Compare QA scores to classify
                titles = load_titles()
                got_score = score_line(en_texts[i] if i < len(en_texts) else "",
                                       ger_texts[i], glossary, names, titles)
                exp_score = score_line(en_texts[i] if i < len(en_texts) else "",
                                       expected_texts[i], glossary, names, titles)
                if got_score["score"] <= exp_score["score"]:
                    improved += 1
                else:
                    regressed += 1
                if changed <= 5:
                    diffs.append((i + 1, expected_texts[i][:60], ger_texts[i][:60]))

        total_changed += changed
        total_improved += improved
        total_regressed += regressed

        print(f"    lines: {n}")
        print(f"    changed: {changed}")
        print(f"    improved: {improved} (QA lower or equal)")
        print(f"    regressed: {regressed} (QA higher)")
        for ln, exp, got in diffs:
            print(f"    L{ln}:")
            print(f"      expected: {exp}")
            print(f"      got:      {got}")

        all_results.append({
            "file": fpath.name,
            "lines": n,
            "changed": changed,
            "improved": improved,
            "regressed": regressed,
        })

    print(f"\n{'='*50}")
    print(f"Regression Summary:")
    for r in all_results:
        status = "PASS" if r["regressed"] == 0 else "REGRESSION"
        print(f"  {r['file']}: {r['changed']}/{r['lines']} changed "
              f"(+{r['improved']} improved, -{r['regressed']} regressed) [{status}]")
    any_regression = any(r["regressed"] > 0 for r in all_results)
    total_status = "PASS" if not any_regression else "REGRESSION DETECTED"
    print(f"  Overall: {total_changed} changed, {total_improved} improved, {total_regressed} regressed")
    print(f"  Result: {total_status}", flush=True)


def translate_fast_to_texts(fpath: Path, cfg: Config) -> list[str] | None:
    """Translate a single SRT file and return German texts (without saving)."""
    try:
        subs = safe_open_srt(fpath)
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None

    eng_texts = [sub.text for sub in subs]
    n = len(eng_texts)

    glossary = load_glossary()
    short_fragments = load_short_fragments()
    names = load_names()
    protected, sfx_map = protect_sfx(eng_texts)
    protected, excl_map = protect_short_exclamations(protected)
    protected, song_map = protect_song_markers(protected)
    protected, ep_map = protect_episode_markers(protected)
    protected, name_map = protect_names(protected, names)
    protected, num_map = protect_numbers(protected)
    protected, multi_map = protect_multispeaker(protected)
    protected, short_map = protect_short_fragments(protected, glossary, short_fragments)

    # Segment: extract content (no ZZZ) for NLLB
    all_content, line_layouts = _segment_texts(protected, sfx_map, excl_map, song_map, ep_map, name_map, num_map, multi_map, short_map)
    _cum_content = [0]
    for lay in line_layouts:
        _cum_content.append(_cum_content[-1] + sum(1 for t, _ in lay if t == 'content'))

    tok, model = load_nllb(cfg)
    forced_bos = tok.convert_tokens_to_ids(cfg.tgt_lang)
    all_trans: dict[int, str] = {}

    batch_size = cfg.batch_size
    for s in range(0, n, batch_size):
        batch_end = min(s + batch_size, n)
        c_start = _cum_content[s]
        c_end = _cum_content[batch_end]
        batch_content = all_content[c_start:c_end]

        if batch_content:
            inputs = tok(batch_content, src_lang=cfg.src_lang, return_tensors="pt",
                         padding=True, truncation=True).to(cfg.device)

            outputs = model.generate(
                **inputs, forced_bos_token_id=forced_bos,
                max_new_tokens=96, num_beams=cfg.num_beams,
                no_repeat_ngram_size=cfg.no_repeat_ngram,
            )
            translated = tok.batch_decode(outputs, skip_special_tokens=True)
        else:
            translated = []

        seg_ptr = 0
        for li in range(s, batch_end):
            n_c = _cum_content[li + 1] - _cum_content[li]
            line_segs = translated[seg_ptr:seg_ptr + n_c] if n_c else []
            seg_ptr += n_c
            all_trans[li + 1] = _reconstruct_line(line_layouts[li], line_segs)

        done = min(s + batch_size, n)
        print(f"    batch: [{done}/{n}]", flush=True)

    ger_texts = [all_trans.get(i + 1, "") for i in range(n)]
    ger_texts = _canonical_restore(
        ger_texts, num_map, name_map, ep_map, song_map, sfx_map, excl_map, multi_map, short_map,
    )
    apply_song_markers(ger_texts, eng_texts)
    apply_glossary(eng_texts, ger_texts, glossary)
    preserve_names(eng_texts, ger_texts, names, glossary)
    apply_german_fixes(ger_texts, load_german_fixes())
    preserve_titles(eng_texts, ger_texts, load_titles())
    cleanup_subtitles(ger_texts)

    return ger_texts


def translate_polish_multi(files: list[Path], cfg: Config,
                           nllb_dir: Path | None = None,
                           polish_model: str | None = None,
                           max_workers: int = 3) -> dict[str, bool]:
    """Run translate_polish on multiple files concurrently.
    Each file's NLLB output is looked up via output_path_for() or nllb_dir.
    NOTE: each call to translate_polish resets ConversationMemory,
    so concurrent calls will race on the shared context. This is acceptable
    because ConversationMemory is best-effort context, not correctness-critical.
    Returns dict of filename → success bool.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    results = {}
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for fpath in files:
            nllb_path = (nllb_dir / output_path_for(fpath).name) if nllb_dir else None
            futures[executor.submit(translate_polish, fpath, cfg,
                                    nllb_path=nllb_path,
                                    polish_model=polish_model)] = fpath
        for future in as_completed(futures):
            fpath = futures[future]
            try:
                ok = future.result()
                results[fpath.name] = ok
            except Exception as e:
                print(f"  [ERROR] Polish failed for {fpath.name}: {e}", flush=True)
                results[fpath.name] = False
    return results


# (CLI lives in subtranslate.py — this module is import-only)
