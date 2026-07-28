"""Compare NLLB vs Opus-MT on E01: full pipeline timing + output diff."""
import sys, time, contextlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from translator.engine import (
    Config, _auto_device, safe_open_srt, output_path_for,
    translate_fast, translate_polish,
    load_glossary, load_german_fixes, load_names, load_titles,
    apply_german_fixes, apply_glossary, preserve_names, preserve_titles,
    cleanup_subtitles,
)

SRC = Path(r"D:\DBZ\0Prsluk of Jade\ger\Pursuit.of.Jade.E01.srt")
OUT_DIR = Path(r"D:\DBZ\0Prsluk of Jade\ger")

def run_model(label: str, model_id: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  RUNNING: {label}")
    print(f"{'='*60}\n")

    cfg = Config(
        model_id=model_id,
        device="cuda",
        batch_size=64,
        num_beams=4,
        force=True,
        mode="fast",
    )

    out_path = OUT_DIR / f"Pursuit.of.Jade.E01_{label}.srt"

    t_start = time.time()
    success = translate_fast(SRC, cfg, progress_callback=None)
    trans_elapsed = time.time() - t_start

    if not success:
        print(f"  translate_fast returned False for {label}")
        return {"error": "translate_fast failed"}

    intermediate = output_path_for(SRC)

    t0 = time.time()
    translate_polish(
        SRC, cfg, nllb_path=intermediate,
        polish_model="qwen2.5:7b",
        progress_callback=None,
    )
    polish_elapsed = time.time() - t0

    final_path = OUT_DIR / f"Pursuit.of.Jade.E01_ger.srt"
    if final_path.exists():
        final_path.rename(out_path)

    subs = safe_open_srt(out_path)
    lines = len(subs)
    total_chars = sum(len(s.text) for s in subs)
    empty = sum(1 for s in subs if not s.text.strip())

    return {
        "trans_time": trans_elapsed,
        "polish_time": polish_elapsed,
        "total_time": trans_elapsed + polish_elapsed,
        "lines": lines,
        "chars": total_chars,
        "empty": empty,
        "out_path": out_path,
    }

print("=" * 60)
print("  NLLB-600M vs Opus-MT EN-DE COMPARISON")
print("  E01 Full Pipeline (translate + polish)")
print("=" * 60)

nllb = run_model("nllb", "facebook/nllb-200-distilled-600M")
opus = run_model("opus", "Helsinki-NLP/opus-mt-en-de")

print(f"\n{'='*60}")
print(f"  RESULTS")
print(f"{'='*60}")

for label, result in [("NLLB-600M", nllb), ("Opus-MT EN-DE", opus)]:
    print(f"\n  {label}:")
    print(f"    Translate:     {result['trans_time']:.1f}s")
    print(f"    Polish:        {result['polish_time']:.1f}s")
    print(f"    Total:         {result['total_time']:.1f}s")
    print(f"    Lines:         {result['lines']}")
    print(f"    Characters:    {result['chars']}")
    print(f"    Empty lines:   {result['empty']}")

# Line-by-line comparison
print(f"\n{'='*60}")
print(f"  LINE-BY-LINE COMPARISON (first 20 lines)")
print(f"{'='*60}")

nllb_subs = safe_open_srt(nllb["out_path"])
opus_subs = safe_open_srt(opus["out_path"])

n = min(len(nllb_subs), len(opus_subs), 20)
same = 0
diff = 0
for i in range(n):
    n_text = nllb_subs[i].text.strip()
    o_text = opus_subs[i].text.strip()
    if n_text == o_text:
        same += 1
    else:
        diff += 1
        print(f"\n  Line {i+1}:")
        print(f"    NLLB:  {n_text[:120]}")
        print(f"    Opus:  {o_text[:120]}")

print(f"\n  Sampled {n} lines: {same} identical, {diff} different")

# Full comparison stats
print(f"\n{'='*60}")
print(f"  FULL COMPARISON STATS")
print(f"{'='*60}")

nllb_all = safe_open_srt(nllb["out_path"])
opus_all = safe_open_srt(opus["out_path"])

n = min(len(nllb_all), len(opus_all))
identical = 0
different = 0
for i in range(n):
    if nllb_all[i].text.strip() == opus_all[i].text.strip():
        identical += 1
    else:
        different += 1

print(f"  Total lines compared: {n}")
print(f"  Identical:            {identical} ({100*identical/n:.1f}%)")
print(f"  Different:            {different} ({100*different/n:.1f}%)")
print(f"\n  Output files:")
print(f"    NLLB: {nllb['out_path']}")
print(f"    Opus: {opus['out_path']}")
