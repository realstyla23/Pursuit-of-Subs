"""Compare Opus-MT output against reference subs."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from translator.engine import safe_open_srt

REF_DIR = Path(r"D:\DBZ\0Prsluk of Jade\pursuit subs")
OUT_DIR = Path(r"D:\DBZ\0Prsluk of Jade\ger")

episodes = ["E01","E02","E03","E04","E05","E06","E07","E08","E09","E10"]

total_identical = 0
total_lines = 0

for ep in episodes:
    ref_path = REF_DIR / f"Pursuit.of.Jade.{ep}-(German).srt"
    out_path = OUT_DIR / f"Pursuit.of.Jade.{ep}_ger.srt"

    if not ref_path.exists() or not out_path.exists():
        print(f"  {ep}: missing files (ref={ref_path.exists()}, out={out_path.exists()})")
        continue

    ref_subs = safe_open_srt(ref_path)
    out_subs = safe_open_srt(out_path)

    n = min(len(ref_subs), len(out_subs))
    identical = 0
    different = 0
    ref_only = 0
    out_only = 0

    for i in range(n):
        r = ref_subs[i].text.strip()
        o = out_subs[i].text.strip()
        if r == o:
            identical += 1
        else:
            different += 1

    if len(ref_subs) != len(out_subs):
        print(f"  {ep}: line count mismatch ref={len(ref_subs)} out={len(out_subs)}")

    pct = 100 * identical / n if n > 0 else 0
    total_identical += identical
    total_lines += n
    print(f"  {ep}: {identical}/{n} identical ({pct:.1f}%)")

total_pct = 100 * total_identical / total_lines if total_lines > 0 else 0
print(f"\n  TOTAL: {total_identical}/{total_lines} ({total_pct:.1f}%) across {len(episodes)} episodes")
