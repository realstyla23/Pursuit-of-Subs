# Task 2: `merge_sentence_blocks()`

**Files:** Modify `translator/engine.py` (add after `calculate_cps`)

## Requirements

Add a function that merges consecutive captions into sentence blocks:

```python
def merge_sentence_blocks(subs, merge_gap_ms=500, max_block_tokens=96):
    _end_punct = re.compile(r'[.?!\]♪"]\s*$')
    _sfx_bracket = re.compile(r'^\s*\[')
    _ep_marker = re.compile(r'^\[(Episode|Trailer|Preview|Teaser)', re.IGNORECASE)
    _multi_speaker = re.compile(r'\\n-')

    blocks = []
    current = None

    for i, sub in enumerate(subs):
        text = sub.text.strip()
        if not text:
            if current:
                blocks.append(current)
                current = None
            continue

        dur = max(1, int((sub.end.ordinal - sub.start.ordinal) / 1000))
        gap = 999999
        if current and i > 0:
            gap = int((sub.start.ordinal - subs[i-1].end.ordinal) / 1000)

        if current and not _end_punct.search(current["text"]) and gap < merge_gap_ms:
            if not _sfx_bracket.match(text) and not _ep_marker.match(text):
                if not _multi_speaker.search(current["text"]):
                    merged_text = current["text"] + " " + text
                    if len(merged_text.split()) <= max_block_tokens * 2:
                        current["text"] = merged_text
                        current["indices"].append(i)
                        current["durations_ms"].append(dur)
                        continue

        if current:
            blocks.append(current)
        current = {"indices": [i], "text": text, "durations_ms": [dur]}

    if current:
        blocks.append(current)

    return blocks
```

## Interface

- Consumes: `subs: pysrt.SubRipFile`, `merge_gap_ms: int = 500`, `max_block_tokens: int = 96`
- Produces: `list[dict]` — each dict has:
  ```python
  {
      "indices": list[int],       # original caption indices in this block
      "text": str,                # merged text
      "durations_ms": list[int],  # duration of each caption in ms
  }
  ```

## Merge conditions

All must be true to merge caption i+1 into current block:
1. Current block text doesn't end in `. ? ! ♪ ] "`
2. Next caption doesn't start with `[` or `[Episode/...]`
3. Time gap between captions < `merge_gap_ms`
4. Current block is not a multi-speaker line (no `\n-`)
5. Merged text doesn't exceed `max_block_tokens * 2` words (crude NLLB token limit)

## Placement

Insert after `calculate_cps()` function in `translator/engine.py`.
