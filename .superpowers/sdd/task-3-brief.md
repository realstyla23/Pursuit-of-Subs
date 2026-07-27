# Task 3: `split_sentence_blocks()`

**Files:** Modify `translator/engine.py` (add after `merge_sentence_blocks`)

## Requirements

Add a function that splits translated merged blocks back into per-caption text:

```python
def split_sentence_blocks(ger_blocks, merge_blocks, subs):
    result = [""] * len(subs)

    for block_idx, blk in enumerate(merge_blocks):
        indices = blk["indices"]
        if len(indices) == 1:
            result[indices[0]] = ger_blocks[block_idx] if block_idx < len(ger_blocks) else blk["text"]
            continue

        src_text = blk["text"]
        ger_text = ger_blocks[block_idx] if block_idx < len(ger_blocks) else src_text

        src_lens = [len(subs[j].text) for j in indices]
        total_src = sum(src_lens)

        if total_src == 0:
            for j in indices:
                result[j] = ger_text
            continue

        # Compute split positions using ratio
        split_positions = []
        cum_ratio = 0
        for j in range(len(indices) - 1):
            cum_ratio += src_lens[j]
            ratio = cum_ratio / total_src
            pos = int(ratio * len(ger_text))
            split_positions.append(pos)

        # Snap to word boundaries
        text_parts = []
        prev = 0
        for sp in split_positions:
            adjusted = sp
            if adjusted > prev and adjusted < len(ger_text):
                while adjusted < len(ger_text) and ger_text[adjusted] == ' ':
                    adjusted += 1
                while adjusted > prev and ger_text[adjusted-1] != ' ' and ger_text[adjusted-1] != '\n':
                    adjusted -= 1
                while adjusted > 0 and ger_text[adjusted-1] == ' ':
                    adjusted -= 1
            text_parts.append(ger_text[prev:adjusted].strip())
            prev = adjusted

        text_parts.append(ger_text[prev:].strip())

        # Assign with CPS check
        for j, part in zip(indices, text_parts):
            dur = subs[j].end.ordinal - subs[j].start.ordinal
            cps = calculate_cps(part, dur)
            if cps > 35:
                text_parts = [blk["text"]] * len(indices)
                break

        for j, part in zip(indices, text_parts):
            result[j] = part

    return result
```

## Interface

- Consumes: `ger_blocks: list[str]` (translated block texts), `merge_blocks: list[dict]` (from Task 2), `subs: pysrt.SubRipFile`
- Produces: `list[str]` — per-caption translated texts, same length as `subs`

## Placement

Insert after `merge_sentence_blocks()` in `translator/engine.py`.

## Dependencies

- Uses `calculate_cps()` from Task 1 (already in engine.py)
- Uses `re` (already imported)
