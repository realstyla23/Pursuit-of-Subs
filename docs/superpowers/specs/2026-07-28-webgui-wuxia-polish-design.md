# Web GUI Wuxia Theme & Usability Polish — Design Spec

## Overview

Polish the web GUI with a Wuxia (Chinese martial arts fantasy) dark theme, fix usability issues (preview panel, parallelism defaults, model management), and clean up stale model dropdown entries.

---

## 1. HF Hub Warning Suppression

**Problem:** `You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN` prints during model load. Comes from `huggingface_hub`, which `transformers` imports — not from `transformers` itself. `engine.py:24` already suppresses transformers logs.

**Fix:** Add `huggingface_hub.logging.set_verbosity_error()` in `engine.py` alongside the existing `transformers.logging.set_verbosity_error()` call.

**Files affected:** `translator/engine.py`

---

## 2. Mode Buttons — Wuxia Card Redesign

**Problem:** Radio-button text mode selectors. Selected state too subtle (white text + blue glow). Yellow badges clash with dark theme.

### Card Spec

| Mode | Icon | Card accent | Badge |
|------|------|-------------|-------|
| Fast | ⚡ | Jade green | None |
| Full | ✦ | Gold | `Recommended` in gold outline |
| Polish | ✎ | Blue-silver | None |
| Learn | ⟳ | Warm orange | `Self-improving` in jade outline |
| Regression | ⚙ | Gray | None |

### Visual states (per card)

- **Default**: Dark (`--bg2: #252525`), subtle border, muted icon and text
- **Hover**: Background lightens ~5%, left accent strip appears
- **Selected**: Background gets accent-tinted fill (very subtle, ~10% opacity), glowing border in accent color via `box-shadow`, name text becomes bright white, left accent strip thickens
- **Disabled** (during job): All radio inputs disabled, cards visibly dimmed

### Layout

- Same horizontal flex-wrap layout, `gap: 12px`
- Each card: `min-width: 130px`, `padding: 10px 12px`, `border-radius: 6px`
- Flex column: **[icon + name] / [description] / [badge]**

### Badges

- No yellow, no star emoji
- Gold outline text for `Recommended` (`color: var(--gold)`, `border: 1px solid var(--gold)`, `border-radius: 3px`, `padding: 1px 6px`)
- Jade outline text for `Self-improving` (`color: var(--jade-light)`, `border: 1px solid var(--jade-light)`)

---

## 3. Color System

Add jade/gold to the CSS custom properties alongside existing colors:

```css
--jade: #2d8a4e;
--jade-light: #4acd7a;
--gold: #c9a84c;
--gold-light: #e8d07a;
```

Current `--accent: #4a9eff` stays as secondary accent for info elements (status bar, scrollbars). Primary accent moves to jade for interactive elements.

---

## 4. Title Styling

"Pursuit of Subs" header gets:

- `letter-spacing: 3px` for epic feel
- `text-shadow: 0 0 12px var(--gold)` subtle gold glow
- `font-weight: 700`, keep current size

---

## 5. Preview Panel Fix

**Problem:** German preview (`current_de` SSE event) reads from output file on disk (`server.py:152-157`). Output file may not exist or be unstaged yet → blank German preview.

**Fix:** Store the last translated German line in a module-level variable in `server.py`. The progress callback (`timed_progress`, `learn_progress`) already has the German text after translation. Write it to a variable instead of a file read. This is also faster (no disk I/O on the hot path).

**Files affected:** `web_gui/server.py`

**No other changes** — batch-level updates (every ~64 lines), same layout, same behavior otherwise.

---

## 6. Parallelism Default & Labels

**Changes:**

| Setting | Current | New |
|---------|---------|-----|
| Default value | `2` | `1` |
| Option 1 label | `1 (sequential)` | Unchanged |
| Option 2 label | `2 (balanced)` | `2 (parallel)` |
| Option 3 label | `3 (fast)` | `3 (aggressive)` |

**Files affected:** `translator/engine.py` (Config default line 87), `web_gui/static/index.html` (dropdown values and labels)

---

## 7. Model Management UI

**Problem:** Translate and polish model dropdowns are hardcoded HTML with no way to enter custom models.

### Translate Model (`modelIdSelect`)

- Keep existing dropdown with `opus-mt-en-de` (selected) and `nllb-200-distilled-600M`
- Add a text input directly below titled "Custom model ID"
- When dropdown changes, text input updates to match (pre-fills the name)
- User can type any HuggingFace model ID or Ollama model name directly

### Polish Model (`polishModelSelect`)

- Keep: `auto` (selected), `qwen2.5:7b`
- Remove: `gemma4:e4b`, `thinkverse/towerinstruct`
- Same text input below for custom names

### Behavior

- Typing in text input deselects dropdown (sets to blank)
- Selecting from dropdown fills the text input
- On submit, the text input value overrides the dropdown value if non-empty

---

## 8. Dropdown Cleanup

Remove `gemma4:e4b` and `thinkverse/towerinstruct` from polish model dropdown. Not referenced in polish code path.

---

## Files Affected Summary

| File | Changes |
|------|---------|
| `translator/engine.py` | Add `huggingface_hub` log suppression; change `polish_parallel` default from 2 to 1 |
| `web_gui/server.py` | Cache German text in variable instead of reading from output file |
| `web_gui/static/index.html` | Mode cards redesign (CSS + HTML); parallelism labels; model dropdowns + custom input; remove gemma4/towerinstruct; title styling; update CSS variables |
