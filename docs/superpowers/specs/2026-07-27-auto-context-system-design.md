# Auto-Context System Design

## Problem
The polish prompt currently has hardcoded show context for "Pursuit of Jade". When switching to a new show, the context must be manually updated.

## Solution: Show Context Registry

### Architecture

```
config/show_contexts/          # Per-show context files
  pursuit_of_jade.json         # "Pursuit of Jade"
  never_ending_summer.json     # "Never Ending Summer"
  sweet_love.json              # "Sweet Love"
  ...
```

### Filename → Show Detection

```
Pursuit.of.Jade.E01.srt                → name="Pursuit of Jade",      slug="pursuit_of_jade"
Pursuit of Jade E01.srt                → name="Pursuit of Jade",      slug="pursuit_of_jade"
Pursuit-of-Jade.S01E01.srt             → name="Pursuit of Jade",      slug="pursuit_of_jade"
Never.Ending.Summer.E01.srt            → name="Never Ending Summer",  slug="never_ending_summer"
```

Strip episode identifiers (`E01`, `S01E01`, `1x01`, `EP01`), normalize separators to spaces.

### Context Fetch (one-time per show)

1. Extract show name → slug
2. Check `config/show_contexts/{slug}.json` — if exists, load
3. If not → try Baidu Baike (`baike.baidu.com/item/{title}`) → fallback Wikipedia REST API
4. Send raw page to Ollama (`qwen2.5:7b`) → extract: synopsis, characters (with genders), locations, key terms
5. Save to `config/show_contexts/{slug}.json`
6. Inject into polish prompt

### Integration Points

- `engine.py: translate_polish()` — replace hardcoded context block (lines 3052-3065) with `get_or_create_show_context(fpath)`
- New functions in `engine.py`:
  - `extract_show_name(fpath) → str`
  - `make_slug(name) → str`
  - `load_show_context(slug) → dict | None`
  - `save_show_context(slug, data) → None`
  - `fetch_show_context_web(show_name) → dict | None` — Baidu Baike + Wikipedia
  - `parse_context_with_llm(raw_text, show_name) → dict` — Ollama extraction
  - `get_or_create_show_context(fpath) → dict` — orchestration entry point

### Context JSON Schema

```json
{
  "show_name": "Pursuit of Jade",
  "slug": "pursuit_of_jade",
  "language": "zh",
  "synopsis": "...",
  "characters": {
    "Fan Changyu": {"gender": "female", "role": "protagonist"},
    "Xie Zheng":   {"gender": "male",   "role": "protagonist"}
  },
  "locations": ["Lin'an", "Xigu Alley"],
  "key_terms": ["Jinzhou Incident", "Blood-Robed Cavalry"]
}
```

### Prompt Injection

Replace the hardcoded block with:

```python
ctx = get_or_create_show_context(fpath)
if ctx:
    context_block = build_context_prompt(ctx)  # formatted string
else:
    context_block = ""  # no context, skip
```

### Error Handling

- Fetch failure → silently skip context (translation still works)
- Missing Baidu/Wikipedia → fallback to empty context
- Parse failure → log warning, continue without context
