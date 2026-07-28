# Web GUI Wuxia Theme & Usability Polish — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Apply Wuxia dark theme to web GUI, fix preview panel German side, fix parallelism defaults, add custom model input, remove stale dropdown entries.

**Architecture:** Three independent areas of change: (1) engine.py for HF warning suppression and parallelism default, (2) server.py for preview panel fix, (3) index.html for all visual/UX redesign. Each is independently testable.

**Tech Stack:** Python (Flask + transformers + huggingface_hub), HTML/CSS/JS (vanilla, no framework)

## Global Constraints

- All changes must be backward-compatible — existing modes/CLI unaffected
- Preview fix must not add disk I/O to translation hot path
- Custom model input in web GUI must accept any HuggingFace model ID or Ollama name
- Dark theme only — no light mode

---

### Task 1: HF Warning Suppression + Parallelism Default

**Files:**
- Modify: `translator/engine.py:24` (HF warning), `translator/engine.py:87` (parallelism default)

**Interfaces:**
- Consumes: nothing
- Produces: silenced HF hub warnings during model load; `polish_parallel` defaults to 1

- [ ] **Step 1: Add huggingface_hub warning suppression**

In `translator/engine.py`, after the existing `transformers.logging.set_verbosity_error()` at line 24, add:

```python
import huggingface_hub
huggingface_hub.logging.set_verbosity_error()
```

- [ ] **Step 2: Change polish_parallel default**

In `translator/engine.py`, find the Config dataclass (around line 87) and change:

```python
polish_parallel: int = 2
```

to:

```python
polish_parallel: int = 1
```

- [ ] **Step 3: Commit**

```bash
git add translator/engine.py
git commit -m "fix: suppress HF hub warning, default polish_parallel to 1"
```

---

### Task 2: Preview Panel Fix — Cache German Text In-Memory

**Files:**
- Modify: `translator/engine.py` (progress callback calls at lines 2389-2390, 2438-2439, 3533-3534, 4812-4813)
- Modify: `web_gui/server.py` (progress callback handlers at lines 149-157 and 233-239)

**Problem:** Server.py reads the output SRT from disk in the progress callback to get `current_de`. If the file hasn't been written/flushed yet, German preview shows empty. This also adds I/O to the hot translation path.

**Fix:** Change progress callbacks from `progress_callback(done, total)` to `progress_callback(done, total, de_text=None)`. Engine passes the current German subtitle text. Server uses it directly.

- [ ] **Step 1: Update translate_fast progress callbacks in engine.py**

At line 2390, change:
```python
progress_callback(done, n)
```
to:
```python
progress_callback(done, n, all_trans[done] if done < len(all_trans) else None)
```

At line 2439, same change:
```python
progress_callback(done, n)
```
to:
```python
progress_callback(done, n, all_trans[done] if done < len(all_trans) else None)
```

- [ ] **Step 2: Update polish progress callback in engine.py**

At line 3534, change:
```python
progress_callback(done, len(suspicious))
```
to:
```python
progress_callback(done, len(suspicious))
```
(Note: polish mode transforms lines, so getting the exact text requires extra mapping. For now, in polish mode, we pass `None` so server falls through gracefully. The main fix is for translate mode.)

- [ ] **Step 3: Update learn progress callback in engine.py**

At line 4813, change:
```python
progress_callback(pass2_done * batch_size, n)
```
to:
```python
progress_callback(pass2_done * batch_size, n)
```
(Same note as polish mode — learn mode transforms lines post-translation. Pass `None` for now.)

- [ ] **Step 4: Update server.py progress callback for fast mode**

In `web_gui/server.py`, in the `timed_progress` function (lines 141-157), replace the disk read with the passed text:

```python
def timed_progress(done, total, de_text=None):
    if _cancel_event.is_set():
        raise KeyboardInterrupt()
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    push_event("progress", {"done": done, "total": total})
    push_event("speed_eta", {"speed": round(rate, 1), "eta": round(eta, 1)})
    if done > 0 and done - 1 < len(eng_texts):
        push_event("current_en", {"text": eng_texts[done - 1]})
    if de_text is not None:
        push_event("current_de", {"text": de_text})
```

- [ ] **Step 5: Update server.py progress callback for learn mode**

In `web_gui/server.py`, in the `learn_progress` function (lines 225-240), apply the same change:

```python
def learn_progress(done, total, de_text=None):
    if _cancel_event.is_set():
        raise KeyboardInterrupt()
    lap_elapsed = time.time() - t0
    rate = done / lap_elapsed if lap_elapsed > 0 else 0
    eta = (total - done) / rate if rate > 0 else 0
    push_event("progress", {"done": done, "total": total})
    push_event("speed_eta", {"speed": round(rate, 1), "eta": round(eta, 1)})
    if done > 0 and done - 1 < len(eng_texts):
        push_event("current_en", {"text": eng_texts[done - 1]})
    if de_text is not None:
        push_event("current_de", {"text": de_text})
```

- [ ] **Step 6: Verify the fix**

Run: `python subtranslate.py --web-gui`
- Add an SRT file, start translation in Fast mode
- Verify German preview box shows text alongside English during translation
- Verify no errors in terminal

- [ ] **Step 7: Commit**

```bash
git add translator/engine.py web_gui/server.py
git commit -m "fix: cache German preview text in memory instead of disk reads"
```

---

### Task 3: Web GUI Wuxia Redesign (index.html)

**Files:**
- Modify: `web_gui/static/index.html`

**Scope:** CSS variables, title styling, mode cards redesign, parallelism labels, dropdown cleanup, custom model input.

#### Step 1: Add jade/gold CSS variables

Replace the `:root` block (line 9):

```css
:root{--bg:#1e1e1e;--bg2:#252525;--bg3:#1a1a1a;--bg4:#121212;--fg:#ddd;--fg2:#aaa;--fg3:#888;--border:#444;--border2:#333;--accent:#4a9eff;--green:#1a6b3c;--green2:#2a8b4c;--green3:#51cf66;--red:#6b1a1a;--red2:#ff6b6b;--yellow:#ffd93d;--purple:#da77f2;--blue:#6bcbff;font-size:13px}
```

Replace with jade/gold colors added:

```css
:root{--bg:#1e1e1e;--bg2:#252525;--bg3:#1a1a1a;--bg4:#121212;--fg:#ddd;--fg2:#aaa;--fg3:#888;--border:#444;--border2:#333;--accent:#2d8a4e;--accent2:#4acd7a;--gold:#c9a84c;--gold-light:#e8d07a;--jade:#2d8a4e;--jade-light:#4acd7a;--green:#1a6b3c;--green2:#2a8b4c;--green3:#51cf66;--red:#6b1a1a;--red2:#ff6b6b;--yellow:#ffd93d;--purple:#da77f2;--blue:#6bcbff;--blue-silver:#7a9bb5;--warm-orange:#d4874a;--mode-fast:#4acd7a;--mode-full:#c9a84c;--mode-polish:#7a9bb5;--mode-learn:#d4874a;--mode-regr:#888;font-size:13px}
```

Note: `--accent` changes from `#4a9eff` (blue) to `#2d8a4e` (jade). Info elements (CUDA indicator) will now use jade instead of blue.

#### Step 2: Style the title

Replace the header CSS rule (line 12):

```css
.header h1{font-size:15px;font-weight:600;color:var(--fg)}
```

With:

```css
.header h1{font-size:15px;font-weight:700;color:var(--fg);letter-spacing:3px;text-shadow:0 0 12px var(--gold)}
```

#### Step 3: Replace mode card CSS (lines 71-77)

Remove the old `.mode-options`, `.mode-option`, `.mode-label`, `.mode-desc`, `.mode-badge` rules (lines 71-77).

Add new mode card CSS:

```css
.mode-options{display:flex;gap:12px;flex-wrap:wrap}
.mode-option{display:flex;flex-direction:column;gap:4px;cursor:pointer;background:var(--bg2);border:1px solid var(--border2);border-radius:6px;padding:10px 12px;min-width:130px;transition:background .15s,border-color .15s,box-shadow .15s;position:relative;overflow:hidden}
.mode-option:hover{background:#2d2d2d;border-color:#555}
.mode-option input[type=radio]{display:none}
.mode-option .mode-card-top{display:flex;align-items:center;gap:8px}
.mode-option .mode-icon{font-size:16px;line-height:1;transition:transform .15s}
.mode-option .mode-label{font-size:12px;font-weight:600;color:var(--fg2);transition:color .15s}
.mode-option .mode-desc{font-size:10px;color:var(--fg3);padding-left:24px}
.mode-option .mode-badge{display:inline-block;font-size:9px;font-weight:600;border-radius:3px;padding:1px 6px;margin-top:2px;align-self:flex-start}

/* Selected state */
.mode-option.selected{border-color:var(--accent2);box-shadow:0 0 8px rgba(74,205,122,.3)}
.mode-option.selected .mode-label{color:#fff}
.mode-option.selected .mode-icon{transform:scale(1.15)}

/* Per-mode accent colors */
.mode-option[data-mode=fast] .mode-icon{color:var(--mode-fast)}
.mode-option[data-mode=fast].selected{border-color:var(--mode-fast);box-shadow:0 0 8px rgba(74,205,122,.3)}
.mode-option[data-mode=full] .mode-icon{color:var(--mode-full)}
.mode-option[data-mode=full].selected{border-color:var(--mode-full);box-shadow:0 0 8px rgba(201,168,76,.3)}
.mode-option[data-mode=polish] .mode-icon{color:var(--mode-polish)}
.mode-option[data-mode=polish].selected{border-color:var(--mode-polish);box-shadow:0 0 8px rgba(122,155,181,.3)}
.mode-option[data-mode=learn] .mode-icon{color:var(--mode-learn)}
.mode-option[data-mode=learn].selected{border-color:var(--mode-learn);box-shadow:0 0 8px rgba(212,135,74,.3)}
.mode-option[data-mode=regression] .mode-icon{color:var(--mode-regr)}
.mode-option[data-mode=regression].selected{border-color:var(--mode-regr);box-shadow:0 0 8px rgba(136,136,136,.3)}

/* Badge variants */
.mode-badge.recommended{color:var(--gold);border:1px solid var(--gold)}
.mode-badge.self-improving{color:var(--jade-light);border:1px solid var(--jade-light)}
```

#### Step 4: Update mode card HTML (lines 201-229)

Replace the current mode card HTML:

```html
<div class="mode-options" id="modeOptions">
  <label class="mode-option" data-mode="fast">
    <input type="radio" name="mode" value="fast">
    <div class="mode-card-top">
      <span class="mode-icon">⚡</span>
      <span class="mode-label">Fast</span>
    </div>
    <span class="mode-desc">Fastest translation</span>
  </label>
  <label class="mode-option" data-mode="full">
    <input type="radio" name="mode" value="full" checked>
    <div class="mode-card-top">
      <span class="mode-icon">✦</span>
      <span class="mode-label">Full</span>
    </div>
    <span class="mode-desc">Translate + polish</span>
    <span class="mode-badge recommended">Recommended</span>
  </label>
  <label class="mode-option" data-mode="polish">
    <input type="radio" name="mode" value="polish">
    <div class="mode-card-top">
      <span class="mode-icon">✎</span>
      <span class="mode-label">Polish</span>
    </div>
    <span class="mode-desc">Polish existing output</span>
  </label>
  <label class="mode-option" data-mode="learn">
    <input type="radio" name="mode" value="learn">
    <div class="mode-card-top">
      <span class="mode-icon">⟳</span>
      <span class="mode-label">Learn</span>
    </div>
    <span class="mode-desc">Translate + auto-fix</span>
    <span class="mode-badge self-improving">Self-improving</span>
  </label>
  <label class="mode-option" data-mode="regression">
    <input type="radio" name="mode" value="regression">
    <div class="mode-card-top">
      <span class="mode-icon">⚙</span>
      <span class="mode-label">Regression</span>
    </div>
    <span class="mode-desc">Developer testing</span>
  </label>
</div>
```

#### Step 5: Update parallelism dropdown labels

In the GPU grid (line 252-257), change:

```html
<label>Parallel:</label>
<select id="polishParallelSelect">
  <option value="1">1 (sequential)</option>
  <option value="2" selected>2 (balanced)</option>
  <option value="3">3 (fast)</option>
</select>
```

To (note: default changes from 2 to 1):

```html
<label>Parallel:</label>
<select id="polishParallelSelect">
  <option value="1" selected>1 (sequential)</option>
  <option value="2">2 (parallel)</option>
  <option value="3">3 (aggressive)</option>
</select>
```

#### Step 6: Remove gemma4/towerinstruct from polish model dropdown

In the GPU grid (lines 245-251), change:

```html
<label>Polish:</label>
<select id="polishModelSelect">
  <option value="auto" selected>AUTO (NVIDIA → Ollama)</option>
  <option value="qwen2.5:7b">qwen2.5:7b (Ollama, 4.7 GB)</option>
  <option value="thinkverse/towerinstruct">TowerInstruct 7B (translation-specialized, 3.8 GB)</option>
  <option value="gemma4:e4b">gemma4:e4b (thinking model, may not work)</option>
</select>
```

To:

```html
<label>Polish:</label>
<select id="polishModelSelect">
  <option value="auto" selected>Auto (Ollama)</option>
  <option value="qwen2.5:7b">qwen2.5:7b (Ollama, 4.7 GB)</option>
</select>
```

#### Step 7: Add custom model input below translate and polish dropdowns

After the translate model select (line 244), add:

```html
<div class="full-row model-custom-row" style="margin-top:2px">
  <input type="text" id="customTranslateModel" placeholder="Or type any model name..." style="width:100%;font-size:11px">
</div>
```

After the polish model select (after line 251, before the parallel label), add:

```html
<div class="full-row model-custom-row" style="margin-top:2px">
  <input type="text" id="customPolishModel" placeholder="Or type any model name..." style="width:100%;font-size:11px">
</div>
```

#### Step 8: Add JS for custom model input behavior

In the `<script>` section, after the variable declarations (around line 387), add event handlers:

```javascript
// ---- Custom model input ----
$('modelIdSelect').addEventListener('change', function() {
  $('customTranslateModel').value = this.value;
});
$('customTranslateModel').addEventListener('input', function() {
  if (this.value) $('modelIdSelect').value = '';
});
$('polishModelSelect').addEventListener('change', function() {
  $('customPolishModel').value = this.value;
});
$('customPolishModel').addEventListener('input', function() {
  if (this.value) $('polishModelSelect').value = '';
});
```

#### Step 9: Update JS model_id/polish_model submission (lines 821-822)

In the start button handler, change the model submission to use custom input if non-empty:

```javascript
model_id: $('customTranslateModel').value || $('modelIdSelect').value,
polish_model: $('customPolishModel').value || $('polishModelSelect').value,
```

Also remove the `var modelIdSelect` etc since we already read them via `$()`.

#### Step 10: Add JS for mode card selection state

Add this to the script section (before or near the custom model input code):

```javascript
// ---- Mode card selection ----
var modeRadios = document.querySelectorAll('.mode-option input[type=radio]');
for (var i = 0; i < modeRadios.length; i++) {
  modeRadios[i].addEventListener('change', function() {
    var cards = document.querySelectorAll('.mode-option');
    for (var j = 0; j < cards.length; j++) cards[j].classList.remove('selected');
    if (this.checked) this.closest('.mode-option').classList.add('selected');
  });
}
// Set initial state
var checkedMode = document.querySelector('.mode-option input[type=radio]:checked');
if (checkedMode) checkedMode.closest('.mode-option').classList.add('selected');
```

#### Step 11: Update mode disabled logic (line 847-848)

Change line 847-848:

```javascript
var modeInputs = document.querySelectorAll('.mode-option input');
for (var i = 0; i < modeInputs.length; i++) modeInputs[i].disabled = true;
```

To also disable the custom model inputs:

```javascript
var modeInputs = document.querySelectorAll('.mode-option input');
for (var i = 0; i < modeInputs.length; i++) modeInputs[i].disabled = true;
$('customTranslateModel').disabled = true;
$('customPolishModel').disabled = true;
```

And in `onJobEnd()` (around line 897), add re-enable:

```javascript
$('customTranslateModel').disabled = false;
$('customPolishModel').disabled = false;
```

#### Step 12: Verify

- Open `http://127.0.0.1:5000` after running `python subtranslate.py --web-gui`
- Verify Wuxia dark theme: jade/gold colors, title styling, card-style mode selectors
- Click each mode card — verify card border glow and icon scale match the mode color
- Verify parallelism dropdown shows 1 (sequential) as default, labels changed
- Verify polish model dropdown no longer has gemma4/towerinstruct
- Type a custom model name in translate/polish input — verify dropdown clears
- Select from dropdown — verify input fills with value
- Start a translation — verify all controls disable (including custom inputs)
- Verify on job end — controls re-enable

#### Step 13: Commit

```bash
git add web_gui/static/index.html
git commit -m "feat: wuxia dark theme redesign, custom model input, fix parallelism/ui labels"
```

---

### Task 4: Final Commit with Design Doc

- [ ] **Step 1: Stage and commit the design doc**

```bash
git add docs/superpowers/specs/2026-07-28-webgui-wuxia-polish-design.md
git add docs/superpowers/plans/2026-07-28-webgui-wuxia-polish.md
git commit -m "docs: add webgui wuxia redesign design doc and plan"
```

- [ ] **Step 2: Push all commits**

```bash
git push origin master
```
