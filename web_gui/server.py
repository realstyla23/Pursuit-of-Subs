"""Flask-based local web GUI for Subtitle Translator."""

import io
import json
import os
import sys
import time
import threading
import traceback
import webbrowser
import uuid
from pathlib import Path

from flask import Flask, Response, jsonify, request, send_from_directory

sys.path.insert(0, str(Path(__file__).parent.parent))
from translator import (
    __version__, Config, _auto_device, find_srt_files, output_path_for,
    translate_fast, translate_polish,
    _checkpoint_path,
)

app = Flask(__name__, static_folder=None)

# ---------------------------------------------------------------------------
# Global state — thread-safe with locks
# ---------------------------------------------------------------------------

_events: list[dict] = []
_events_lock = threading.Lock()
_worker_thread: threading.Thread | None = None
_cancel_event = threading.Event()
_job_running = False
_job_lock = threading.Lock()

_CUDA_AVAILABLE = False
_GPU_TOTAL_MB = 0
try:
    import torch
    _CUDA_AVAILABLE = torch.cuda.is_available()
    if _CUDA_AVAILABLE:
        _GPU_TOTAL_MB = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
except Exception:
    pass


def push_event(event: str, data: dict):
    with _events_lock:
        _events.append({"event": event, "data": data})


def pending_events(since: int) -> list[dict]:
    with _events_lock:
        return list(enumerate(_events[since:], start=since))


def _format_event(idx: int, ev: dict) -> str:
    lines = [
        f"id: {idx}",
        f"event: {ev['event']}",
        f"data: {json.dumps(ev['data'], ensure_ascii=False)}",
    ]
    return "\n".join(lines) + "\n\n"


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def _worker(files: list[Path], cfg: Config, output_dir: str | None,
            mode: str, polish_model: str, polish_parallel: int,
            ollama_host: str, resume: bool):
    global _job_running
    try:
        total_files = len(files)
        for idx, fpath in enumerate(files):
            if _cancel_event.is_set():
                push_event("log", {"message": "Cancelled by user", "level": "warn"})
                break

            push_event("file_progress", {"index": idx + 1, "total": total_files})
            push_event("current_file", {"name": fpath.name,
                                        "episode": idx + 1, "total": total_files})
            push_event("log", {"message": f"[{idx + 1}/{total_files}] {fpath.name}",
                               "level": "info"})

            if output_dir:
                out = Path(output_dir) / output_path_for(fpath).name
            else:
                out = output_path_for(fpath)

            has_checkpoint = _checkpoint_path(out).exists()
            if out.exists() and not cfg.force and not has_checkpoint:
                push_event("log", {"message": f"  [SKIP] {fpath.name} (already exists)",
                                   "level": "skip"})
                push_event("file_done", {"name": fpath.name, "stats": {"skipped": True}})
                continue

            file_cfg = Config(
                src_lang=cfg.src_lang,
                tgt_lang=cfg.tgt_lang,
                device=cfg.device,
                batch_size=cfg.batch_size,
                num_beams=cfg.num_beams,
                ollama_model=polish_model or cfg.ollama_model,
                ollama_host=ollama_host,
                input_dir=str(fpath.parent),
                force=cfg.force,
                resume=resume or has_checkpoint,
                use_tm=cfg.use_tm,
                proxy_base_url=cfg.proxy_base_url,
                proxy_api_key=cfg.proxy_api_key,
                mode="fast",
                polish_parallel=polish_parallel,
            )

            # Pre-read SRT for live preview
            import pysrt
            try:
                eng_subs = pysrt.open(str(fpath), encoding="utf-8")
                eng_texts = [s.text for s in eng_subs]
            except Exception as e:
                push_event("log", {"message": f"  [SKIP] {fpath.name} (unreadable SRT: {e})",
                                   "level": "skip"})
                push_event("file_done", {"name": fpath.name,
                                         "stats": {"error": f"unreadable SRT: {e}"}})
                continue

            push_event("step_changed", {"step": "Translating (NLLB)"})

            def timed_progress(done, total):
                if _cancel_event.is_set():
                    raise KeyboardInterrupt()
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (total - done) / rate if rate > 0 else 0
                push_event("progress", {"done": done, "total": total})
                push_event("speed_eta", {"speed": round(rate, 1), "eta": round(eta, 1)})
                if done > 0 and done - 1 < len(eng_texts):
                    push_event("current_en", {"text": eng_texts[done - 1]})
                # Read current DE from output checkpoint
                try:
                    ger_subs = pysrt.open(str(out), encoding="utf-8")
                    if done <= len(ger_subs):
                        push_event("current_de", {"text": ger_subs[done - 1].text})
                except Exception:
                    pass

            t0 = time.time()

            try:
                push_event("step_changed", {"step": "Protecting placeholders"})
                success = translate_fast(fpath, file_cfg,
                                         progress_callback=timed_progress,
                                         output_path=out)
            except KeyboardInterrupt:
                push_event("log", {"message": "  Cancelled by user", "level": "warn"})
                break
            except Exception as e:
                push_event("error", {"message": f"Error translating {fpath.name}: {e}"})
                push_event("file_done", {"name": fpath.name,
                                         "stats": {"error": str(e)}})
                continue

            if not success:
                push_event("file_done", {"name": fpath.name,
                                         "stats": {"error": "translate_fast returned False"}})
                continue

            # Polish pass
            if mode in ("polish", "full"):
                model_label = polish_model.split(":")[0] if polish_model else "qwen"
                push_event("step_changed", {"step": f"Polishing ({model_label})"})
                try:
                    translate_polish(fpath, file_cfg, nllb_path=out,
                                    polish_model=polish_model or None)
                except Exception as e:
                    push_event("log", {"message": f"  [WARN] Polish error (non-fatal): {e}",
                                       "level": "warn"})

            # QA report
            stats = {"skipped": False}
            qa_path = out.with_suffix(".qa_report.json")
            if qa_path.exists():
                try:
                    with open(qa_path, encoding="utf-8") as f:
                        qa_data = json.load(f)
                    stats["qa_score"] = qa_data.get("total_score", 0)
                    stats["suspicious"] = sum(
                        1 for s in qa_data.get("scores", {}).values()
                        if isinstance(s, (int, float)) and s >= 5
                    )
                except Exception:
                    pass

            try:
                out_subs = pysrt.open(str(out), encoding="utf-8")
                stats["lines"] = len(out_subs)
            except Exception:
                pass

            push_event("file_done", {"name": fpath.name, "stats": stats})
            push_event("log", {"message": f"  DONE: {out.name}", "level": "done"})

        push_event("step_changed", {"step": "Complete"})
        push_event("all_done", {})

    except Exception as e:
        push_event("error", {"message": f"Worker error: {e}\n{traceback.format_exc()}"})
        push_event("all_done", {})
    finally:
        _job_running = False


# ---------------------------------------------------------------------------
# Error handlers — always return JSON
# ---------------------------------------------------------------------------

@app.errorhandler(400)
def bad_request(e):
    return jsonify({"error": str(e)}), 400

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "not found"}), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": str(e)}), 405

@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "internal server error"}), 500


# ---------------------------------------------------------------------------
# Routes — API
# ---------------------------------------------------------------------------

@app.route("/api/start", methods=["POST"])
def api_start():
    global _worker_thread, _job_running

    with _job_lock:
        if _job_running:
            return jsonify({"error": "Job already running"}), 409
        _job_running = True
        _cancel_event.clear()
        with _events_lock:
            _events.clear()

    body = request.get_json(force=True)
    file_paths = [Path(p) for p in body.get("files", [])]
    mode = body.get("mode", "full")
    device = body.get("device", "cuda")
    batch_size = body.get("batch_size", 64)
    num_beams = body.get("num_beams", 4)
    resume = body.get("resume", False)
    output_dir = body.get("output_dir") or None

    cfg = Config(
        device=_auto_device(device),
        batch_size=batch_size,
        num_beams=num_beams,
        force=True,
        use_tm=body.get("use_tm", False),
        proxy_base_url=body.get("proxy_base_url", ""),
        proxy_api_key=body.get("proxy_api_key", ""),
    )

    polish_model = body.get("polish_model", "gemma4:e4b")
    polish_parallel = body.get("polish_parallel", 1)
    ollama_host = body.get("ollama_host", "http://127.0.0.1:11434")

    _worker_thread = threading.Thread(
        target=_worker,
        args=(file_paths, cfg, output_dir, mode, polish_model, polish_parallel, ollama_host, resume),
        daemon=True,
    )
    _worker_thread.start()

    return jsonify({"status": "started", "files": len(file_paths)})


@app.route("/api/cancel", methods=["POST"])
def api_cancel():
    _cancel_event.set()
    return jsonify({"status": "cancelling"})


@app.route("/api/events")
def api_events():
    last_id = request.headers.get("Last-Event-ID")
    since = int(last_id) + 1 if last_id else 0

    def generate():
        nonlocal since
        try:
            while True:
                evts = pending_events(since)
                if evts:
                    for idx, ev in evts:
                        yield _format_event(idx, ev)
                        if ev["event"] == "all_done":
                            return
                    since = evts[-1][0] + 1
                else:
                    time.sleep(0.2)
        except GeneratorExit:
            pass
        except Exception:
            traceback.print_exc()

    return Response(generate(), mimetype="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "X-Accel-Buffering": "no",
                        "Connection": "keep-alive",
                    })


@app.route("/api/status")
def api_status():
    gpu_free_mb = 0
    gpu_used_mb = 0
    if _CUDA_AVAILABLE:
        try:
            import torch
            free_b, total_b = torch.cuda.mem_get_info()
            gpu_free_mb = free_b / 1024 / 1024
            gpu_used_mb = (total_b - free_b) / 1024 / 1024
        except Exception:
            pass

    return jsonify({
        "cuda_available": _CUDA_AVAILABLE,
        "gpu_total_mb": round(_GPU_TOTAL_MB, 0),
        "gpu_used_mb": round(gpu_used_mb, 0),
        "gpu_free_mb": round(gpu_free_mb, 0),
        "job_running": _job_running,
        "version": __version__,
    })


@app.route("/api/browse")
def api_browse():
    path_str = request.args.get("path", "")
    if not path_str:
        if sys.platform == "win32":
            import ctypes
            drives = []
            bitmask = ctypes.windll.kernel32.GetLogicalDrives()
            for letter in range(26):
                if bitmask & (1 << letter):
                    drives.append(f"{chr(65 + letter)}:\\")
            return jsonify({"parent": None, "entries": [
                {"name": d, "is_dir": True, "path": d} for d in drives
            ]})
        return jsonify({"parent": None, "entries": []})

    path = Path(path_str)
    if not path.exists():
        return jsonify({"error": "Path not found"}), 404

    parent = str(path.parent) if path.parent != path else None
    entries = []
    try:
        sorted_entries = sorted(path.iterdir(), key=lambda x: (not x.is_dir(), x.name.lower()))
        for child in sorted_entries:
            if child.is_dir():
                entries.append({
                    "name": child.name,
                    "is_dir": True,
                    "path": str(child),
                })
            elif child.suffix.lower() == ".srt":
                entries.append({
                    "name": child.name,
                    "is_dir": False,
                    "path": str(child),
                    "size": child.stat().st_size,
                })
    except PermissionError:
        pass

    return jsonify({"parent": parent, "entries": entries})


@app.route("/api/open-folder", methods=["POST"])
def api_open_folder():
    body = request.get_json(force=True)
    path_str = body.get("path", "")
    path = Path(path_str) if path_str else Path.cwd()
    if not path.exists():
        path = Path.cwd()
    try:
        os.startfile(str(path))
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


_UPLOAD_DIR = Path(__file__).parent.parent / "uploads"


def _save_upload(name: str, content: str | bytes, is_bytes: bool = False) -> tuple[Path, int]:
    """Save an uploaded file preserving the original name, inside a unique batch subdirectory."""
    _UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    batch_dir = _UPLOAD_DIR / uuid.uuid4().hex[:8]
    batch_dir.mkdir(exist_ok=True)
    dest = batch_dir / name
    if is_bytes:
        dest.write_bytes(content)
    else:
        dest.write_text(content, encoding="utf-8")
    return dest.resolve(), dest.stat().st_size


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if request.content_type and "json" in request.content_type:
        body = request.get_json(force=True, silent=True) or {}
        name = body.get("name", "")
        content = body.get("content", "")
        if not name or not content:
            return jsonify({"error": "name and content required"}), 400
        if not name.lower().endswith(".srt"):
            return jsonify({"error": "only .srt files accepted"}), 400
        path, size = _save_upload(name, content)
        return jsonify({"path": str(path), "size": size})
    else:
        f = request.files.get("file")
        if not f or not f.filename:
            return jsonify({"error": "no file provided"}), 400
        if not f.filename.lower().endswith(".srt"):
            return jsonify({"error": "only .srt files accepted"}), 400
        path, size = _save_upload(f.filename, f.read(), is_bytes=True)
        return jsonify({"path": str(path), "size": size})


# ---------------------------------------------------------------------------
# Routes — static files
# ---------------------------------------------------------------------------

_STATIC_DIR = Path(__file__).parent / "static"


@app.route("/")
def index():
    return send_from_directory(str(_STATIC_DIR), "index.html")


@app.route("/<path:filename>")
def static_files(filename):
    return send_from_directory(str(_STATIC_DIR), filename)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------

def launch_web_gui(port: int = 5000, open_browser: bool = True):
    url = f"http://127.0.0.1:{port}"
    print(f"  Web GUI: {url}")
    print(f"  Press Ctrl+C to stop the server.")

    if open_browser:
        webbrowser.open(url)

    try:
        app.run(host="127.0.0.1", port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        pass
    finally:
        _cancel_event.set()
        print("\n  Server stopped.")


if __name__ == "__main__":
    launch_web_gui()
