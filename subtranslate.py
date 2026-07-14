"""Subtitle Translator v4 — CLI entry point.

Usage:
    python subtranslate.py --mode fast --input-dir .
    python subtranslate.py --gui           (launches GUI)
    python subtranslate.py --test
"""

import argparse
import sys

from translator import (
    __version__, Config, _auto_device,
    find_srt_files, output_path_for,
    translate_fast, translate_polish, translate_llm,
    run_test, run_benchmark, run_regression,
    _checkpoint_path,
)


def main():
    p = argparse.ArgumentParser(
        description=f"Subtitle Translator v{__version__} — NLLB-600M GPU batch translator"
    )
    p.add_argument("--mode", choices=["fast", "polish", "full", "test", "benchmark", "regression", "llm"],
                   default="fast")
    p.add_argument("--input-dir", default=".")
    p.add_argument("--force", action="store_true")
    p.add_argument("--batch-size", type=int, default=64,
                   help="NLLB GPU batch size (default 64)")
    p.add_argument("--device", default="cuda", help="cuda or cpu")
    p.add_argument("--num-beams", type=int, default=4,
                   help="NLLB beam search width (default 4, higher=better quality but slower)")
    p.add_argument("--resume", action="store_true",
                   help="Resume interrupted translation from checkpoint")
    p.add_argument("--cache", action="store_true",
                   help="Enable translation-memory cache (reuses exact-match EN→DE from prior runs). "
                        "Off by default to avoid stale data.")
    p.add_argument("--ignore-tm", "--fresh", dest="ignore_tm", action="store_true",
                   help="Deprecated: TM is off by default now. Use --cache to enable.")
    p.add_argument("--proxy-base-url",
                   help="OpenAI-compatible proxy URL (e.g. http://127.0.0.1:6446). Falls back to PROXY_BASE_URL env var.")
    p.add_argument("--proxy-api-key",
                   help="API key for proxy. Falls back to PROXY_API_KEY env var.")
    p.add_argument("--gui", action="store_true",
                   help="Launch the PySide6 graphical interface")
    p.add_argument("--web-gui", action="store_true",
                   help="Launch the browser-based web GUI (Flask + SSE)")
    a = p.parse_args()

    if a.gui:
        from translator.gui import launch_gui
        launch_gui()
        return

    if a.web_gui:
        from web_gui.server import launch_web_gui
        launch_web_gui()
        return

    cfg = Config(
        mode=a.mode,
        input_dir=a.input_dir,
        force=a.force,
        batch_size=a.batch_size,
        num_beams=a.num_beams,
        device=_auto_device(a.device),
        resume=a.resume,
        use_tm=a.cache or a.ignore_tm,
        proxy_base_url=a.proxy_base_url or "",
        proxy_api_key=a.proxy_api_key or "",
    )

    try:
        if cfg.mode == "test":
            run_test(cfg)
            return

        if cfg.mode == "benchmark":
            run_benchmark(cfg)
            return

        if cfg.mode == "regression":
            run_regression(cfg)
            return

        files = find_srt_files(cfg.input_dir)
        if not files:
            print(f"No files in '{cfg.input_dir}'")
            return

        for f in files:
            out = output_path_for(f)
            if cfg.mode == "fast":
                has_checkpoint = _checkpoint_path(out).exists() if cfg.resume else False
                if out.exists() and not cfg.force and not has_checkpoint:
                    print(f"  Skip {f.name}")
                    continue
                if cfg.resume and not has_checkpoint:
                    print(f"  No checkpoint for {f.name}, starting fresh")
                translate_fast(f, cfg)

            elif cfg.mode == "polish":
                translate_polish(f, cfg)

            elif cfg.mode == "full":
                has_checkpoint = _checkpoint_path(out).exists() if cfg.resume else False
                if out.exists() and not cfg.force and not has_checkpoint:
                    print(f"  Skip {f.name}")
                else:
                    if cfg.resume and not has_checkpoint:
                        print(f"  No checkpoint for {f.name}, starting fresh")
                    translate_fast(f, cfg)
                translate_polish(f, cfg)

            elif cfg.mode == "llm":
                if out.exists() and not cfg.force:
                    print(f"  Skip {f.name}")
                    continue
                translate_llm(f, cfg)
                translate_polish(f, cfg, polish_model=cfg.polish_model)

    except KeyboardInterrupt:
        print("\nInterrupted by user.", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
