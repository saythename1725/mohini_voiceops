"""Resolve the app folder and separately supplied capstone dataset folder."""

import os
from pathlib import Path

from dotenv import load_dotenv


APP_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(APP_ROOT / ".env")


def _dataset_root() -> Path:
    configured = os.getenv("VOICEOPS_DATASET_ROOT", "").strip().strip('"').strip("'")
    if configured:
        path = Path(configured).expanduser()
        if not path.is_absolute():
            path = (APP_ROOT / path).resolve()
        return path

    # Development layout: the app is nested beside the supplied dataset.
    for path in (APP_ROOT.parent, APP_ROOT):
        if all((path / name).is_dir() for name in ("kb", "data", "tools")):
            return path
    return APP_ROOT


DATASET_ROOT = _dataset_root()
# Backwards-compatible name used by the evaluator and agent.
ROOT = DATASET_ROOT


def require_dataset_root() -> Path:
    required = ("kb", "data", "tools")
    missing = [name for name in required if not (DATASET_ROOT / name).is_dir()]
    if missing:
        raise FileNotFoundError(
            "Capstone dataset folders were not found. Set VOICEOPS_DATASET_ROOT in .env "
            "to the supplied dataset directory containing kb/, data/, and tools/. "
            f"Missing: {', '.join(missing)} (checked {DATASET_ROOT})."
        )
    return DATASET_ROOT
