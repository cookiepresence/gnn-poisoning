import json
from pathlib import Path
import random
from typing import Any

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def log_line(path: Path, payload: dict[str, Any]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")

def create_rundir(run_dir: str, run_name: str):
    run_dir: Path = Path(run_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_dir = run_dir / "model"
    model_dir.mkdir(exist_ok=True)

    return run_dir

def write_summary(run_dir: Path, payload: dict[str, Any]) -> None:
    write_json(run_dir / "summary.json", payload)

def write_metrics(run_dir: Path, payload: dict[str, Any]) -> None:
    log_line(run_dir / "metrics.jsonl", payload)
