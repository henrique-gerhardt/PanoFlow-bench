from __future__ import annotations

import json
import os
import platform
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
CONFIG_DIR = ROOT / "config"
RESULTS = ROOT / "results"
RAW_LOGS = RESULTS / "raw_logs"
OPTIONAL_PREDICTIONS = RESULTS / "optional_predictions"
ENV_PLACEHOLDER = re.compile(r"^\$\{([A-Z0-9_]+)\}$")


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_results_dirs() -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    RAW_LOGS.mkdir(parents=True, exist_ok=True)
    OPTIONAL_PREDICTIONS.mkdir(parents=True, exist_ok=True)


def load_configs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest = load_yaml(ROOT / "manifest.yaml")
    datasets_cfg = load_yaml(CONFIG_DIR / "datasets.yaml")
    runtime_cfg = load_yaml(CONFIG_DIR / "runtime.yaml")
    experiment_cfg = load_yaml(CONFIG_DIR / "experiment.yaml")
    return manifest, datasets_cfg, runtime_cfg, experiment_cfg


def scenario_bundle(
    scenario: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    manifest, datasets_cfg, runtime_cfg, experiment_cfg = load_configs()
    active_scenario = scenario or experiment_cfg.get("scenario")
    if active_scenario not in experiment_cfg:
        raise KeyError(f"Scenario '{active_scenario}' not found in experiment.yaml")
    scenario_cfg = experiment_cfg[active_scenario]
    dataset_key = scenario_cfg["dataset_key"]
    datasets_map = datasets_cfg["datasets"]
    if dataset_key not in datasets_map:
        raise KeyError(f"Dataset key '{dataset_key}' not found in datasets.yaml")
    dataset_cfg = datasets_map[dataset_key]
    return manifest, runtime_cfg, experiment_cfg, scenario_cfg, dataset_cfg


def _expand_candidate(candidate: str | None) -> str | None:
    if candidate is None:
        return None
    match = ENV_PLACEHOLDER.match(candidate)
    if match:
        return os.environ.get(match.group(1))
    return os.path.expanduser(candidate)


def _candidate_to_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def resolve_path_candidates(candidates: list[str] | None) -> tuple[Path | None, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    if not candidates:
        return None, attempts

    for candidate in candidates:
        expanded = _expand_candidate(candidate)
        if not expanded:
            attempts.append(
                {
                    "candidate": candidate,
                    "expanded": None,
                    "exists": False,
                }
            )
            continue

        path = _candidate_to_path(expanded)
        exists = path.exists()
        attempts.append(
            {
                "candidate": candidate,
                "expanded": str(path),
                "exists": exists,
            }
        )
        if exists:
            return path, attempts

    return None, attempts


def shell_join(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def safe_run(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        output = subprocess.check_output(
            command,
            cwd=str(cwd) if cwd else None,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception:
        return None
    return output.strip()


def repo_commit() -> str | None:
    return safe_run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT)


def python_executable() -> str:
    return os.environ.get("PYTHON", "python")


def runtime_device(runtime_cfg: dict[str, Any]) -> str:
    return runtime_cfg["runtime"].get("device", "cuda")


def runtime_gpus(runtime_cfg: dict[str, Any]) -> list[int]:
    gpus = runtime_cfg["runtime"].get("gpus", [0])
    return [int(gpu) for gpu in gpus]


def runtime_change_gpu(runtime_cfg: dict[str, Any]) -> bool:
    gpus = runtime_gpus(runtime_cfg)
    return bool(runtime_cfg["runtime"].get("change_gpu", False) or gpus != [0])


def environment_payload_base() -> dict[str, Any]:
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "repo_commit": repo_commit(),
    }
