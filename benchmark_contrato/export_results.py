from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_contrato.common import (
    RESULTS,
    REPO_ROOT,
    ensure_results_dirs,
    environment_payload_base,
    load_yaml,
    python_executable,
    repo_commit,
    safe_run,
    scenario_bundle,
    write_json,
)


def write_metadata(scenario: str) -> None:
    manifest, runtime_cfg, experiment_cfg, scenario_cfg, dataset_cfg = scenario_bundle(scenario)
    payload = {
        "method_name": manifest.get("method_name"),
        "method_family": manifest.get("method_family"),
        "paper_year": manifest.get("paper_year"),
        "framework": manifest.get("framework"),
        "scenario": scenario,
        "model": scenario_cfg.get("model"),
        "dataset_key": scenario_cfg.get("dataset_key"),
        "validation": scenario_cfg.get("validation"),
        "dataset_adapter": dataset_cfg.get("adapter"),
        "checkpoint_candidates": scenario_cfg.get("checkpoint_candidates"),
        "repo_commit": repo_commit(),
    }
    write_json(RESULTS / "metadata.json", payload)


def write_environment(scenario: str) -> None:
    payload = environment_payload_base()
    payload.update(
        {
            "scenario": scenario,
            "framework": "pytorch",
            "python_executable": python_executable(),
            "torch_version": safe_run(
                [python_executable(), "-c", "import torch; print(torch.__version__)"]
            ),
            "torchvision_version": safe_run(
                [python_executable(), "-c", "import torchvision; print(torchvision.__version__)"]
            ),
            "cuda_available": safe_run(
                [python_executable(), "-c", "import torch; print(torch.cuda.is_available())"]
            ),
            "cuda_version_torch": safe_run(
                [python_executable(), "-c", "import torch; print(torch.version.cuda)"]
            ),
            "gpu_name": safe_run(
                [
                    python_executable(),
                    "-c",
                    "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)",
                ]
            ),
            "nvidia_smi": safe_run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader",
                ]
            ),
            "repo_root": str(REPO_ROOT),
        }
    )
    write_json(RESULTS / "environment.json", payload)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["metadata", "environment"], required=True)
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    ensure_results_dirs()

    if args.stage == "metadata":
        write_metadata(args.scenario)
    else:
        write_environment(args.scenario)


if __name__ == "__main__":
    main()
