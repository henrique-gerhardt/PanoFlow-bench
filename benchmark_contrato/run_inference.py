from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_contrato.common import RESULTS, ensure_results_dirs, write_json
from benchmark_contrato.evaluate import prepare_run_context, write_quality_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    ensure_results_dirs()
    context = prepare_run_context(args.scenario)

    run_config = {
        "scenario": context["scenario"],
        "execution_backend": context["execution_backend"],
        "official_cli_supported": context["official_cli_supported"],
        "official_command": context["official_command"],
        "runtime": context["runtime_cfg"]["runtime"],
        "scenario_config": context["scenario_cfg"],
        "dataset_config": context["dataset_cfg"],
        "resolved_paths": {
            "checkpoint_path": str(context["checkpoint_path"]) if context["checkpoint_path"] else None,
            "dataset_root": str(context["dataset_root"]) if context["dataset_root"] else None,
        },
        "path_resolution_attempts": {
            "checkpoint": context["checkpoint_attempts"],
            "dataset": context["dataset_attempts"],
        },
    }
    write_json(RESULTS / "run_config.json", run_config)
    write_quality_metrics(context)


if __name__ == "__main__":
    main()
