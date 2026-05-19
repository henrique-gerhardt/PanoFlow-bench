from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
except Exception:
    torch = None

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark_contrato.common import RESULTS, ensure_results_dirs, write_json
from benchmark_contrato.evaluate import (
    instantiate_architecture,
    load_model_for_inference,
    predict_cfe,
    predict_standard,
    prepare_run_context,
)


def _null_efficiency_payload() -> dict[str, float | int | str | None]:
    return {
        "parameters": None,
        "checkpoint_size_mb": None,
        "flops_g": None,
        "latency_mean_ms": None,
        "latency_median_ms": None,
        "latency_p95_ms": None,
        "max_gpu_memory_mb": None,
        "fps": None,
        "notes": "Metricas nulas ate existir ambiente valido para profiling.",
    }


def _checkpoint_size_mb(path: Path | None) -> float | None:
    if path is None or not path.exists():
        return None
    return round(path.stat().st_size / (1024 * 1024), 3)


def _parameter_count(model_name: str, dataset_name: str, iters: int, eval_iters: int) -> int | None:
    try:
        model = instantiate_architecture(
            model_name=model_name,
            dataset_name=dataset_name,
            iters=iters,
            eval_iters=eval_iters,
        )
    except Exception:
        return None
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _profile_dummy_inference(context: dict[str, object]) -> dict[str, float | None]:
    if torch is None:
        raise RuntimeError("torch nao esta disponivel neste ambiente")

    runtime_cfg = context["runtime_cfg"]
    scenario_cfg = context["scenario_cfg"]
    gpus = runtime_cfg["runtime"].get("gpus", [0])
    gpu_id = int(gpus[0])
    warmup_runs = int(runtime_cfg["runtime"].get("warmup_runs", 10))
    measured_runs = int(runtime_cfg["runtime"].get("measured_runs", 30))
    height = int(scenario_cfg.get("input_height", 512))
    width = int(scenario_cfg.get("input_width", 1024))

    _, model_core = load_model_for_inference(context, dataset_name="Flow360")
    image1 = torch.rand(1, 3, height, width, device=f"cuda:{gpu_id}") * 255.0
    image2 = torch.rand(1, 3, height, width, device=f"cuda:{gpu_id}") * 255.0
    inference_fn = predict_cfe if scenario_cfg.get("use_cfe", False) else predict_standard

    latencies = []
    torch.cuda.reset_peak_memory_stats(gpu_id)

    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = inference_fn(model_core, image1, image2)
            torch.cuda.synchronize(gpu_id)

        for _ in range(measured_runs):
            torch.cuda.synchronize(gpu_id)
            start = time.perf_counter()
            _ = inference_fn(model_core, image1, image2)
            torch.cuda.synchronize(gpu_id)
            latencies.append((time.perf_counter() - start) * 1000.0)

    latencies_array = np.array(latencies, dtype=np.float64)
    return {
        "latency_mean_ms": round(float(latencies_array.mean()), 3),
        "latency_median_ms": round(float(np.median(latencies_array)), 3),
        "latency_p95_ms": round(float(np.percentile(latencies_array, 95)), 3),
        "max_gpu_memory_mb": round(
            float(torch.cuda.max_memory_allocated(gpu_id) / (1024 * 1024)),
            3,
        ),
        "fps": round(float(1000.0 / latencies_array.mean()), 3),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", required=True)
    args = parser.parse_args()

    ensure_results_dirs()
    context = prepare_run_context(args.scenario)

    payload = _null_efficiency_payload()
    payload["checkpoint_size_mb"] = _checkpoint_size_mb(context["checkpoint_path"])
    payload["parameters"] = _parameter_count(
        model_name=context["scenario_cfg"]["model"],
        dataset_name="Flow360",
        iters=int(context["scenario_cfg"].get("iters", 20)),
        eval_iters=int(context["scenario_cfg"].get("eval_iters", 12)),
    )

    notes = []
    if context["scenario"] == "standardized_efficiency":
        if torch is None:
            notes.append("torch indisponivel no ambiente atual.")
        elif not torch.cuda.is_available():
            notes.append("CUDA indisponivel; profiling real depende do host Linux com GPU.")
        elif context["checkpoint_path"] is None:
            notes.append("Checkpoint nao resolvido; profiling real requer um checkpoint valido.")
        else:
            metrics = _profile_dummy_inference(context)
            payload.update(metrics)
            notes.append(
                "Latencia e memoria medidas com dummy input ERP 1024x512 usando a logica oficial de inferencia."
            )

    if payload["parameters"] is None:
        notes.append("Falha ao instanciar a arquitetura para contar parametros.")

    if payload["flops_g"] is None:
        notes.append("FLOPs permanecem null por nao haver tracing confiavel nesta iteracao.")

    payload["notes"] = " ".join(notes) if notes else payload["notes"]
    write_json(RESULTS / "efficiency_metrics.json", payload)


if __name__ == "__main__":
    main()
