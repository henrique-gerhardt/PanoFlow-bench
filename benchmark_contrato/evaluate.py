from __future__ import annotations

import io
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opticalflow.api import init_model
from opticalflow.api.evaluate import (
    Not360Exception,
    validate_flow360,
    validate_flow360_cfe,
    validate_omni,
    validate_omni_cfe,
)
from opticalflow.utils import flow_utils
from opticalflow.utils.flow_utils import convert_360_gt
from opticalflow.utils.utils import InputPadder

from benchmark_contrato.common import (
    RAW_LOGS,
    RESULTS,
    ensure_results_dirs,
    python_executable,
    resolve_path_candidates,
    runtime_change_gpu,
    runtime_device,
    runtime_gpus,
    scenario_bundle,
    shell_join,
    write_json,
)


FLOW360_ALL_RE = re.compile(r"Validation FLow360 \(all\) EPE: ([0-9eE+.-]+)")
OMNI_ALL_RE = re.compile(r"Validation Omni \(all\) EPE: ([0-9eE+.-]+)")


def _null_quality_payload() -> dict[str, Any]:
    return {
        "epe_global": None,
        "epe_polar": None,
        "epe_equatorial": None,
        "epe_by_latitude": None,
        "per_scene_epe": None,
    }


def _stat_bucket() -> dict[str, Any]:
    return {"sum": 0.0, "count": 0}


def _update_stat(bucket: dict[str, Any], values: torch.Tensor) -> None:
    if values.numel() == 0:
        return
    bucket["sum"] += float(values.sum().item())
    bucket["count"] += int(values.numel())


def _mean(bucket: dict[str, Any]) -> float | None:
    if bucket["count"] == 0:
        return None
    return round(bucket["sum"] / bucket["count"], 6)


def _command_value(candidates: list[str] | None, attempts: list[dict[str, Any]]) -> str | None:
    for attempt in attempts:
        if attempt["expanded"]:
            return attempt["expanded"]
    if candidates:
        return candidates[0]
    return None


def build_official_eval_command(
    scenario_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    checkpoint_value: str | None,
    dataset_value: str | None,
) -> list[str]:
    command = [
        python_executable(),
        "./tools/eval.py",
        "--model",
        scenario_cfg["model"],
    ]

    if checkpoint_value:
        command.extend(["--restore_ckpt", checkpoint_value])

    if scenario_cfg.get("use_cfe", False):
        command.append("--CFE")

    validation = scenario_cfg.get("validation", [])
    if validation:
        command.extend(["--validation", *validation])

    if "Flow360" in validation and dataset_value:
        command.extend(["--val_Flow360_root", dataset_value])

    if "Omni" in validation and dataset_value:
        command.extend(["--val_Omni_root", dataset_value])

    if runtime_change_gpu(runtime_cfg):
        command.append("--change_gpu")
        command.extend(["--gpus", *[str(gpu) for gpu in runtime_gpus(runtime_cfg)]])

    command.extend(
        [
            "--DEVICE",
            runtime_device(runtime_cfg),
            "--iters",
            str(scenario_cfg.get("iters", 20)),
            "--eval_iters",
            str(scenario_cfg.get("eval_iters", 12)),
        ]
    )
    return command


def prepare_run_context(scenario: str) -> dict[str, Any]:
    manifest, runtime_cfg, experiment_cfg, scenario_cfg, dataset_cfg = scenario_bundle(scenario)

    checkpoint_path, checkpoint_attempts = resolve_path_candidates(
        scenario_cfg.get("checkpoint_candidates")
    )
    dataset_root, dataset_attempts = resolve_path_candidates(
        dataset_cfg.get("root_candidates")
    )

    checkpoint_value = _command_value(
        scenario_cfg.get("checkpoint_candidates"),
        checkpoint_attempts,
    )
    dataset_value = _command_value(
        dataset_cfg.get("root_candidates"),
        dataset_attempts,
    )

    if scenario == "standardized_efficiency":
        backend = "dummy_inference_smoke_test"
    elif dataset_cfg["adapter"] == "flow360_sequence_npy":
        backend = "flow360_sequence_npy"
    elif dataset_cfg["adapter"] == "omni_official_layout":
        backend = "omni_official_api"
    elif dataset_cfg["adapter"] == "flow360_official_layout":
        backend = "flow360_official_api"
    else:
        raise ValueError(f"Unsupported dataset adapter: {dataset_cfg['adapter']}")

    official_command = build_official_eval_command(
        scenario_cfg=scenario_cfg,
        runtime_cfg=runtime_cfg,
        checkpoint_value=checkpoint_value,
        dataset_value=dataset_value,
    )

    official_cli_supported = dataset_cfg["adapter"] in {
        "flow360_official_layout",
        "omni_official_layout",
    }

    return {
        "manifest": manifest,
        "runtime_cfg": runtime_cfg,
        "experiment_cfg": experiment_cfg,
        "scenario": scenario,
        "scenario_cfg": scenario_cfg,
        "dataset_cfg": dataset_cfg,
        "checkpoint_path": checkpoint_path,
        "checkpoint_attempts": checkpoint_attempts,
        "dataset_root": dataset_root,
        "dataset_attempts": dataset_attempts,
        "official_command": official_command,
        "official_cli_supported": official_cli_supported,
        "execution_backend": backend,
    }


def _require_cuda(runtime_cfg: dict[str, Any]) -> None:
    device = runtime_device(runtime_cfg)
    if device != "cuda":
        raise RuntimeError(
            "O contrato atual foi preparado para execucao CUDA. "
            f"device configurado: {device!r}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA nao esta disponivel neste ambiente. "
            "Execute o contrato em Linux com GPU NVIDIA e NVIDIA Container Toolkit."
        )


def make_model_args(
    *,
    scenario_cfg: dict[str, Any],
    runtime_cfg: dict[str, Any],
    dataset_name: str,
    checkpoint_path: Path | None,
    dataset_root: Path | None,
    train: bool,
) -> SimpleNamespace:
    return SimpleNamespace(
        model=scenario_cfg["model"],
        CFE=scenario_cfg.get("use_cfe", False),
        restore_ckpt=str(checkpoint_path) if checkpoint_path else None,
        iters=int(scenario_cfg.get("iters", 20)),
        eval_iters=int(scenario_cfg.get("eval_iters", 12)),
        train=train,
        eval=True,
        dataset=dataset_name,
        val_Flow360_root=str(dataset_root) if dataset_name == "Flow360" and dataset_root else None,
        val_Omni_root=str(dataset_root) if dataset_name == "Omni" and dataset_root else None,
        validation=scenario_cfg.get("validation", [dataset_name]),
        cvt_gt=bool(scenario_cfg.get("cvt_gt", True)),
        change_gpu=runtime_change_gpu(runtime_cfg),
        gpus=runtime_gpus(runtime_cfg),
        DEVICE=runtime_device(runtime_cfg),
        dropout=0.0,
        mixed_precision=False,
        alternate_corr=False,
        dcn=True,
    )


def instantiate_architecture(
    model_name: str,
    dataset_name: str,
    iters: int = 20,
    eval_iters: int = 12,
) -> torch.nn.Module:
    args = SimpleNamespace(
        model=model_name,
        train=False,
        dataset=dataset_name,
        iters=iters,
        eval_iters=eval_iters,
        dropout=0.0,
        mixed_precision=False,
        alternate_corr=False,
        dcn=True,
    )
    return init_model(args)


def load_model_for_inference(context: dict[str, Any], dataset_name: str) -> tuple[torch.nn.Module, torch.nn.Module]:
    _require_cuda(context["runtime_cfg"])

    checkpoint_path = context["checkpoint_path"]
    if checkpoint_path is None:
        raise FileNotFoundError(
            "Checkpoint nao encontrado. Tentativas: "
            f"{context['checkpoint_attempts']}"
        )

    args = make_model_args(
        scenario_cfg=context["scenario_cfg"],
        runtime_cfg=context["runtime_cfg"],
        dataset_name=dataset_name,
        checkpoint_path=checkpoint_path,
        dataset_root=context["dataset_root"],
        train=True,
    )
    model = init_model(args)
    model.eval()
    model_core = model.module if hasattr(model, "module") else model
    model_core.eval()
    return model, model_core


def _load_rgb_tensor(path: Path) -> torch.Tensor:
    image = np.array(flow_utils.read_gen(str(path))).astype(np.uint8)[..., :3]
    return torch.from_numpy(image).permute(2, 0, 1).float()


def _load_flow_tensor(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        flow = np.load(path).astype(np.float32)
    else:
        flow = np.array(flow_utils.read_gen(str(path))).astype(np.float32)
    return torch.from_numpy(flow).permute(2, 0, 1).float()


def _iter_flow360_sequence_samples(dataset_cfg: dict[str, Any], dataset_root: Path):
    split_root = dataset_root / dataset_cfg.get("split", "test")
    if not split_root.exists():
        raise FileNotFoundError(f"Split root not found: {split_root}")

    scene_dirs = sorted(
        path for path in split_root.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    for scene_dir in scene_dirs:
        frames = sorted((scene_dir / dataset_cfg["image_dir"]).glob(f"*{dataset_cfg['image_ext']}"))
        flows = sorted(
            (scene_dir / dataset_cfg["forward_flow_dir"]).glob(f"*{dataset_cfg['flow_ext']}")
        )
        expected_flows = max(0, len(frames) - 1)
        if expected_flows != len(flows):
            raise ValueError(
                f"Scene {scene_dir.name} has {len(frames)} frames but {len(flows)} flows"
            )
        for index, flow_path in enumerate(flows):
            yield {
                "scene_id": scene_dir.name,
                "pair_index": index,
                "image1": frames[index],
                "image2": frames[index + 1],
                "flow": flow_path,
            }


def _load_flow360_sequence_sample(
    sample: dict[str, Any],
    dataset_cfg: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    image1 = _load_rgb_tensor(sample["image1"])
    image2 = _load_rgb_tensor(sample["image2"])
    flow_gt = _load_flow_tensor(sample["flow"])

    if image1.shape[1:] != flow_gt.shape[1:]:
        raise ValueError(
            "Image/flow resolution mismatch for "
            f"{sample['scene_id']} pair {sample['pair_index']}"
        )

    max_abs_flow = float(dataset_cfg["valid_mask"].get("max_abs_flow", 1000.0))
    valid = torch.isfinite(flow_gt).all(dim=0)
    valid = valid & (flow_gt[0].abs() < max_abs_flow) & (flow_gt[1].abs() < max_abs_flow)
    return image1, image2, flow_gt, valid


def predict_standard(model_core: torch.nn.Module, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    image_pair = torch.stack((image1, image2))
    _, flow_pred = model_core._model(image_pair, test_mode=True)
    return padder.unpad(flow_pred[0]).cpu()


def predict_cfe(model_core: torch.nn.Module, image1: torch.Tensor, image2: torch.Tensor) -> torch.Tensor:
    padder = InputPadder(image1.shape)
    image1, image2 = padder.pad(image1, image2)
    image_pair = torch.stack((image1, image2))

    fmap1, fmap2, cnet1 = model_core._model(image_pair, test_mode=True, gen_fmap=True)

    img_a1 = fmap1[:, :, :, 0 : fmap1.shape[3] // 2]
    img_b1 = fmap1[:, :, :, fmap1.shape[3] // 2 :]
    img_a2 = fmap2[:, :, :, 0 : fmap2.shape[3] // 2]
    img_b2 = fmap2[:, :, :, fmap2.shape[3] // 2 :]

    cnet_a1 = cnet1[:, :, :, 0 : fmap1.shape[3] // 2]
    cnet_b1 = cnet1[:, :, :, fmap1.shape[3] // 2 :]

    pair_b1a1 = torch.stack(
        (
            torch.cat([img_b1, img_a1], dim=3),
            torch.cat([img_b2, img_a2], dim=3),
            torch.cat([cnet_b1, cnet_a1], dim=3),
        )
    )
    pair_a1b1 = torch.stack(
        (
            torch.cat([img_a1, img_b1], dim=3),
            torch.cat([img_a2, img_b2], dim=3),
            torch.cat([cnet_a1, cnet_b1], dim=3),
        )
    )

    _, flow_b1a1 = model_core._model(pair_b1a1, test_mode=True, skip_encode=True)
    _, flow_a1b1 = model_core._model(pair_a1b1, test_mode=True, skip_encode=True)

    flow_a = torch.minimum(
        flow_b1a1[:, :, :, flow_b1a1.shape[3] // 2 :],
        flow_a1b1[:, :, :, 0 : flow_a1b1.shape[3] // 2],
    )
    flow_b = torch.minimum(
        flow_b1a1[:, :, :, 0 : flow_b1a1.shape[3] // 2],
        flow_a1b1[:, :, :, flow_a1b1.shape[3] // 2 :],
    )
    flow_pred = torch.cat([flow_a, flow_b], dim=3)
    flow_pred[:, :, :, flow_pred.shape[3] // 2] = flow_pred[
        :, :, :, (flow_pred.shape[3] // 2) + 1
    ]
    flow_pred[:, :, :, (flow_pred.shape[3] // 2) - 1] = flow_pred[
        :, :, :, (flow_pred.shape[3] // 2) - 2
    ]
    return padder.unpad(flow_pred[0]).cpu()


def run_dummy_inference_smoke_test(context: dict[str, Any]) -> dict[str, Any]:
    _, model_core = load_model_for_inference(context, dataset_name="Flow360")

    height = int(context["scenario_cfg"].get("input_height", 512))
    width = int(context["scenario_cfg"].get("input_width", 1024))
    gpu_id = runtime_gpus(context["runtime_cfg"])[0]
    image1 = torch.rand(1, 3, height, width, device=f"cuda:{gpu_id}") * 255.0
    image2 = torch.rand(1, 3, height, width, device=f"cuda:{gpu_id}") * 255.0

    with torch.no_grad():
        if context["scenario_cfg"].get("use_cfe", False):
            flow_pred = predict_cfe(model_core, image1, image2)
        else:
            flow_pred = predict_standard(model_core, image1, image2)

    smoke_log = RAW_LOGS / "standardized_efficiency_smoke_test.log"
    smoke_log.write_text(
        "\n".join(
            [
                f"scenario={context['scenario']}",
                f"model={context['scenario_cfg']['model']}",
                f"use_cfe={context['scenario_cfg'].get('use_cfe', False)}",
                f"input_shape={[1, 3, height, width]}",
                f"output_shape={list(flow_pred.shape)}",
            ]
        ),
        encoding="utf-8",
    )

    payload = _null_quality_payload()
    payload.update(
        {
            "scenario": context["scenario"],
            "dataset": "dummy_erp_input",
            "dataset_adapter": "dummy_input",
            "pipeline": "official_model_api",
            "execution_backend": context["execution_backend"],
            "official_cli_supported": False,
            "official_cli_executed": False,
            "official_cli_command": shell_join(context["official_command"]),
            "smoke_test": {
                "input_height": height,
                "input_width": width,
                "output_shape": list(flow_pred.shape),
                "raw_log": str(smoke_log),
            },
            "notes": [
                "Cenario de eficiencia usa dummy input ERP para validar carregamento do modelo.",
                "As metricas de latencia e memoria sao geradas em efficiency_metrics.json por profile.py.",
            ],
        }
    )
    return payload


def _latitude_layout(height: int, width: int, runtime_cfg: dict[str, Any]) -> dict[str, Any]:
    regional = runtime_cfg["regional_metrics"]
    latitudes = 90.0 - ((np.arange(height) + 0.5) * 180.0 / height)
    lat_map = np.repeat(latitudes[:, None], width, axis=1)

    bins = []
    edges = regional["latitude_bins_deg"]
    for index in range(len(edges) - 1):
        start = float(edges[index])
        end = float(edges[index + 1])
        if index == len(edges) - 2:
            mask = (lat_map >= start) & (lat_map <= end)
            label = f"[{start}, {end}]"
        else:
            mask = (lat_map >= start) & (lat_map < end)
            label = f"[{start}, {end})"
        bins.append(
            {
                "label": label,
                "min_latitude_deg": start,
                "max_latitude_deg": end,
                "mask": torch.from_numpy(mask),
                "stat": _stat_bucket(),
            }
        )

    equatorial = torch.from_numpy(np.abs(lat_map) <= float(regional["equatorial_band_abs_deg"]))
    polar = torch.from_numpy(np.abs(lat_map) >= float(regional["polar_band_abs_deg"]))
    return {"bins": bins, "equatorial": equatorial, "polar": polar}


def evaluate_flow360_sequence(context: dict[str, Any]) -> dict[str, Any]:
    dataset_root = context["dataset_root"]
    if dataset_root is None:
        raise FileNotFoundError(
            "Dataset Flow360 nao encontrado. Tentativas: "
            f"{context['dataset_attempts']}"
        )

    model, model_core = load_model_for_inference(context, dataset_name="Flow360")
    del model

    gpu_id = runtime_gpus(context["runtime_cfg"])[0]
    latitude_layout: dict[str, Any] | None = None
    global_stat = _stat_bucket()
    polar_stat = _stat_bucket()
    equatorial_stat = _stat_bucket()
    scene_stats: dict[str, dict[str, Any]] = {}
    pair_count = 0
    raw_lines = [
        f"scenario={context['scenario']}",
        f"dataset_root={dataset_root}",
        f"checkpoint={context['checkpoint_path']}",
        f"model={context['scenario_cfg']['model']}",
        f"use_cfe={context['scenario_cfg'].get('use_cfe', False)}",
    ]

    with torch.no_grad():
        for sample in _iter_flow360_sequence_samples(context["dataset_cfg"], dataset_root):
            image1, image2, flow_gt, valid = _load_flow360_sequence_sample(
                sample,
                context["dataset_cfg"],
            )

            if context["scenario_cfg"].get("use_cfe", False) and context["scenario_cfg"].get(
                "cvt_gt", True
            ):
                flow_gt = convert_360_gt(flow_gt)

            if context["scenario_cfg"].get("use_cfe", False):
                if flow_gt[0].max() > flow_gt.shape[2] // 2:
                    raise Not360Exception()

            image1 = image1[None].cuda(gpu_id)
            image2 = image2[None].cuda(gpu_id)

            if context["scenario_cfg"].get("use_cfe", False):
                flow_pred = predict_cfe(model_core, image1, image2)
            else:
                flow_pred = predict_standard(model_core, image1, image2)

            epe_map = torch.sum((flow_pred - flow_gt) ** 2, dim=0).sqrt()
            valid_mask = valid.bool() & torch.isfinite(epe_map)
            valid_epe = epe_map[valid_mask]

            if latitude_layout is None:
                latitude_layout = _latitude_layout(
                    height=epe_map.shape[0],
                    width=epe_map.shape[1],
                    runtime_cfg=context["runtime_cfg"],
                )

            _update_stat(global_stat, valid_epe)

            scene_entry = scene_stats.setdefault(
                sample["scene_id"],
                {"stat": _stat_bucket(), "pair_count": 0},
            )
            _update_stat(scene_entry["stat"], valid_epe)
            scene_entry["pair_count"] += 1

            polar_values = epe_map[valid_mask & latitude_layout["polar"]]
            equatorial_values = epe_map[valid_mask & latitude_layout["equatorial"]]
            _update_stat(polar_stat, polar_values)
            _update_stat(equatorial_stat, equatorial_values)

            for latitude_bin in latitude_layout["bins"]:
                values = epe_map[valid_mask & latitude_bin["mask"]]
                _update_stat(latitude_bin["stat"], values)

            pair_count += 1

    per_scene_epe = [
        {
            "scene_id": scene_id,
            "pair_count": scene_info["pair_count"],
            "epe": _mean(scene_info["stat"]),
        }
        for scene_id, scene_info in sorted(scene_stats.items())
    ]
    for scene_info in per_scene_epe:
        raw_lines.append(
            f"scene={scene_info['scene_id']} pairs={scene_info['pair_count']} epe={scene_info['epe']}"
        )

    raw_log = RAW_LOGS / "flow360_sequence_eval.log"
    raw_log.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    payload = _null_quality_payload()
    payload.update(
        {
            "scenario": context["scenario"],
            "dataset": "Flow360",
            "dataset_root": str(dataset_root),
            "dataset_adapter": context["dataset_cfg"]["adapter"],
            "pipeline": "official_model_api_with_local_dataset_adapter",
            "execution_backend": context["execution_backend"],
            "official_cli_supported": False,
            "official_cli_executed": False,
            "official_cli_command": shell_join(context["official_command"]),
            "official_cli_note": (
                "O CLI tools/eval.py do repositorio espera o layout weather-based original, "
                "mas o dataset encontrado foi fornecido como split/<scene_id>/frames+fflows."
            ),
            "checkpoint_path": str(context["checkpoint_path"]),
            "pair_count": pair_count,
            "scene_count": len(scene_stats),
            "epe_global": _mean(global_stat),
            "epe_polar": _mean(polar_stat),
            "epe_equatorial": _mean(equatorial_stat),
            "epe_by_latitude": [
                {
                    "label": latitude_bin["label"],
                    "min_latitude_deg": latitude_bin["min_latitude_deg"],
                    "max_latitude_deg": latitude_bin["max_latitude_deg"],
                    "epe": _mean(latitude_bin["stat"]),
                    "pixel_count": latitude_bin["stat"]["count"],
                }
                for latitude_bin in (latitude_layout["bins"] if latitude_layout else [])
            ],
            "per_scene_epe": per_scene_epe,
            "raw_log": str(raw_log),
            "notes": [
                "Inferencia realizada com a inicializacao oficial do modelo PanoFlow.",
                "A avaliacao de Flow360 foi adaptada localmente para o layout de dataset encontrado no workspace.",
            ],
        }
    )
    return payload


def _run_validation_with_capture(callback, *args, **kwargs) -> tuple[dict[str, Any], str]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        results = callback(*args, **kwargs)
    return results, buffer.getvalue()


def _evaluate_official_api(
    *,
    context: dict[str, Any],
    dataset_name: str,
    callback,
    regex: re.Pattern[str],
) -> dict[str, Any]:
    dataset_root = context["dataset_root"]
    if dataset_root is None:
        raise FileNotFoundError(
            f"Dataset {dataset_name} nao encontrado. Tentativas: {context['dataset_attempts']}"
        )

    model, model_core = load_model_for_inference(context, dataset_name=dataset_name)
    caller_model = model if runtime_change_gpu(context["runtime_cfg"]) else model_core
    kwargs = {"gpus": runtime_gpus(context["runtime_cfg"])}
    if context["scenario_cfg"].get("use_cfe", False):
        kwargs["cvt_gt"] = bool(context["scenario_cfg"].get("cvt_gt", True))

    results, stdout = _run_validation_with_capture(callback, caller_model, str(dataset_root), **kwargs)

    match = regex.search(stdout)
    raw_log = RAW_LOGS / f"{dataset_name.lower()}_official_api.log"
    raw_log.write_text(stdout, encoding="utf-8")

    payload = _null_quality_payload()
    payload.update(
        {
            "scenario": context["scenario"],
            "dataset": dataset_name,
            "dataset_root": str(dataset_root),
            "dataset_adapter": context["dataset_cfg"]["adapter"],
            "pipeline": "official_model_api",
            "execution_backend": context["execution_backend"],
            "official_cli_supported": context["official_cli_supported"],
            "official_cli_executed": False,
            "official_cli_command": shell_join(context["official_command"]),
            "checkpoint_path": str(context["checkpoint_path"]),
            "metrics_by_subset": results,
            "epe_global": round(float(match.group(1)), 6) if match else None,
            "raw_log": str(raw_log),
            "notes": [
                "Metricas obtidas via funcoes oficiais de avaliacao do repositorio.",
                "Latitudes regionais nao sao calculadas neste backend.",
            ],
        }
    )
    return payload


def write_quality_metrics(context: dict[str, Any]) -> dict[str, Any]:
    ensure_results_dirs()

    if context["scenario"] == "standardized_efficiency":
        payload = run_dummy_inference_smoke_test(context)
    elif context["execution_backend"] == "flow360_sequence_npy":
        payload = evaluate_flow360_sequence(context)
    elif context["execution_backend"] == "omni_official_api":
        callback = validate_omni_cfe if context["scenario_cfg"].get("use_cfe", False) else validate_omni
        payload = _evaluate_official_api(
            context=context,
            dataset_name="Omni",
            callback=callback,
            regex=OMNI_ALL_RE,
        )
    elif context["execution_backend"] == "flow360_official_api":
        callback = (
            validate_flow360_cfe
            if context["scenario_cfg"].get("use_cfe", False)
            else validate_flow360
        )
        payload = _evaluate_official_api(
            context=context,
            dataset_name="Flow360",
            callback=callback,
            regex=FLOW360_ALL_RE,
        )
    else:
        raise ValueError(f"Unsupported execution backend: {context['execution_backend']}")

    write_json(RESULTS / "quality_metrics.json", payload)
    return payload
