import argparse
import json
import os
import pathlib
import time
from typing import Callable, Dict, List

import hydra
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from denoiser import Denoiser
from jit_push_policy import JitPushTPolicy
from main_jit import get_args_parser


def _build_parser() -> argparse.ArgumentParser:
    base = get_args_parser()
    parser = argparse.ArgumentParser(
        "Policy-level inference benchmark",
        parents=[base],
        add_help=True,
    )
    parser.add_argument(
        "--model_family",
        type=str,
        default="jit",
        choices=["jit", "uva"],
        help="Which policy implementation to benchmark.",
    )
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dataset_type", type=str, default="pusht", choices=["pusht"])
    parser.add_argument("--action_dim", type=int, default=2)
    parser.add_argument("--action_horizon", type=int, default=1)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument(
        "--max_condition_frames",
        type=int,
        default=4,
        help="Requested history frame count. UVA requires at least 4.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--bench_mode",
        type=str,
        default="action_step",
        choices=["action_step"],
        help="Strategy-level benchmark: end-to-end policy.predict_action.",
    )
    parser.add_argument("--bench_warmup", type=int, default=30)
    parser.add_argument("--bench_iters", type=int, default=200)
    parser.add_argument("--bench_dtype", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--bench_no_amp", action="store_true")
    parser.add_argument("--bench_report_json", type=str, default="")
    parser.add_argument(
        "--resume",
        type=str,
        default="",
        help="Optional checkpoint path to load model/policy weights.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--normalizer_type",
        type=str,
        default="none",
        choices=["none", "all"],
        help="For synthetic benchmark input, `none` is recommended.",
    )
    parser.add_argument(
        "--policy_obs_frames",
        type=int,
        default=4,
        help="Observation horizon fed into policy.predict_action.",
    )
    parser.add_argument(
        "--policy_action_steps",
        type=int,
        default=1,
        help="Number of rollout actions requested from policy output.",
    )
    return parser


def _get_amp_dtype(dtype_str: str) -> torch.dtype:
    if dtype_str == "fp16":
        return torch.float16
    if dtype_str == "bf16":
        return torch.bfloat16
    return torch.float32


def _time_function(fn: Callable[[], None], iters: int, warmup: int, use_cuda: bool) -> List[float]:
    for _ in range(warmup):
        fn()
    if use_cuda:
        torch.cuda.synchronize()

    latencies_ms = []
    for _ in range(iters):
        if use_cuda:
            torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        if use_cuda:
            torch.cuda.synchronize()
        end = time.perf_counter()
        latencies_ms.append((end - start) * 1000.0)
    return latencies_ms


def _load_uva_cfg(dataset_type: str) -> OmegaConf:
    if dataset_type != "pusht":
        raise NotImplementedError(f"Unsupported dataset_type: {dataset_type}")

    config_dir = pathlib.Path(__file__).parent.joinpath("unified_video_action", "config")
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.absolute())):
        cfg = compose(config_name="uva_pusht")
    OmegaConf.resolve(cfg)
    return cfg


def _load_policy_from_cfg(cfg: OmegaConf, device: torch.device):
    model = hydra.utils.instantiate(
        cfg.model.policy,
        task_name=cfg.task.name,
        task_modes=cfg.task.task_modes,
        normalizer_type=cfg.task.dataset.normalizer_type,
        language_emb_model=cfg.task.dataset.language_emb_model,
    ).to(device)
    model.eval()
    return model


def _load_uva_checkpoint_if_any(model, resume_path: str) -> str:
    if not resume_path:
        return ""
    if not os.path.isfile(resume_path):
        raise FileNotFoundError(f"Checkpoint not found: {resume_path}")

    payload = torch.load(resume_path, map_location="cpu")
    if isinstance(payload, dict) and "state_dicts" in payload:
        if "model" in payload["state_dicts"]:
            state_dict = payload["state_dicts"]["model"]
        elif "ema_model" in payload["state_dicts"]:
            state_dict = payload["state_dicts"]["ema_model"]
        else:
            raise KeyError("No `model` or `ema_model` in checkpoint state_dicts.")
    else:
        state_dict = payload

    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            remapped[k.replace("module.", "", 1)] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[benchmark] loaded checkpoint: {resume_path}")
    if missing:
        print(f"[benchmark] missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[benchmark] unexpected keys (first 5): {unexpected[:5]}")
    return resume_path


def _resolve_jit_resume_path(resume_arg: str) -> str:
    if not resume_arg:
        return ""
    if os.path.isfile(resume_arg):
        return resume_arg
    if os.path.isdir(resume_arg):
        candidate = os.path.join(resume_arg, "checkpoint-last.pth")
        if os.path.exists(candidate):
            return candidate
    return ""


def _load_jit_checkpoint_if_any(model: Denoiser, resume_arg: str) -> str:
    ckpt_path = _resolve_jit_resume_path(resume_arg)
    if not ckpt_path:
        return ""

    payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
    remapped = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            remapped[k.replace("module.", "", 1)] = v
        else:
            remapped[k] = v
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    print(f"[benchmark] loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"[benchmark] missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[benchmark] unexpected keys (first 5): {unexpected[:5]}")
    return ckpt_path


def main():
    args = _build_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    bsz = int(args.batch_size)
    img = int(args.img_size)
    requested_obs_frames = int(args.policy_obs_frames)
    t_obs = requested_obs_frames

    obs_dict = {
        "image": torch.rand(bsz, t_obs, 3, img, img, device=device, dtype=torch.float32),
        "agent_pos": torch.rand(bsz, t_obs, int(args.action_dim), device=device, dtype=torch.float32),
    }

    amp_dtype = _get_amp_dtype(args.bench_dtype)
    use_amp = (not args.bench_no_amp) and (amp_dtype != torch.float32) and device.type == "cuda"

    if args.model_family == "uva":
        cfg = _load_uva_cfg(args.dataset_type)
        cfg.model.policy.action_model_params.predict_action = True
        cfg.model.policy.selected_training_mode = "policy_model"
        cfg.task.dataset.normalizer_type = args.normalizer_type
        cfg.model.policy.n_action_steps = int(args.policy_action_steps)

        model = _load_policy_from_cfg(cfg, device)
        ckpt_path = _load_uva_checkpoint_if_any(model, args.resume)

        # UVA policy expects at least 4 history frames.
        if t_obs < 4:
            t_obs = 4
            obs_dict["image"] = torch.rand(
                bsz, t_obs, 3, img, img, device=device, dtype=torch.float32
            )
            obs_dict["agent_pos"] = torch.rand(
                bsz, t_obs, int(args.action_dim), device=device, dtype=torch.float32
            )
            print("[benchmark] policy_obs_frames < 4 requested; clamped to 4 for UVA.")
    else:
        denoiser = Denoiser(args).to(device)
        denoiser.eval()
        ckpt_path = _load_jit_checkpoint_if_any(denoiser, args.resume)
        model = JitPushTPolicy(
            denoiser=denoiser,
            n_action_steps=int(args.policy_action_steps),
            max_condition_frames=int(args.max_condition_frames),
            img_size=int(args.img_size),
            action_dim=int(args.action_dim),
            action_horizon=int(args.action_horizon),
            normalizer=None,
            normalizer_type=args.normalizer_type,
            action_stats=None,
            device=device,
        )
        model.eval()

    def run_once():
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                _ = model.predict_action(obs_dict)

    latencies_ms = _time_function(
        run_once,
        iters=int(args.bench_iters),
        warmup=int(args.bench_warmup),
        use_cuda=(device.type == "cuda"),
    )

    lat_arr = np.array(latencies_ms, dtype=np.float64)
    avg_ms = float(lat_arr.mean())
    std_ms = float(lat_arr.std())
    p50_ms = float(np.percentile(lat_arr, 50))
    p90_ms = float(np.percentile(lat_arr, 90))
    p95_ms = float(np.percentile(lat_arr, 95))
    throughput = float(bsz * 1000.0 / avg_ms)

    result: Dict[str, object] = {
        "model_family": args.model_family,
        "bench_mode": "action_step",
        "mode_note": "strategy-level end-to-end predict_action",
        "device": args.device,
        "dtype": args.bench_dtype if use_amp else "fp32",
        "dataset_type": args.dataset_type,
        "batch_size": bsz,
        "img_size": img,
        "max_condition_frames": int(args.max_condition_frames),
        "effective_obs_frames": t_obs,
        "action_dim": int(args.action_dim),
        "action_horizon": int(args.action_horizon),
        "policy_action_steps": int(args.policy_action_steps),
        "normalizer_type": args.normalizer_type,
        "checkpoint": ckpt_path if ckpt_path else "none",
        "iters": int(args.bench_iters),
        "warmup": int(args.bench_warmup),
        "avg_ms": avg_ms,
        "std_ms": std_ms,
        "p50_ms": p50_ms,
        "p90_ms": p90_ms,
        "p95_ms": p95_ms,
        "throughput_samples_per_s": throughput,
    }

    print("=" * 72)
    for key, val in result.items():
        print(f"{key}: {val}")
    print("=" * 72)

    if args.bench_report_json:
        with open(args.bench_report_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[benchmark] report saved to: {args.bench_report_json}")


if __name__ == "__main__":
    main()
