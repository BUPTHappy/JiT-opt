import argparse
import os
import time
from typing import Callable, Dict, List

import numpy as np
import torch

from denoiser import Denoiser
from main_jit import get_args_parser


def _build_parser() -> argparse.ArgumentParser:
    base = get_args_parser()
    parser = argparse.ArgumentParser(
        "JiT inference benchmark",
        parents=[base],
        add_help=True,
    )
    parser.add_argument(
        "--bench_mode",
        type=str,
        default="action_step",
        choices=["action_step", "action_sampling", "video_sampling"],
        help="Benchmark mode: one action denoise step / full action sampling / full video sampling",
    )
    parser.add_argument("--bench_iters", type=int, default=100, help="Number of measured iterations")
    parser.add_argument("--bench_warmup", type=int, default=20, help="Number of warmup iterations")
    parser.add_argument(
        "--bench_dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Compute dtype for inference benchmark",
    )
    parser.add_argument(
        "--bench_no_amp",
        action="store_true",
        help="Disable autocast and run with fp32 compute",
    )
    parser.add_argument(
        "--bench_report_json",
        type=str,
        default="",
        help="Optional path to save benchmark results json",
    )
    return parser


def _resolve_resume_path(resume_arg: str) -> str:
    if not resume_arg:
        return ""
    if os.path.isfile(resume_arg):
        return resume_arg
    if os.path.isdir(resume_arg):
        candidate = os.path.join(resume_arg, "checkpoint-last.pth")
        if os.path.exists(candidate):
            return candidate
    return ""


def _load_checkpoint_if_any(model: Denoiser, resume_arg: str) -> str:
    ckpt_path = _resolve_resume_path(resume_arg)
    if not ckpt_path:
        return ""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[benchmark] loaded checkpoint: {ckpt_path}")
    if missing:
        print(f"[benchmark] missing keys (first 5): {missing[:5]}")
    if unexpected:
        print(f"[benchmark] unexpected keys (first 5): {unexpected[:5]}")
    return ckpt_path


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


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.device != "cuda" and args.device != "cpu":
        raise ValueError(f"--device must be cuda or cpu for this script, got {args.device}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)
    model = Denoiser(args).to(device)
    model.eval()
    ckpt_path = _load_checkpoint_if_any(model, args.resume)

    bsz = int(args.batch_size)
    k = int(args.max_condition_frames)
    img = int(args.img_size)
    action_dim_total = int(args.action_dim) * int(args.action_horizon)

    # Use [-1, 1] synthetic input matching training/inference normalization.
    condition_frames = torch.rand(bsz, k, 3, img, img, device=device) * 2.0 - 1.0
    target_frame = torch.rand(bsz, 3, img, img, device=device) * 2.0 - 1.0
    noisy_action = torch.randn(bsz, action_dim_total, device=device)
    action_t = torch.rand(bsz, device=device)

    amp_dtype = _get_amp_dtype(args.bench_dtype)
    use_amp = (not args.bench_no_amp) and (amp_dtype != torch.float32) and device.type == "cuda"

    if args.bench_mode == "action_step":
        def run_once():
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    _img_pred, _action_pred = model.net(
                        target_frame,
                        action_t,
                        y=None,
                        condition_frames=condition_frames,
                        text_latents=None,
                        return_action=True,
                        noisy_action=noisy_action,
                        action_t=action_t,
                    )
    elif args.bench_mode == "action_sampling":
        steps = int(args.num_sampling_steps)

        def run_once():
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    _ = model.generate_action(
                        target_frame=target_frame,
                        condition_frames=condition_frames,
                        steps=steps,
                    )
    else:
        steps = int(args.num_sampling_steps)

        def run_once():
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    _ = model.generate(condition_frames=condition_frames)

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
        "bench_mode": args.bench_mode,
        "device": args.device,
        "dtype": args.bench_dtype if use_amp else "fp32",
        "batch_size": bsz,
        "img_size": img,
        "max_condition_frames": k,
        "action_dim": int(args.action_dim),
        "action_horizon": int(args.action_horizon),
        "num_sampling_steps": int(args.num_sampling_steps),
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

    if args.bench_mode in ("action_sampling", "video_sampling"):
        per_step_ms = avg_ms / max(1, int(args.num_sampling_steps))
        result["avg_ms_per_sampling_step"] = per_step_ms
        result["sampling_steps_per_s"] = float(1000.0 / per_step_ms)

    print("=" * 72)
    for k_, v_ in result.items():
        print(f"{k_}: {v_}")
    print("=" * 72)

    if args.bench_report_json:
        import json
        with open(args.bench_report_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"[benchmark] report saved to: {args.bench_report_json}")


if __name__ == "__main__":
    main()
