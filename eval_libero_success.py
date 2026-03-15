#!/usr/bin/env python3
"""
Evaluate JiT-opt on LIBERO-10 simulation tasks.
Runs all *.hdf5 tasks under --dataset_path and reports aggregated test mean score.
"""

import argparse
import glob
import json
import os
import sys
from typing import Dict, List

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)
from uva_path_utils import ensure_uva_on_sys_path
ensure_uva_on_sys_path(anchor_file=__file__)

from denoiser import Denoiser
from jit_libero_policy import JitLiberoPolicy
from unified_video_action.env_runner.libero_image_runner import LiberoImageRunner


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate JiT on LIBERO-10")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to JiT checkpoint")
    parser.add_argument("--dataset_path", type=str, required=True, help="Directory that contains LIBERO *.hdf5")
    parser.add_argument("--output_dir", type=str, default="./eval_libero_output")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Model args
    parser.add_argument("--model", type=str, default="JiT-B/16")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--max_condition_frames", type=int, default=2)
    parser.add_argument("--action_dim", type=int, default=10)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--use_ema", action="store_true", default=True)

    # Env runner args (aligned with UVA libero10 defaults)
    parser.add_argument("--n_train", type=int, default=1)
    parser.add_argument("--n_train_vis", type=int, default=0)
    parser.add_argument("--n_test", type=int, default=3)
    parser.add_argument("--n_test_vis", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--n_obs_steps", type=int, default=16)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--test_start_seed", type=int, default=100000)
    parser.add_argument("--fps", type=int, default=10)
    return parser.parse_args()


def _build_shape_meta(img_size: int) -> dict:
    return {
        "image_resolution": img_size,
        "action": {"shape": [10]},
        "obs": {
            "agentview_image": {"shape": [3, img_size, img_size], "type": "rgb"},
        },
    }


def _load_action_stats_from_ckpt(ckpt: dict):
    if "args" in ckpt and hasattr(ckpt["args"], "_action_stats"):
        return ckpt["args"]._action_stats
    return None


def _load_model(args, checkpoint_path: str, device: torch.device):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    action_horizon = args.action_horizon
    if action_horizon is None:
        if "args" in ckpt and hasattr(ckpt["args"], "action_horizon"):
            action_horizon = ckpt["args"].action_horizon
        else:
            action_horizon = 1

    model_args = argparse.Namespace(
        model=args.model,
        img_size=args.img_size,
        max_condition_frames=args.max_condition_frames,
        text_latent_dim=512,
        use_text_condition=False,
        action_dim=args.action_dim,
        action_horizon=action_horizon,
        class_num=1000,
        attn_dropout=0.0,
        proj_dropout=0.0,
        label_drop_prob=0.1,
        sampling_method="heun",
        num_sampling_steps=50,
        cfg=1.5,
        interval_min=0.1,
        interval_max=1.0,
        noise_scale=1.0,
        t_eps=5e-2,
        condition_drop_prob=0.0,
        text_drop_prob=0.0,
        P_mean=-0.8,
        P_std=0.8,
        ema_decay1=0.9999,
        ema_decay2=0.9996,
    )
    denoiser = Denoiser(model_args).to(device).eval()

    if args.use_ema and "model_ema1" in ckpt:
        state_dict = ckpt["model_ema1"]
        print("Using EMA model weights")
    elif "model" in ckpt:
        state_dict = ckpt["model"]
        print("Using regular model weights")
    else:
        state_dict = ckpt
        print("Using raw checkpoint dictionary")

    missing, unexpected = denoiser.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"Missing keys: {missing[:5]}{'...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"Unexpected keys: {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}")

    action_stats = _load_action_stats_from_ckpt(ckpt)
    return denoiser, action_horizon, action_stats


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    denoiser, action_horizon, action_stats = _load_model(args, args.checkpoint, device)
    policy = JitLiberoPolicy(
        denoiser=denoiser,
        n_action_steps=args.n_action_steps,
        max_condition_frames=args.max_condition_frames,
        img_size=args.img_size,
        action_dim=args.action_dim,
        action_horizon=action_horizon,
        action_stats=action_stats,
        device=device,
    )
    policy.eval()

    task_files = sorted(glob.glob(os.path.join(args.dataset_path, "*.hdf5")))
    if not task_files:
        raise ValueError(f"No .hdf5 files found in {args.dataset_path}")

    shape_meta = _build_shape_meta(args.img_size)
    all_logs: Dict[str, float] = {}

    for task_file in task_files:
        task_name = os.path.basename(task_file)
        print(f"\nEvaluating task: {task_name}")
        runner = LiberoImageRunner(
            task_dir=task_file,
            output_dir=args.output_dir,
            dataset_path=args.dataset_path,
            shape_meta=shape_meta,
            n_train=args.n_train,
            n_train_vis=args.n_train_vis,
            n_test=args.n_test,
            n_test_vis=args.n_test_vis,
            test_start_seed=args.test_start_seed,
            max_steps=args.max_steps,
            n_obs_steps=args.n_obs_steps,
            n_action_steps=args.n_action_steps,
            render_obs_key="agentview_image",
            fps=args.fps,
            crf=22,
            past_action=False,
            abs_action=True,
            tqdm_interval_sec=1.0,
            n_envs=None,
        )
        log = runner.run(policy)
        all_logs.update(log)

    test_scores = [v for k, v in all_logs.items() if "test/" in k and "_mean_score" in k]
    train_scores = [v for k, v in all_logs.items() if "train/" in k and "_mean_score" in k]
    result = {
        "checkpoint": args.checkpoint,
        "dataset_path": args.dataset_path,
        "test_mean_score": float(np.mean(test_scores)) if test_scores else None,
        "train_mean_score": float(np.mean(train_scores)) if train_scores else None,
    }

    for k, v in all_logs.items():
        if isinstance(v, (float, int, np.floating)):
            result[k] = float(v)
        else:
            result[k] = str(v)

    out_path = os.path.join(args.output_dir, "eval_libero_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print("\n=== LIBERO Evaluation Result ===")
    print(f"test_mean_score: {result['test_mean_score']}")
    print(f"train_mean_score: {result['train_mean_score']}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
