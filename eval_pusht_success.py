#!/usr/bin/env python3
"""
Evaluate JiT-opt action policy on PushT environment.
Computes success rate (mean_score) by running policy in simulation.
References UVA's PushTImageRunner and eval_sim.py.
"""

import argparse
import os
import sys
import pickle
import json

import torch
import numpy as np

# Add paths for imports
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
UVA_ROOT = os.path.join(SCRIPT_DIR, "..", "unified_video_action")
sys.path.insert(0, UVA_ROOT)

from denoiser import Denoiser
from jit_push_policy import JitPushTPolicy
from unified_video_action.env_runner.pusht_image_runner import PushTImageRunner


def get_args():
    parser = argparse.ArgumentParser(description="Evaluate JiT on PushT for success rate")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to JiT checkpoint")
    parser.add_argument("--output_dir", type=str, default="./eval_pusht_output", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")

    # Model args (must match training)
    parser.add_argument("--model", type=str, default="JiT-B/16")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--max_condition_frames", type=int, default=2)
    parser.add_argument("--action_dim", type=int, default=2, help="PushT action dim = 2")

    # Normalizer
    parser.add_argument("--normalizer_path", type=str, default=None,
                        help="Path to normalizer.pkl (from UVA pushT training). If None, uses default [-1,1]->[0,512]")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="Path to pushT zarr dataset. If provided with --normalizer_path=None, fits normalizer from data")

    # Wandb (like UVA)
    parser.add_argument("--wandb", action="store_true",
                        help="Log to wandb including rollout videos")
    parser.add_argument("--wandb_project", type=str, default="jit-pusht",
                        help="Wandb project name")
    parser.add_argument("--wandb_run_name", type=str, default=None,
                        help="Wandb run name (default: checkpoint basename)")

    # Env / eval config (aligned with UVA pusht.yaml)
    parser.add_argument("--n_test", type=int, default=50, help="Number of test episodes")
    parser.add_argument("--n_train", type=int, default=6, help="Number of train episodes (for logging)")
    parser.add_argument("--n_test_vis", type=int, default=None,
                        help="Number of test episodes to record video (default: 4 with --wandb, else 0)")
    parser.add_argument("--n_train_vis", type=int, default=None,
                        help="Number of train episodes to record video (default: 2 with --wandb, else 0)")
    parser.add_argument("--max_steps", type=int, default=300)
    parser.add_argument("--n_obs_steps", type=int, default=16)
    parser.add_argument("--n_action_steps", type=int, default=8)
    parser.add_argument("--test_start_seed", type=int, default=100000)
    parser.add_argument("--train_start_seed", type=int, default=0)
    parser.add_argument("--fix_goal", action="store_true", default=True)
    parser.add_argument("--legacy_test", action="store_true", default=True)
    parser.add_argument("--use_ema", action="store_true", default=True, help="Use EMA model for evaluation")

    return parser.parse_args()


def load_normalizer_from_dataset(dataset_path):
    """Fit normalizer from PushT zarr dataset (same as UVA)."""
    from unified_video_action.dataset.pusht_image_dataset import PushTImageDataset
    from unified_video_action.model.common.normalizer import LinearNormalizer

    dataset = PushTImageDataset(
        dataset_path=dataset_path,
        horizon=1,
        val_ratio=0,
    )
    return dataset.get_normalizer(mode="limits")


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Video recording defaults (like UVA: n_test_vis=4, n_train_vis=2)
    if args.n_test_vis is None:
        args.n_test_vis = 4 if args.wandb else 0
    if args.n_train_vis is None:
        args.n_train_vis = 2 if args.wandb else 0

    # Init wandb for video logging
    if args.wandb:
        import wandb
        run_name = args.wandb_run_name or os.path.splitext(os.path.basename(args.checkpoint))[0]
        wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    # Build model args for Denoiser
    model_args = argparse.Namespace(
        model=args.model,
        img_size=args.img_size,
        max_condition_frames=args.max_condition_frames,
        text_latent_dim=512,
        use_text_condition=False,
        action_dim=args.action_dim,
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

    denoiser = Denoiser(model_args)
    denoiser.to(device)
    denoiser.eval()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model" in ckpt:
        sd = ckpt["model"]
    else:
        sd = ckpt

    if args.use_ema and "model_ema1" in ckpt:
        sd = ckpt["model_ema1"]
        if any(k.startswith("net.") for k in sd.keys()):
            sd = {k.replace("net.", ""): v for k, v in sd.items()}
        print("Loaded EMA model")
    else:
        if any(k.startswith("net.") for k in sd.keys()):
            sd = {k.replace("net.", ""): v for k, v in sd.items()}
        print("Loaded regular model")

    denoiser.net.load_state_dict(sd, strict=False)

    # Action stats for denormalization (preferred over UVA normalizer)
    action_stats = None
    normalizer = None
    normalizer_type = "none"

    if args.dataset_path and os.path.exists(args.dataset_path):
        # Compute action stats directly from the zarr dataset (matches training normalization)
        import zarr as _zarr
        ds_path = args.dataset_path
        if not ds_path.endswith('.zarr'):
            candidates = [f for f in os.listdir(ds_path) if f.endswith('.zarr')]
            if candidates:
                ds_path = os.path.join(ds_path, candidates[0])
        if os.path.exists(ds_path):
            zs = _zarr.open(ds_path, mode='r')
            if 'data' in zs and 'action' in zs['data']:
                all_actions = zs['data']['action'][:].astype(np.float32)
                action_stats = {
                    'min': all_actions.min(axis=0),
                    'max': all_actions.max(axis=0),
                }
                print(f"Computed action stats from dataset: min={action_stats['min']}, max={action_stats['max']}")

    if action_stats is None:
        # Fallback to UVA normalizer
        if args.normalizer_path and os.path.exists(args.normalizer_path):
            with open(args.normalizer_path, "rb") as f:
                normalizer = pickle.load(f)
            normalizer_type = "all"
            print(f"Loaded normalizer from {args.normalizer_path}")
        elif args.dataset_path and os.path.exists(args.dataset_path):
            normalizer = load_normalizer_from_dataset(args.dataset_path)
            normalizer_type = "all"
            print(f"Fitted normalizer from dataset {args.dataset_path}")

    policy = JitPushTPolicy(
        denoiser=denoiser,
        n_action_steps=args.n_action_steps,
        max_condition_frames=args.max_condition_frames,
        img_size=args.img_size,
        action_dim=args.action_dim,
        normalizer=normalizer,
        normalizer_type=normalizer_type,
        action_stats=action_stats,
        device=device,
    )
    policy.eval()

    # Create PushT env runner (from UVA)
    env_runner = PushTImageRunner(
        output_dir=args.output_dir,
        n_train=args.n_train,
        n_train_vis=args.n_train_vis,
        train_start_seed=args.train_start_seed,
        n_test=args.n_test,
        n_test_vis=args.n_test_vis,
        legacy_test=args.legacy_test,
        test_start_seed=args.test_start_seed,
        max_steps=args.max_steps,
        n_obs_steps=args.n_obs_steps,
        n_action_steps=args.n_action_steps,
        fps=10,
        render_size=96,
        past_action=False,
        fix_goal=args.fix_goal,
    )

    print("Running PushT evaluation...")
    runner_log = env_runner.run(policy)

    # Log to wandb (including videos, like UVA)
    if args.wandb:
        import wandb
        wandb.log(runner_log)

    # Extract scores
    test_mean = runner_log.get("test/mean_score", None)
    train_mean = runner_log.get("train/mean_score", None)

    result = {
        "test_mean_score": float(test_mean) if test_mean is not None else None,
        "train_mean_score": float(train_mean) if train_mean is not None else None,
        "checkpoint": args.checkpoint,
    }
    for k, v in runner_log.items():
        if k in result or "video" in k.lower():
            continue
        if isinstance(v, (int, float, np.floating)):
            result[k] = float(v)
        elif hasattr(v, "_path"):  # wandb.Video
            result[k] = str(getattr(v, "_path", v))
        else:
            result[k] = str(v) if v is not None else None

    out_path = os.path.join(args.output_dir, "eval_pusht_result.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print("\n=== PushT Evaluation Result ===")
    print(f"Test mean score (success rate): {test_mean}")
    print(f"Train mean score: {train_mean}")
    print(f"Result saved to {out_path}")

    return result


if __name__ == "__main__":
    main()
