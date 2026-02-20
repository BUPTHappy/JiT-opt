"""
PushT video dataset for JiT-opt.
Outputs (condition_frames, target_frame, action) compatible with JiT training.
PushT zarr format: data/img, data/state, data/action, meta/episode_ends
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import zarr


class PushTVideoDataset(Dataset):
    """PushT dataset for JiT video-to-action training."""

    def __init__(
        self,
        dataset_path: str,
        max_condition_frames: int = 2,
        image_size: int = 256,
        split: str = "train",
        val_ratio: float = 0.02,
        seed: int = 42,
        normalize_action: bool = True,
        **kwargs
    ):
        dataset_path = os.path.expanduser(dataset_path)
        dataset_path = os.path.normpath(os.path.abspath(dataset_path))

        if not os.path.exists(dataset_path):
            raise ValueError(f"dataset_path does not exist: {dataset_path}")

        # Accept: full path to .zarr, or dir containing .zarr, or path without .zarr suffix
        if dataset_path.endswith(".zarr") and os.path.exists(dataset_path):
            zarr_path = dataset_path
        elif os.path.isdir(dataset_path):
            candidates = [f for f in os.listdir(dataset_path) if f.endswith(".zarr")]
            if not candidates:
                raise ValueError(f"No .zarr found in {dataset_path}")
            zarr_path = os.path.join(dataset_path, candidates[0])
        elif os.path.exists(dataset_path + ".zarr"):
            zarr_path = dataset_path + ".zarr"
        else:
            raise ValueError(f"dataset_path not found: {dataset_path} (tried + .zarr)")

        self.zarr_path = zarr_path
        self.max_condition_frames = max_condition_frames
        self.image_size = image_size
        self.split = split
        self.normalize_action = normalize_action

        zarr_store = zarr.open(zarr_path, mode="r")
        if "data" not in zarr_store or "img" not in zarr_store["data"]:
            raise ValueError(f"PushT zarr must have data/img. Found: {list(zarr_store.get('data', {}).keys())}")

        images = zarr_store["data"]["img"]  # (N, H, W, 3) uint8
        actions = zarr_store["data"]["action"]  # (N, 2) float32
        episode_ends = zarr_store["meta"]["episode_ends"][:]  # (n_episodes,)

        episode_starts = [0] + list(episode_ends[:-1])
        n_episodes = len(episode_ends)

        # Compute action normalization stats from ALL data (before train/val split)
        self.action_stats = None
        if normalize_action:
            all_actions = actions[:].astype(np.float32)
            action_min = all_actions.min(axis=0)  # (action_dim,)
            action_max = all_actions.max(axis=0)  # (action_dim,)
            self.action_stats = {
                'min': action_min,
                'max': action_max,
            }
            print(f"PushT action stats: min={action_min}, max={action_max}")

        # Train/val split
        np.random.seed(seed)
        perm = np.random.permutation(n_episodes)
        n_val = max(1, int(n_episodes * val_ratio))
        val_indices = set(perm[:n_val])
        train_indices = set(perm[n_val:])

        if split == "train":
            episode_indices = sorted(train_indices)
        else:
            episode_indices = sorted(val_indices)

        self.images = images
        self.actions = actions
        self.episode_ends = episode_ends
        self.episode_starts = episode_starts
        self.episode_indices = episode_indices

        self.index_pool = []
        for ep_idx in episode_indices:
            start = episode_starts[ep_idx]
            end = episode_ends[ep_idx]
            for frame_idx in range(start + max_condition_frames, end):
                self.index_pool.append((ep_idx, frame_idx))

        print(f"PushTVideoDataset ({split}): {len(self.index_pool)} samples from {len(episode_indices)} episodes")
        if normalize_action:
            print(f"  Action normalization: ON ([-1, 1])")

    def __len__(self):
        return len(self.index_pool)

    def _normalize_action(self, action):
        """Normalize action from raw space to [-1, 1] using precomputed min/max."""
        if self.action_stats is None:
            return action
        a_min = self.action_stats['min']
        a_max = self.action_stats['max']
        # raw -> [0, 1] -> [-1, 1]
        action = (action - a_min) / (a_max - a_min + 1e-8) * 2.0 - 1.0
        return action

    def __getitem__(self, idx):
        ep_idx, frame_idx = self.index_pool[idx]
        start = self.episode_starts[ep_idx]

        condition_frames = []
        for i in range(self.max_condition_frames):
            cond_frame = self.images[frame_idx - self.max_condition_frames + i]  # (H, W, 3)
            condition_frames.append(cond_frame)
        condition_frames = np.stack(condition_frames)

        target_frame = self.images[frame_idx]
        action = self.actions[frame_idx].astype(np.float32)

        if self.normalize_action:
            action = self._normalize_action(action)

        h, w = target_frame.shape[:2]
        condition_frames = torch.from_numpy(condition_frames).float()
        target_frame = torch.from_numpy(target_frame).float()

        condition_frames = condition_frames.permute(0, 3, 1, 2) / 127.5 - 1.0
        target_frame = target_frame.permute(2, 0, 1) / 127.5 - 1.0

        if h != self.image_size or w != self.image_size:
            condition_frames = F.interpolate(
                condition_frames, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False
            )
            target_frame = F.interpolate(
                target_frame.unsqueeze(0), size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False
            ).squeeze(0)

        return {
            "condition_frames": condition_frames,
            "target_frame": target_frame,
            "action": torch.from_numpy(action).float(),
        }
