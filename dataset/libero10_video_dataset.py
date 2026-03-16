"""
LIBERO-10 video dataset adapter for JiT-opt.
Returns samples compatible with JiT training:
{
    "condition_frames": (K, 3, H, W) in [-1, 1],
    "target_frame": (3, H, W) in [-1, 1],
    "action": (action_horizon * 10,) in [-1, 1] (optional normalization)
}
"""

import glob
import os
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import h5py
import numpy as np
import scipy.spatial.transform as st
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


def _axis_angle_to_rot6d(axis_angle: np.ndarray) -> np.ndarray:
    """Convert axis-angle (..., 3) to rotation-6D (..., 6)."""
    rot = st.Rotation.from_rotvec(axis_angle)
    rot_mat = rot.as_matrix()  # (..., 3, 3)
    return rot_mat[..., :2, :].reshape(axis_angle.shape[:-1] + (6,))


def _convert_action_to_10d(action: np.ndarray) -> np.ndarray:
    """
    Convert 7D action [pos(3), axis-angle(3), gripper(1)] to
    10D action [pos(3), rot6d(6), gripper(1)].
    If input is already 10D, return as-is.
    """
    if action.shape[-1] == 10:
        return action.astype(np.float32)
    if action.shape[-1] != 7:
        raise ValueError(f"Expected action dim 7 or 10, got {action.shape[-1]}")
    pos = action[..., :3]
    rot_axis_angle = action[..., 3:6]
    gripper = action[..., 6:]
    rot6d = _axis_angle_to_rot6d(rot_axis_angle)
    return np.concatenate([pos, rot6d, gripper], axis=-1).astype(np.float32)


@dataclass
class _EpisodeRef:
    file_idx: int
    demo_key: str
    length: int


class Libero10VideoDataset(Dataset):
    def __init__(
        self,
        dataset_root_dir: str,
        max_condition_frames: int = 2,
        image_size: int = 128,
        split: str = "train",
        val_ratio: float = 0.02,
        seed: int = 42,
        normalize_action: bool = True,
        action_horizon: int = 1,
        dataset_names: Optional[List[str]] = None,
        image_obs_key: str = "agentview_rgb",
        match_uva_image_transform: bool = True,
        use_augmentation: bool = False,
        random_crop_pad: int = 4,
        blur_prob: float = 0.2,
        blur_kernel_size: int = 5,
        blur_sigma_min: float = 0.1,
        blur_sigma_max: float = 1.5,
        **kwargs,
    ):
        super().__init__()
        if split not in ("train", "val"):
            raise ValueError(f"split must be train/val, got {split}")
        if action_horizon < 1:
            raise ValueError(f"action_horizon must be >=1, got {action_horizon}")
        if max_condition_frames < 1:
            raise ValueError(f"max_condition_frames must be >=1, got {max_condition_frames}")

        self.dataset_root_dir = os.path.abspath(os.path.expanduser(dataset_root_dir))
        self.max_condition_frames = int(max_condition_frames)
        self.image_size = int(image_size)
        self.split = split
        self.normalize_action = bool(normalize_action)
        self.action_horizon = int(action_horizon)
        self.image_obs_key = image_obs_key
        self.match_uva_image_transform = bool(match_uva_image_transform)
        self.use_augmentation = bool(use_augmentation and split == "train")
        self.random_crop_pad = max(0, int(random_crop_pad))
        self.blur_prob = float(blur_prob)
        self.blur_kernel_size = int(blur_kernel_size)
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1
        self.blur_sigma_min = float(blur_sigma_min)
        self.blur_sigma_max = float(max(blur_sigma_min, blur_sigma_max))

        if not os.path.isdir(self.dataset_root_dir):
            raise ValueError(f"LIBERO dataset directory not found: {self.dataset_root_dir}")

        self.file_paths = self._resolve_hdf5_files(self.dataset_root_dir, dataset_names)
        if not self.file_paths:
            raise ValueError(
                f"No .hdf5 files found in {self.dataset_root_dir} "
                f"(dataset_names={dataset_names})"
            )

        self.files: List[h5py.File] = [h5py.File(p, "r") for p in self.file_paths]

        all_episodes: List[_EpisodeRef] = []
        all_actions_for_stats: List[np.ndarray] = []

        for file_idx, h5_file in enumerate(self.files):
            if "data" not in h5_file:
                continue
            demo_keys = sorted(
                [k for k in h5_file["data"].keys() if k.startswith("demo_")],
                key=lambda x: int(x.split("_")[-1]),
            )
            for demo_key in demo_keys:
                demo = h5_file["data"][demo_key]
                if "obs" not in demo or self.image_obs_key not in demo["obs"]:
                    continue
                if "actions" not in demo:
                    continue
                length = int(demo["actions"].shape[0])
                if length <= self.max_condition_frames + self.action_horizon:
                    continue
                all_episodes.append(_EpisodeRef(file_idx=file_idx, demo_key=demo_key, length=length))
                all_actions_for_stats.append(_convert_action_to_10d(demo["actions"][:].astype(np.float32)))

        if not all_episodes:
            raise ValueError("No valid episodes found in LIBERO dataset.")

        # Action stats from all episodes to match PushT handling.
        self.action_stats = None
        if self.normalize_action and all_actions_for_stats:
            stacked = np.concatenate(all_actions_for_stats, axis=0)
            self.action_stats = {
                "min": stacked.min(axis=0),
                "max": stacked.max(axis=0),
            }

        # Train/val split by episode.
        rng = np.random.RandomState(seed)
        perm = rng.permutation(len(all_episodes))
        n_val = max(1, int(len(all_episodes) * val_ratio))
        val_set = set(perm[:n_val].tolist())

        if split == "train":
            selected_indices = [i for i in range(len(all_episodes)) if i not in val_set]
        else:
            selected_indices = [i for i in range(len(all_episodes)) if i in val_set]
        self.episodes = [all_episodes[i] for i in selected_indices]

        # Build sample index pool: (episode_idx, frame_idx)
        self.index_pool: List[Tuple[int, int]] = []
        for ep_idx, ep in enumerate(self.episodes):
            # frame_idx is the target frame index.
            frame_start = self.max_condition_frames
            frame_end = ep.length - self.action_horizon
            for frame_idx in range(frame_start, frame_end + 1):
                self.index_pool.append((ep_idx, frame_idx))

        print(
            f"Libero10VideoDataset ({split}): "
            f"{len(self.index_pool)} samples from {len(self.episodes)} episodes, "
            f"{len(self.file_paths)} files"
        )
        if self.action_stats is not None:
            print(f"  action_min={self.action_stats['min']}")
            print(f"  action_max={self.action_stats['max']}")
        if self.use_augmentation:
            print(
                f"  augmentation: random_crop_pad={self.random_crop_pad}, "
                f"blur_prob={self.blur_prob}, blur_kernel={self.blur_kernel_size}, "
                f"blur_sigma=({self.blur_sigma_min}, {self.blur_sigma_max})"
            )

    @staticmethod
    def _resolve_hdf5_files(dataset_root_dir: str, dataset_names: Optional[List[str]]) -> List[str]:
        if dataset_names:
            files = []
            for name in dataset_names:
                name = name.strip()
                if not name:
                    continue
                if not name.endswith(".hdf5"):
                    name = f"{name}.hdf5"
                path = os.path.join(dataset_root_dir, name)
                if os.path.exists(path):
                    files.append(path)
            if files:
                return sorted(files)
            print(
                "Warning: no matching LIBERO files found for dataset_names, "
                "falling back to all *.hdf5 files."
            )
        return sorted(glob.glob(os.path.join(dataset_root_dir, "*.hdf5")))

    def __len__(self):
        return len(self.index_pool)

    def _normalize_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_stats is None:
            return action
        a_min = self.action_stats["min"]
        a_max = self.action_stats["max"]
        return (action - a_min) / (a_max - a_min + 1e-8) * 2.0 - 1.0

    def __getitem__(self, idx: int):
        ep_idx, frame_idx = self.index_pool[idx]
        ep = self.episodes[ep_idx]
        demo = self.files[ep.file_idx]["data"][ep.demo_key]

        # frames: (T, H, W, 3) uint8
        frames_ds = demo["obs"][self.image_obs_key]
        condition = frames_ds[frame_idx - self.max_condition_frames : frame_idx]  # (K, H, W, 3)
        target = frames_ds[frame_idx]  # (H, W, 3)

        # action sequence from current target frame onwards.
        actions_raw = demo["actions"][frame_idx : frame_idx + self.action_horizon].astype(np.float32)
        actions_10d = _convert_action_to_10d(actions_raw)
        if self.normalize_action:
            actions_10d = self._normalize_action(actions_10d)
        action = actions_10d.reshape(-1)

        condition_t = torch.from_numpy(condition).float().permute(0, 3, 1, 2) / 127.5 - 1.0
        target_t = torch.from_numpy(target).float().permute(2, 0, 1) / 127.5 - 1.0

        # Match UVA LIBERO preprocessing:
        # rotate 180deg in image plane then horizontal flip.
        # This aligns replayed hdf5 frames with environment camera convention.
        if self.match_uva_image_transform:
            condition_t = torch.rot90(condition_t, k=2, dims=(-2, -1))
            condition_t = torch.flip(condition_t, dims=(-1,))
            target_t = torch.rot90(target_t, k=2, dims=(-2, -1))
            target_t = torch.flip(target_t, dims=(-1,))

        h, w = target.shape[:2]
        if h != self.image_size or w != self.image_size:
            condition_t = F.interpolate(
                condition_t,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
            target_t = F.interpolate(
                target_t.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        if self.use_augmentation:
            condition_t, target_t = self._augment_frames(condition_t, target_t)

        return {
            "condition_frames": condition_t,
            "target_frame": target_t,
            "action": torch.from_numpy(action).float(),
        }

    def _augment_frames(self, condition_frames: torch.Tensor, target_frame: torch.Tensor):
        """
        Apply temporally-consistent augmentation on condition+target frames.
        Inputs are in [-1, 1]:
          condition_frames: (K, C, H, W)
          target_frame: (C, H, W)
        """
        frames = torch.cat([condition_frames, target_frame.unsqueeze(0)], dim=0)  # (K+1, C, H, W)

        if self.random_crop_pad > 0:
            pad = self.random_crop_pad
            frames = F.pad(frames, (pad, pad, pad, pad), mode="reflect")
            _, _, hp, wp = frames.shape
            top = random.randint(0, hp - self.image_size)
            left = random.randint(0, wp - self.image_size)
            frames = frames[:, :, top : top + self.image_size, left : left + self.image_size]

        if self.blur_prob > 0 and random.random() < self.blur_prob:
            sigma = random.uniform(self.blur_sigma_min, self.blur_sigma_max)
            frames = TF.gaussian_blur(frames, kernel_size=self.blur_kernel_size, sigma=sigma)

        return frames[:-1], frames[-1]
