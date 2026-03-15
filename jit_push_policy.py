"""
JiT Policy wrapper for PushT environment evaluation.
Implements BaseImagePolicy interface compatible with PushTImageRunner from UVA.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from uva_path_utils import ensure_uva_on_sys_path

# Add UVA path for imports (supports unified_video_action or uva-bo repo name).
ensure_uva_on_sys_path(anchor_file=__file__)

from unified_video_action.policy.base_image_policy import BaseImagePolicy
from unified_video_action.model.common.normalizer import LinearNormalizer
from unified_video_action.utils.data_utils import unnormalize_future_action


class JitPushTPolicy(BaseImagePolicy):
    """
    Policy wrapper that adapts JiT model for PushT environment rollout.
    Uses last 2 frames as condition, last frame as target, predicts action and chunks to 8 steps.
    """

    def __init__(
        self,
        denoiser,
        n_action_steps=8,
        max_condition_frames=2,
        img_size=256,
        action_dim=2,
        action_horizon=None,
        normalizer=None,
        normalizer_type="all",
        action_stats=None,
        device=None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.n_action_steps = n_action_steps
        self.max_condition_frames = max_condition_frames
        self.img_size = img_size
        self.action_dim = action_dim
        # Auto-detect action_horizon from model if not provided
        if action_horizon is None:
            action_horizon = getattr(denoiser.net, 'action_horizon', 1)
        self.action_horizon = action_horizon
        self.normalizer_type = normalizer_type
        self.action_stats = action_stats

        if normalizer is not None:
            self.normalizer = normalizer
        else:
            self.normalizer = LinearNormalizer()

        self._device = device or next(denoiser.parameters()).device

    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())

    def predict_action(self, obs_dict):
        """
        obs_dict: image (B, T, 3, 96, 96), agent_pos (B, T, 2)
        return: {"action": (B, n_action_steps, 2)}
        """
        device = self._device
        image = obs_dict["image"].to(device)  # (B, T, 3, 96, 96)

        B, T, C, H, W = image.shape
        n_cond = min(self.max_condition_frames, max(1, T - 1))
        n_target = 1

        # Use last n_cond frames for condition, last frame for target
        if T >= n_cond + 1:
            condition_frames = image[:, -n_cond - 1 : -1]  # (B, n_cond, 3, 96, 96)
        else:
            # Pad by repeating first frame if not enough history
            condition_frames = image[:, -1:].repeat(1, n_cond, 1, 1, 1)
        target_frame = image[:, -1]  # (B, 3, 96, 96)

        # Resize to JiT input size
        if H != self.img_size or W != self.img_size:
            condition_frames = F.interpolate(
                condition_frames.reshape(B * n_cond, C, H, W),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(B, n_cond, C, self.img_size, self.img_size)
            target_frame = F.interpolate(
                target_frame, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )

        # Convert from [0, 1] to [-1, 1]
        condition_frames = condition_frames * 2.0 - 1.0
        target_frame = target_frame * 2.0 - 1.0

        # Get action prediction from JiT diffusion action sampler
        with torch.no_grad():
            action_pred_flat = self.denoiser.generate_action(
                target_frame=target_frame,
                condition_frames=condition_frames,
            )  # (B, action_dim * action_horizon)

        # Reshape to (B, action_horizon, action_dim)
        action_pred = action_pred_flat.reshape(B, self.action_horizon, self.action_dim)
        # Keep predictions inside normalized range before denormalization.
        action_pred = torch.clamp(action_pred, -1.0, 1.0)

        # Denormalize each step from [-1, 1] back to original space
        if self.action_stats is not None:
            a_min = torch.tensor(self.action_stats['min'], dtype=action_pred.dtype, device=device)
            a_max = torch.tensor(self.action_stats['max'], dtype=action_pred.dtype, device=device)
            action_pred = (action_pred + 1.0) / 2.0 * (a_max - a_min) + a_min
        elif self.normalizer_type == "all" and "action" in self.normalizer.params_dict:
            orig_shape = action_pred.shape
            action_pred = unnormalize_future_action(
                normalizer=self.normalizer,
                normalizer_type=self.normalizer_type,
                actions=action_pred.reshape(B, -1),
            ).reshape(orig_shape)
        else:
            action_pred = (action_pred + 1) * 256
            action_pred = torch.clamp(action_pred, 0, 512)

        # Pad or truncate to n_action_steps
        if self.action_horizon >= self.n_action_steps:
            action_pred = action_pred[:, :self.n_action_steps]
        else:
            # Repeat last action to fill remaining steps
            pad = action_pred[:, -1:].expand(-1, self.n_action_steps - self.action_horizon, -1)
            action_pred = torch.cat([action_pred, pad], dim=1)

        return {"action": action_pred}
