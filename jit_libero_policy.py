"""
JiT policy wrapper for LIBERO simulation evaluation.
Compatible with UVA's LiberoImageRunner.
"""

import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

_UVA_ROOT = os.path.join(os.path.dirname(__file__), "..", "unified_video_action")
if os.path.exists(_UVA_ROOT) and _UVA_ROOT not in sys.path:
    sys.path.append(_UVA_ROOT)

from unified_video_action.policy.base_image_policy import BaseImagePolicy


class JitLiberoPolicy(BaseImagePolicy):
    def __init__(
        self,
        denoiser,
        n_action_steps: int = 8,
        max_condition_frames: int = 2,
        img_size: int = 128,
        action_dim: int = 10,
        action_horizon: Optional[int] = None,
        action_stats: Optional[dict] = None,
        device=None,
    ):
        super().__init__()
        self.denoiser = denoiser
        self.n_action_steps = int(n_action_steps)
        self.max_condition_frames = int(max_condition_frames)
        self.img_size = int(img_size)
        self.action_dim = int(action_dim)
        if action_horizon is None:
            action_horizon = getattr(denoiser.net, "action_horizon", 1)
        self.action_horizon = int(action_horizon)
        self.action_stats = action_stats
        self._device = device or next(denoiser.parameters()).device

    def set_normalizer(self, normalizer):
        # JiT uses min/max action stats for now; keep API compatibility.
        return

    def reset(self):
        return

    def _denormalize_action(self, action: torch.Tensor) -> torch.Tensor:
        if self.action_stats is None:
            return action
        a_min = torch.tensor(self.action_stats["min"], dtype=action.dtype, device=action.device)
        a_max = torch.tensor(self.action_stats["max"], dtype=action.dtype, device=action.device)
        return (action + 1.0) / 2.0 * (a_max - a_min) + a_min

    def predict_action(
        self,
        obs_dict: Dict[str, torch.Tensor],
        language_goal: Optional[List[str]] = None,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        obs_dict["agentview_image"]: (B, T, 3, H, W), usually in [0, 1].
        return:
            {"action": (B, n_action_steps, action_dim)}
        """
        device = self._device
        image = obs_dict["agentview_image"].to(device)
        image = image.to(torch.float32)

        bsz, t_hist, c, h, w = image.shape
        n_cond = min(self.max_condition_frames, max(1, t_hist - 1))

        if t_hist >= n_cond + 1:
            condition_frames = image[:, -n_cond - 1 : -1]
        else:
            condition_frames = image[:, -1:].repeat(1, n_cond, 1, 1, 1)
        target_frame = image[:, -1]

        # Resize to JiT input resolution if needed.
        if h != self.img_size or w != self.img_size:
            condition_frames = F.interpolate(
                condition_frames.reshape(bsz * n_cond, c, h, w),
                size=(self.img_size, self.img_size),
                mode="bilinear",
                align_corners=False,
            ).reshape(bsz, n_cond, c, self.img_size, self.img_size)
            target_frame = F.interpolate(
                target_frame, size=(self.img_size, self.img_size), mode="bilinear", align_corners=False
            )

        # Convert [0, 1] -> [-1, 1].
        if image.max() <= 1.5 and image.min() >= -0.5:
            condition_frames = condition_frames * 2.0 - 1.0
            target_frame = target_frame * 2.0 - 1.0

        with torch.no_grad():
            action_pred_flat = self.denoiser.generate_action(
                target_frame=target_frame,
                condition_frames=condition_frames,
                text_latents=None,
            )  # (B, action_horizon * action_dim)

        action_pred = action_pred_flat.reshape(bsz, self.action_horizon, self.action_dim)
        action_pred = torch.clamp(action_pred, -1.0, 1.0)
        action_pred = self._denormalize_action(action_pred)

        # Runner expects fixed rollout chunk length.
        if self.action_horizon >= self.n_action_steps:
            action_pred = action_pred[:, : self.n_action_steps]
        else:
            pad = action_pred[:, -1:].expand(-1, self.n_action_steps - self.action_horizon, -1)
            action_pred = torch.cat([action_pred, pad], dim=1)

        return {"action": action_pred}
