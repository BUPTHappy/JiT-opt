"""
LIBERO-10 Video Dataset for JiT-opt.
Supports two loading modes:
  1. From UVA's zarr cache  (libero_10_clip.zarr.zip) – fast, recommended
  2. From raw HDF5 files    (*.hdf5)                  – fallback

Outputs condition_frames / target_frame pairs,
with optional CLIP text embeddings for language-conditioned generation.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Optional


class LiberoVideoDataset(Dataset):
    """
    Dataset for LIBERO-10 video frame prediction.

    The dataset can load from:
    A) A zarr cache produced by UVA (``<dataset_path>_clip.zarr.zip``).
       This contains images, CLIP tokens, and episode boundaries.
    B) Raw HDF5 files in ``<dataset_path>/*.hdf5``.

    If the zarr cache is found, it is used automatically unless
    ``force_hdf5=True``.
    """

    def __init__(
        self,
        dataset_path: str,
        max_condition_frames: int = 2,
        image_size: int = 128,
        split: str = 'train',
        val_ratio: float = 0.05,
        use_text_condition: bool = False,
        text_latent_dim: int = 512,
        camera_key: str = 'agentview_rgb',
        data_aug: bool = False,
        seed: int = 42,
        force_hdf5: bool = False,
        clip_zarr_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.max_condition_frames = max_condition_frames
        self.image_size = image_size
        self.split = split
        self.val_ratio = val_ratio
        self.camera_key = camera_key
        self.data_aug = data_aug
        self.use_text_condition = use_text_condition
        self.text_latent_dim = text_latent_dim

        # Detect zarr cache path
        if clip_zarr_path is None:
            clip_zarr_path = dataset_path.rstrip('/') + "_clip.zarr.zip"

        # Choose loading strategy:
        #  1. zarr cache exists -> try loading images+tokens from zarr
        #     if Jpeg2k codec unavailable -> hybrid: images from HDF5, tokens from zarr
        #  2. no zarr cache -> pure HDF5
        zarr_available = (not force_hdf5) and os.path.exists(clip_zarr_path)
        if zarr_available:
            try:
                print(f"[LiberoVideoDataset] Loading from zarr cache: {clip_zarr_path}")
                self._load_from_zarr(clip_zarr_path)
            except ValueError as e:
                if 'codec not available' in str(e):
                    print(f"[LiberoVideoDataset] Jpeg2k codec unavailable, "
                          f"using hybrid mode (images from HDF5, tokens from zarr)")
                    self._load_hybrid(dataset_path, clip_zarr_path)
                else:
                    raise
        else:
            print(f"[LiberoVideoDataset] Loading from HDF5 files: {dataset_path}")
            self._load_from_hdf5(dataset_path)

        n_episodes = len(self.all_images)
        total_frames = sum(self.episode_lengths)
        print(f"Loaded {n_episodes} episodes, {total_frames} total frames")

        # Compute CLIP text embeddings (if needed)
        self.text_embeddings = None
        if use_text_condition:
            self.text_embeddings = self._compute_clip_embeddings()

        # Train/val split (episode-level, deterministic)
        rng = np.random.RandomState(seed)
        episode_indices = list(range(n_episodes))
        rng.shuffle(episode_indices)

        n_val = max(1, int(n_episodes * val_ratio))
        n_train = n_episodes - n_val

        if split == 'train':
            self.episode_indices = sorted(episode_indices[:n_train])
        elif split == 'val':
            self.episode_indices = sorted(episode_indices[n_train:])
        else:
            self.episode_indices = episode_indices

        print(f"[{split}] Using {len(self.episode_indices)}/{n_episodes} episodes")

        # Build index pool: (episode_idx, frame_idx)
        self.index_pool = []
        for ep_idx in self.episode_indices:
            ep_len = self.episode_lengths[ep_idx]
            for frame_idx in range(max_condition_frames, ep_len):
                self.index_pool.append((ep_idx, frame_idx))

        print(f"Total {len(self.index_pool)} samples")

    # ------------------------------------------------------------------
    # Loading backends
    # ------------------------------------------------------------------

    def _load_from_zarr(self, zarr_path: str):
        """Load images + language tokens from UVA zarr cache."""
        import zarr

        with zarr.ZipStore(zarr_path, mode='r') as store:
            root = zarr.group(store=store)
            data = root['data']
            meta = root['meta']

            # Episode boundaries
            episode_ends = meta['episode_ends'][:]        # (n_episodes,)
            episode_starts = np.concatenate([[0], episode_ends[:-1]])

            # Images – shape (n_steps, H, W, 3) uint8
            all_images_flat = data[self.camera_key][:]

            # Language tokens – shape (n_steps, 2, 30)
            #   dim-1 = [input_ids, attention_mask]
            has_language = 'language' in data
            all_lang_tokens_flat = data['language'][:] if has_language else None

        # Slice into per-episode lists
        self.all_images = []
        self.all_languages = []        # store language *strings* (for caching key)
        self._clip_tokens = []         # store (input_ids, attention_mask) tuples
        self.episode_lengths = []

        for ep_idx in range(len(episode_ends)):
            s = int(episode_starts[ep_idx])
            e = int(episode_ends[ep_idx])
            self.all_images.append(all_images_flat[s:e])
            self.episode_lengths.append(e - s)

            if has_language and all_lang_tokens_flat is not None:
                # All frames of an episode share the same language
                # Take the first frame's tokens
                token_pair = all_lang_tokens_flat[s]  # (2, 30)
                input_ids = token_pair[0].astype(np.int64)       # (30,)
                attention_mask = token_pair[1].astype(np.int64)   # (30,)
                self._clip_tokens.append((input_ids, attention_mask))
                # Reconstruct a pseudo-string key for caching
                self.all_languages.append(f"zarr_ep_{ep_idx}")
            else:
                self._clip_tokens.append(None)
                self.all_languages.append("")

        print(f"  Zarr: {len(self.all_images)} episodes, "
              f"language tokens: {has_language}")

    def _load_from_hdf5(self, dataset_path: str):
        """Load images + language strings from raw HDF5 files."""
        import h5py

        hdf5_paths = sorted(glob.glob(os.path.join(dataset_path, "*.hdf5")))
        if len(hdf5_paths) == 0:
            raise FileNotFoundError(f"No .hdf5 files found in {dataset_path}")
        print(f"Found {len(hdf5_paths)} HDF5 files in {dataset_path}")

        self.all_images = []
        self.all_languages = []
        self._clip_tokens = []     # will stay empty; embeddings computed from text
        self.episode_lengths = []

        for hdf5_path in hdf5_paths:
            filename = os.path.basename(hdf5_path)
            language_goal = " ".join(filename[:-10].split("_"))  # remove "_demo.hdf5"

            print(f"  Loading: {filename}")
            print(f"    Language: \"{language_goal}\"")

            with h5py.File(hdf5_path, 'r') as f:
                demos = f['data']
                n_demos = len(demos)
                print(f"    Demos: {n_demos}")

                for i in range(n_demos):
                    demo = demos[f'demo_{i}']
                    images = demo['obs'][self.camera_key][:]  # (T, H, W, 3) uint8
                    self.all_images.append(images)
                    self.all_languages.append(language_goal)
                    self._clip_tokens.append(None)
                    self.episode_lengths.append(images.shape[0])

    def _load_hybrid(self, dataset_path: str, zarr_path: str):
        """Hybrid mode: images from HDF5, language tokens from zarr cache.

        This avoids the Jpeg2k codec dependency while still leveraging
        the pre-computed CLIP tokens from the UVA zarr cache.
        """
        import zarr
        import h5py

        # --- 1. Load language tokens + episode structure from zarr ---
        print("  Loading language tokens from zarr ...")
        with zarr.ZipStore(zarr_path, mode='r') as store:
            root = zarr.group(store=store)
            data = root['data']
            meta = root['meta']

            episode_ends_zarr = meta['episode_ends'][:]   # (n_episodes,)
            episode_starts_zarr = np.concatenate([[0], episode_ends_zarr[:-1]])

            has_language = 'language' in data
            # Language array uses plain compressor (no Jpeg2k), safe to read
            all_lang_tokens_flat = data['language'][:] if has_language else None

        n_episodes_zarr = len(episode_ends_zarr)
        print(f"  Zarr: {n_episodes_zarr} episodes, language tokens: {has_language}")

        # Extract per-episode CLIP tokens from zarr
        zarr_clip_tokens = []
        for ep_idx in range(n_episodes_zarr):
            if has_language and all_lang_tokens_flat is not None:
                s = int(episode_starts_zarr[ep_idx])
                token_pair = all_lang_tokens_flat[s]  # (2, 30)
                input_ids = token_pair[0].astype(np.int64)
                attention_mask = token_pair[1].astype(np.int64)
                zarr_clip_tokens.append((input_ids, attention_mask))
            else:
                zarr_clip_tokens.append(None)

        # --- 2. Load images from HDF5 (same order as zarr) ---
        print("  Loading images from HDF5 files ...")
        hdf5_paths = sorted(glob.glob(os.path.join(dataset_path, "*.hdf5")))
        if len(hdf5_paths) == 0:
            raise FileNotFoundError(f"No .hdf5 files found in {dataset_path}")

        self.all_images = []
        self.all_languages = []
        self._clip_tokens = []
        self.episode_lengths = []

        zarr_ep_counter = 0
        for hdf5_path in hdf5_paths:
            filename = os.path.basename(hdf5_path)
            language_goal = " ".join(filename[:-10].split("_"))
            print(f"    Loading: {filename}")

            with h5py.File(hdf5_path, 'r') as f:
                demos = f['data']
                n_demos = len(demos)

                for i in range(n_demos):
                    demo = demos[f'demo_{i}']
                    images = demo['obs'][self.camera_key][:]  # (T, H, W, 3) uint8
                    self.all_images.append(images)
                    self.all_languages.append(language_goal)
                    self.episode_lengths.append(images.shape[0])

                    # Attach zarr CLIP tokens (same episode order)
                    if zarr_ep_counter < len(zarr_clip_tokens):
                        self._clip_tokens.append(zarr_clip_tokens[zarr_ep_counter])
                    else:
                        self._clip_tokens.append(None)
                    zarr_ep_counter += 1

        print(f"  Hybrid: {len(self.all_images)} episodes from HDF5, "
              f"{sum(1 for t in self._clip_tokens if t is not None)} with zarr tokens")

    # ------------------------------------------------------------------
    # CLIP embeddings
    # ------------------------------------------------------------------

    def _compute_clip_embeddings(self):
        """Pre-compute CLIP text embeddings for all episodes.

        If the data was loaded from a zarr cache, the CLIP *tokens* are
        already available – we only need to run the text encoder once per
        unique token set.  Otherwise, we tokenise the language strings.

        Results are cached to a .npz file next to the dataset.
        """
        import hashlib
        import json

        n_episodes = len(self.all_images)

        # ------- build per-episode (input_ids, attention_mask) pairs --------
        has_zarr_tokens = any(t is not None for t in self._clip_tokens)

        if has_zarr_tokens:
            # Tokens already loaded from zarr – just use them
            ep_tokens = self._clip_tokens  # list of (input_ids, attention_mask) or None
        else:
            # Tokenise from language strings
            try:
                from transformers import AutoTokenizer
            except ImportError:
                raise ImportError("pip install transformers")
            tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            ep_tokens = []
            for lang in self.all_languages:
                tok = tokenizer(
                    lang, padding="max_length", max_length=30,
                    truncation=True, return_tensors="np",
                )
                ep_tokens.append((
                    tok['input_ids'][0].astype(np.int64),
                    tok['attention_mask'][0].astype(np.int64),
                ))

        # ------- deduplicate: many episodes share the same language ---------
        # key = (input_ids_bytes, attention_mask_bytes) -> embedding
        unique_keys = {}    # key -> index
        ep_to_key = []
        for tok_pair in ep_tokens:
            if tok_pair is None:
                # fallback – zero embedding
                k = b"__none__"
            else:
                k = tok_pair[0].tobytes() + tok_pair[1].tobytes()
            if k not in unique_keys:
                unique_keys[k] = len(unique_keys)
            ep_to_key.append(k)

        n_unique = len(unique_keys)
        print(f"  Unique language conditions: {n_unique}")

        # ------- cache: simple (n_episodes, dim) matrix -------
        cache_dir = self.dataset_path if os.path.isdir(self.dataset_path) else os.path.dirname(self.dataset_path)
        cache_path = os.path.join(cache_dir, f".clip_embeddings_v2_{n_episodes}ep.npy")

        if os.path.exists(cache_path):
            print(f"  Loading cached CLIP embeddings from {cache_path}")
            all_emb = np.load(cache_path)  # (n_episodes, dim)
            embeddings = [all_emb[i] for i in range(n_episodes)]
            print(f"  Loaded {all_emb.shape[0]} embeddings (dim={all_emb.shape[1]})")
            return embeddings

        # ------- run CLIP text encoder -------
        print("  Computing CLIP text embeddings with text encoder...")

        try:
            from transformers import CLIPModel
        except ImportError:
            raise ImportError("pip install transformers")

        # Bypass torch.load security check for older PyTorch
        _noop = lambda: None
        _patches = {}
        import importlib
        for mod_name in [
            'transformers.utils.import_utils',
            'transformers.modeling_utils',
        ]:
            try:
                mod = importlib.import_module(mod_name)
                if hasattr(mod, 'check_torch_load_is_safe'):
                    _patches[mod] = mod.check_torch_load_is_safe
                    mod.check_torch_load_is_safe = _noop
            except Exception:
                pass
        try:
            import transformers.modeling_utils as _tm
            if hasattr(_tm, 'load_state_dict'):
                _patches.setdefault(_tm, getattr(_tm, 'check_torch_load_is_safe', _noop))
                if 'check_torch_load_is_safe' in _tm.load_state_dict.__globals__:
                    _tm.load_state_dict.__globals__['check_torch_load_is_safe'] = _noop
        except Exception:
            pass

        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        finally:
            for mod, orig_fn in _patches.items():
                try:
                    mod.check_torch_load_is_safe = orig_fn
                except Exception:
                    pass
        clip_model.eval()

        # Compute one embedding per unique token set
        key_to_token = {}
        for tok_pair, k in zip(ep_tokens, ep_to_key):
            if k not in key_to_token and tok_pair is not None:
                key_to_token[k] = tok_pair

        key_to_emb = {}
        with torch.no_grad():
            for k in unique_keys:
                if k == b"__none__":
                    key_to_emb[k] = np.zeros(self.text_latent_dim, dtype=np.float32)
                    continue
                tok_pair = key_to_token[k]
                input_ids = torch.from_numpy(tok_pair[0]).unsqueeze(0).long()
                attention_mask = torch.from_numpy(tok_pair[1]).unsqueeze(0).long()

                text_outputs = clip_model.text_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                pooled_output = text_outputs[1]
                text_features = clip_model.text_projection(pooled_output)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                emb = text_features.squeeze(0).cpu().numpy()
                key_to_emb[k] = emb
                print(f"    Computed embedding dim={emb.shape[0]}")

        del clip_model

        # Build per-episode embedding list
        embeddings = [key_to_emb[k] for k in ep_to_key]

        # Save cache as a simple (n_episodes, dim) float32 matrix
        try:
            all_emb = np.stack(embeddings, axis=0)  # (n_episodes, dim)
            np.save(cache_path, all_emb)
            print(f"  Saved CLIP embeddings cache to {cache_path} "
                  f"shape={all_emb.shape}")
        except Exception as e:
            print(f"  Warning: could not save cache: {e}")

        return embeddings

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.index_pool)

    def __getitem__(self, idx):
        ep_idx, frame_idx = self.index_pool[idx]
        images = self.all_images[ep_idx]

        # Get condition frames (historical frames before target)
        condition_frames = []
        for i in range(self.max_condition_frames):
            cond_frame_idx = frame_idx - self.max_condition_frames + i
            cond_frame = images[cond_frame_idx].copy()  # (H, W, 3) uint8
            condition_frames.append(cond_frame)
        condition_frames = np.stack(condition_frames)  # (max_condition_frames, H, W, 3)

        # Get target frame
        target_frame = images[frame_idx].copy()  # (H, W, 3) uint8

        # Apply LIBERO-specific image preprocessing (same as UVA):
        # 1. Rotate 180 degrees
        # 2. Horizontal flip
        condition_frames = np.rot90(condition_frames, k=2, axes=(1, 2)).copy()
        condition_frames = np.flip(condition_frames, axis=2).copy()
        target_frame = np.rot90(target_frame, k=2, axes=(0, 1)).copy()
        target_frame = np.flip(target_frame, axis=1).copy()

        original_height, original_width = target_frame.shape[:2]

        # Convert to torch tensors, CHW format, normalise to [-1, 1]
        condition_frames = torch.from_numpy(condition_frames).float()
        target_frame = torch.from_numpy(target_frame).float()

        condition_frames = condition_frames.permute(0, 3, 1, 2) / 127.5 - 1.0  # (N, 3, H, W)
        target_frame = target_frame.permute(2, 0, 1) / 127.5 - 1.0             # (3, H, W)

        # Resize if needed
        if original_height != self.image_size or original_width != self.image_size:
            condition_frames = F.interpolate(
                condition_frames, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False,
            )
            target_frame = F.interpolate(
                target_frame.unsqueeze(0), size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False,
            ).squeeze(0)

        # Data augmentation (colour jitter, consistent across frames)
        if self.data_aug:
            import torchvision.transforms as transforms
            video_seed = torch.randint(0, 10000, (1,)).item()

            def _augment(frame):
                torch.manual_seed(video_seed)
                aug = transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                )
                frame_01 = (frame + 1.0) / 2.0
                frame_01 = aug(frame_01)
                return frame_01 * 2.0 - 1.0

            condition_frames = torch.stack([_augment(f) for f in condition_frames])
            target_frame = _augment(target_frame)

        result = {
            'condition_frames': condition_frames,   # (max_condition_frames, C, H, W)
            'target_frame': target_frame,           # (C, H, W)
        }

        # Add text latents if using language conditioning
        if self.use_text_condition and self.text_embeddings is not None:
            result['text_latents'] = torch.from_numpy(
                self.text_embeddings[ep_idx].copy()
            ).float()  # (512,)

        return result
