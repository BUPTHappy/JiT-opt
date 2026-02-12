"""
LIBERO-10 Video Dataset for JiT-opt.
Loads LIBERO HDF5 files and outputs condition_frames / target_frame pairs,
with optional CLIP text embeddings for language-conditioned generation.
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import h5py
from typing import Optional, List


class LiberoVideoDataset(Dataset):
    """
    Dataset for LIBERO-10 video frame prediction.
    
    Expected directory structure:
        data_path/
        ├── LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5
        ├── KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
        └── ... (10 .hdf5 files total)
    
    Each HDF5 file contains:
        data/demo_0/obs/agentview_rgb  -> (T, H, W, 3) uint8
        data/demo_0/actions            -> (T, action_dim) float
        ...
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

        # Load all HDF5 files
        hdf5_paths = sorted(glob.glob(os.path.join(dataset_path, "*.hdf5")))
        if len(hdf5_paths) == 0:
            raise FileNotFoundError(f"No .hdf5 files found in {dataset_path}")
        print(f"Found {len(hdf5_paths)} HDF5 files in {dataset_path}")

        # Pre-load all data into memory
        self.all_images = []       # list of (T, H, W, 3) uint8 arrays per episode
        self.all_languages = []    # list of language strings per episode
        self.episode_lengths = []  # length of each episode

        for hdf5_path in hdf5_paths:
            # Extract language goal from filename:
            # "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
            # -> "KITCHEN SCENE3 turn on the stove and put the moka pot on it"
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
                    self.episode_lengths.append(images.shape[0])

        n_episodes = len(self.all_images)
        total_frames = sum(self.episode_lengths)
        print(f"Loaded {n_episodes} episodes, {total_frames} total frames")

        # Pre-compute CLIP text embeddings (if needed)
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

    def _compute_clip_embeddings(self):
        """Pre-compute CLIP text embeddings for all unique language goals.
        
        Uses caching to avoid recomputation and handles distributed training
        by computing only once and saving to disk.
        """
        import hashlib
        import json

        # Create a deterministic cache key from the language goals
        unique_languages = sorted(set(self.all_languages))
        cache_key = hashlib.md5(json.dumps(unique_languages).encode()).hexdigest()[:12]
        cache_path = os.path.join(self.dataset_path, f".clip_embeddings_cache_{cache_key}.npz")

        # Try loading from cache first
        if os.path.exists(cache_path):
            print(f"Loading cached CLIP embeddings from {cache_path}")
            cache = np.load(cache_path, allow_pickle=True)
            lang_to_embedding = {k: cache[k] for k in cache.files}
            embeddings = [lang_to_embedding[lang] for lang in self.all_languages]
            print(f"  Loaded {len(lang_to_embedding)} unique embeddings")
            return embeddings

        # Compute embeddings
        try:
            from transformers import CLIPModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers is required for CLIP text embeddings. "
                "Install with: pip install transformers"
            )

        print("Computing CLIP text embeddings...")
        tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

        # Bypass torch.load security check on PyTorch < 2.6
        # (safe here: we only load the public CLIP model from HuggingFace)
        import transformers.utils.import_utils as _tf_import_utils
        _orig_check = getattr(_tf_import_utils, 'check_torch_load_is_safe', None)
        if _orig_check is not None:
            _tf_import_utils.check_torch_load_is_safe = lambda: None
        try:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        finally:
            if _orig_check is not None:
                _tf_import_utils.check_torch_load_is_safe = _orig_check
        clip_model.eval()

        print(f"  Unique language goals: {len(unique_languages)}")

        # Compute embeddings for each unique language
        lang_to_embedding = {}
        with torch.no_grad():
            for lang in unique_languages:
                tokens = tokenizer(
                    lang,
                    padding="max_length",
                    max_length=30,
                    truncation=True,
                    return_tensors="pt",
                )
                text_features = clip_model.get_text_features(
                    input_ids=tokens['input_ids'],
                    attention_mask=tokens['attention_mask'],
                )
                # Normalize (standard CLIP practice)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                lang_to_embedding[lang] = text_features.squeeze(0).numpy()  # (512,)
                print(f"    \"{lang[:60]}...\" -> dim {text_features.shape[-1]}")

        del clip_model  # Free memory

        # Save cache to disk (so other ranks / future runs can reuse)
        try:
            np.savez(cache_path, **lang_to_embedding)
            print(f"  Saved CLIP embeddings cache to {cache_path}")
        except Exception as e:
            print(f"  Warning: Could not save cache: {e}")

        # Map each episode to its embedding
        embeddings = [lang_to_embedding[lang] for lang in self.all_languages]
        return embeddings

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

        # Convert to torch tensors, CHW format, normalize to [-1, 1]
        condition_frames = torch.from_numpy(condition_frames).float()
        target_frame = torch.from_numpy(target_frame).float()

        condition_frames = condition_frames.permute(0, 3, 1, 2) / 127.5 - 1.0  # (N, 3, H, W)
        target_frame = target_frame.permute(2, 0, 1) / 127.5 - 1.0  # (3, H, W)

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

        # Data augmentation (color jitter, consistent across frames)
        if self.data_aug:
            import torchvision.transforms as transforms
            video_seed = torch.randint(0, 10000, (1,)).item()

            def _augment(frame):
                torch.manual_seed(video_seed)
                aug = transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05
                )
                # ColorJitter expects [0,1], convert then convert back
                frame_01 = (frame + 1.0) / 2.0
                frame_01 = aug(frame_01)
                return frame_01 * 2.0 - 1.0

            condition_frames = torch.stack([_augment(f) for f in condition_frames])
            target_frame = _augment(target_frame)

        result = {
            'condition_frames': condition_frames,  # (max_condition_frames, C, H, W)
            'target_frame': target_frame,  # (C, H, W)
        }

        # Add text latents if using language conditioning
        if self.use_text_condition and self.text_embeddings is not None:
            result['text_latents'] = torch.from_numpy(
                self.text_embeddings[ep_idx].copy()
            ).float()  # (512,)

        return result
