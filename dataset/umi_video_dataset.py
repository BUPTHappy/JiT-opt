import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import zarr
from typing import Optional, List, Dict, Any

# 注册imagecodecs codec（用于UMI数据集的JPEG-XL压缩）
def _register_jpegxl_codec():
    """注册JPEG-XL codec用于UMI数据集"""
    try:
        from numcodecs.registry import register_codec, get_codec
        from numcodecs.abc import Codec
        import imagecodecs
        
        # 检查是否已经注册
        try:
            get_codec({"id": "imagecodecs_jpegxl"})
            # 已经注册，跳过
            return True
        except (ValueError, TypeError):
            # 未注册，继续注册
            pass
        
        if not imagecodecs.JPEGXL:
            print("⚠ Warning: imagecodecs.JPEGXL not available")
            return False
        
        # 定义JpegXl codec类
        class JpegXl(Codec):
            """JPEG XL codec for numcodecs."""
            codec_id = "imagecodecs_jpegxl"
            
            def __init__(
                self,
                # encode
                level=None,
                effort=None,
                distance=None,
                lossless=None,
                decodingspeed=None,
                photometric=None,
                planar=None,
                usecontainer=None,
                # decode
                index=None,
                keeporientation=None,
                # both
                numthreads=None,
            ):
                self.level = level
                self.effort = effort
                self.distance = distance
                self.lossless = bool(lossless) if lossless is not None else None
                self.decodingspeed = decodingspeed
                self.photometric = photometric
                self.planar = planar
                self.usecontainer = usecontainer
                self.index = index
                self.keeporientation = keeporientation
                self.numthreads = numthreads
            
            def encode(self, buf):
                buf = np.asarray(buf)
                return imagecodecs.jpegxl_encode(
                    buf,
                    level=self.level,
                    effort=self.effort,
                    distance=self.distance,
                    lossless=self.lossless,
                    decodingspeed=self.decodingspeed,
                    photometric=self.photometric,
                    planar=self.planar,
                    usecontainer=self.usecontainer,
                    numthreads=self.numthreads,
                )
            
            def decode(self, buf, out=None):
                return imagecodecs.jpegxl_decode(
                    buf,
                    index=self.index,
                    keeporientation=self.keeporientation,
                    numthreads=self.numthreads,
                    out=out,
                )
        
        # 注册codec
        register_codec(JpegXl)
        print("Registered imagecodecs_jpegxl codec")
        return True
        
    except ImportError as e:
        print(f"Warning: Could not import required modules: {e}")
        # 尝试使用UVA的codec注册
        try:
            import sys
            uva_path = os.path.join(os.path.dirname(__file__), '../../unified_video_action')
            if os.path.exists(uva_path):
                sys.path.insert(0, uva_path)
                from unified_video_action.codecs.imagecodecs_numcodecs import register_codecs
                register_codecs()
                print("Registered codecs from UVA")
                return True
        except Exception as e2:
            print(f"Warning: Could not register codecs from UVA: {e2}")
            return False
    except Exception as e:
        print(f"Warning: Could not register codecs: {e}")
        return False

# 在导入zarr之前注册codec
_register_jpegxl_codec()


class UmiVideoDataset(Dataset):
    # 支持的图像 key 列表，按优先级排列
    SUPPORTED_IMAGE_KEYS = ['camera0_rgb', 'img']

    def __init__(
        self,
        dataset_root_dir: str,
        max_condition_frames: int = 2,
        image_size: int = 256,
        split: str = 'train',  # 'train' or 'val'
        dataset_names: Optional[List[str]] = None,  # 数据集名称列表，如 ['cup_arrangement_0', 'towel_folding_0', 'mouse_arrangement_0']
        used_episode_indices_file: Optional[str] = None,  # JSON文件，指定使用的episode索引
        dataset_configs: Optional[Dict[str, Dict[str, Any]]] = None,  # 数据集配置（mask_mirror等）
        image_key: Optional[str] = None,  # 图像数据的 key，None 表示自动检测（兼容 UMI 的 camera0_rgb 和 PushT 的 img）
        **kwargs
    ):
        self.dataset_root_dir = dataset_root_dir
        self.max_condition_frames = max_condition_frames
        self.image_size = image_size
        self.split = split
        self.image_key = image_key  # None = 自动检测
        
        # 确定要加载的数据集
        if dataset_names is None:
            # 如果没有指定，尝试自动发现
            if os.path.isdir(dataset_root_dir):
                # 查找所有.zarr目录
                dataset_names = [f.replace('.zarr', '') for f in os.listdir(dataset_root_dir) 
                               if os.path.isdir(os.path.join(dataset_root_dir, f)) and f.endswith('.zarr')]
            else:
                raise ValueError("dataset_names must be provided if dataset_root_dir is not a directory")
        
        self.dataset_names = dataset_names
        print(f"Loading datasets: {self.dataset_names}")
        
        # 加载episode索引（如果提供）
        self.used_episode_indices = {}
        if used_episode_indices_file and os.path.exists(used_episode_indices_file):
            with open(used_episode_indices_file, 'r') as f:
                self.used_episode_indices = json.load(f)
            print(f"Loaded episode indices from {used_episode_indices_file}")
        
        # 数据集配置
        self.dataset_configs = dataset_configs or {}
        
        self.zarr_stores = []
        self.dataset_info = []  # 存储数据集信息
        self.index_pool = []  # (dataset_idx, episode_idx, frame_idx)
        
        # 加载所有数据集
        for dataset_name in self.dataset_names:
            zarr_path = os.path.join(dataset_root_dir, dataset_name + '.zarr')
            
            if not os.path.exists(zarr_path):
                print(f"Warning: {zarr_path} does not exist, skipping dataset {dataset_name}")
                continue
            
            print(f"Loading dataset: {dataset_name} from {zarr_path}")
            zarr_store = zarr.open(zarr_path, mode='r')
            
            # 检查数据格式，自动检测图像 key
            if 'data' not in zarr_store:
                print(f"Warning: {zarr_path} does not have 'data' group, skipping")
                continue
            
            # 确定图像 key：优先使用显式指定的 image_key，否则自动检测
            detected_image_key = self.image_key
            if detected_image_key is None:
                for candidate_key in self.SUPPORTED_IMAGE_KEYS:
                    if candidate_key in zarr_store['data']:
                        detected_image_key = candidate_key
                        break
            
            if detected_image_key is None or detected_image_key not in zarr_store['data']:
                available_keys = list(zarr_store['data'].keys()) if hasattr(zarr_store['data'], 'keys') else []
                print(f"Warning: {zarr_path} does not have a supported image key "
                      f"(tried: {self.SUPPORTED_IMAGE_KEYS}, available: {available_keys}), skipping")
                continue
            
            images = zarr_store['data'][detected_image_key]
            print(f"  Using image key: '{detected_image_key}', shape: {images.shape}")
            
            if 'meta' not in zarr_store or 'episode_ends' not in zarr_store['meta']:
                print(f"Warning: {zarr_path} does not have meta/episode_ends, skipping")
                continue
            
            episode_ends = zarr_store['meta']['episode_ends'][:]  # episode结束索引
            
            # 获取要使用的episode索引
            if dataset_name in self.used_episode_indices:
                used_episodes = self.used_episode_indices[dataset_name]
                print(f"  Using {len(used_episodes)} episodes from {dataset_name}")
            else:
                used_episodes = None  # 使用所有episodes
                print(f"  Using all {len(episode_ends)} episodes from {dataset_name}")
            
            self.zarr_stores.append({
                'store': zarr_store,
                'images': images,
                'episode_ends': episode_ends,
                'dataset_name': dataset_name,
                'used_episodes': used_episodes
            })
            self.dataset_info.append({
                'name': dataset_name,
                'total_frames': len(images),
                'num_episodes': len(episode_ends)
            })
        
        # 创建索引池
        self._create_index_pool()
        
        print(f"Loaded {len(self.zarr_stores)} dataset(s), total {len(self.index_pool)} samples")
    
    def _create_index_pool(self):
        """创建所有有效的 (dataset_idx, episode_idx, frame_idx) 索引
        
        按 episode 级别划分 train/val（95%/5%），确保同一个 episode 的帧
        不会同时出现在训练集和验证集中。
        """
        self.index_pool = []
        
        for dataset_idx, zarr_data in enumerate(self.zarr_stores):
            episode_ends = zarr_data['episode_ends']
            used_episodes = zarr_data['used_episodes']
            episode_starts = [0] + list(episode_ends[:-1])
            
            # 确定要处理的episodes
            if used_episodes is not None:
                # 只处理指定的episodes
                episode_indices = [i for i in used_episodes if 0 <= i < len(episode_ends)]
            else:
                # 处理所有episodes
                episode_indices = list(range(len(episode_ends)))
            
            # 按 episode 划分 train/val（95%/5%）
            num_episodes = len(episode_indices)
            num_train = max(1, int(num_episodes * 0.95))  # 至少保留1个train episode
            
            if self.split == 'train':
                episode_indices = episode_indices[:num_train]
            elif self.split == 'val':
                episode_indices = episode_indices[num_train:]
                if len(episode_indices) == 0:
                    # 如果episode太少不够分，val用最后一个episode
                    episode_indices = [episode_indices[-1]] if num_episodes > 0 else []
            # else: 使用全部（兼容性）
            
            dataset_name = zarr_data['dataset_name']
            print(f"  [{self.split}] {dataset_name}: using {len(episode_indices)}/{num_episodes} episodes")
            
            for ep_idx in episode_indices:
                start = episode_starts[ep_idx]
                end = episode_ends[ep_idx]
                
                # 确保有足够的历史帧
                for frame_idx in range(start + self.max_condition_frames, end):
                    self.index_pool.append((dataset_idx, ep_idx, frame_idx))
    
    def __len__(self):
        return len(self.index_pool)
    
    def __getitem__(self, idx):
        dataset_idx, ep_idx, frame_idx = self.index_pool[idx]
        zarr_data = self.zarr_stores[dataset_idx]
        images = zarr_data['images']
        
        # 获取条件帧
        condition_frames = []
        for i in range(self.max_condition_frames):
            cond_frame_idx = frame_idx - self.max_condition_frames + i
            cond_frame = images[cond_frame_idx]  # (H, W, 3) uint8
            condition_frames.append(cond_frame)
        condition_frames = np.stack(condition_frames)  # (max_condition_frames, H, W, 3)
        
        # 获取目标帧
        target_frame = images[frame_idx]  # (H, W, 3) uint8
        
        # 获取原始图像尺寸（动态检测）
        original_height, original_width = target_frame.shape[:2]
        
        # 转换为torch tensor
        condition_frames = torch.from_numpy(condition_frames).float()
        target_frame = torch.from_numpy(target_frame).float()
        
        # 转换为CHW格式并归一化到[-1, 1]
        condition_frames = condition_frames.permute(0, 3, 1, 2) / 127.5 - 1.0  # (max_condition_frames, 3, H, W)
        target_frame = target_frame.permute(2, 0, 1) / 127.5 - 1.0  # (3, H, W)
        
        # Resize如果需要（动态检测原始尺寸）
        if original_height != self.image_size or original_width != self.image_size:
            condition_frames = F.interpolate(
                condition_frames, size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            )
            target_frame = F.interpolate(
                target_frame.unsqueeze(0), size=(self.image_size, self.image_size),
                mode='bilinear', align_corners=False
            ).squeeze(0)
        
        result = {
            'condition_frames': condition_frames,  # (max_condition_frames, C, H, W)
            'target_frame': target_frame,  # (C, H, W)
        }

        return result