import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import zarr
from zarr.storage import DirectoryStore
from typing import Optional, List, Dict, Any
import scipy.spatial.transform as st

# 注册imagecodecs codec（用于UMI数据集的JPEG-XL压缩）
def _register_jpegxl_codec():
    """注册JPEG-XL codec用于UMI数据集"""
    from numcodecs.registry import get_codec
    
    # 首先检查是否已经注册
    try:
        get_codec({"id": "imagecodecs_jpegxl"})
        print("✓ imagecodecs_jpegxl codec already registered")
        return True
    except (ValueError, TypeError):
        # 未注册，继续尝试注册
        pass
    
    # 方法1: 尝试从UVA导入并注册（推荐方法）
    try:
        import sys
        uva_path = os.path.join(os.path.dirname(__file__), '../../unified_video_action')
        if os.path.exists(uva_path):
            sys.path.insert(0, uva_path)
            from unified_video_action.codecs.imagecodecs_numcodecs import register_codecs
            # 尝试注册所有codecs（UVA的register_codecs会处理JPEGXL不可用的情况）
            register_codecs(codecs=None, force=False, verbose=False)
            # 验证注册成功
            try:
                get_codec({"id": "imagecodecs_jpegxl"})
                print("✓ Registered imagecodecs_jpegxl codec from UVA")
                return True
            except (ValueError, TypeError):
                pass
    except Exception as e:
        pass  # 继续尝试其他方法
    
    # 方法2: 尝试直接导入imagecodecs并注册
    try:
        from numcodecs.registry import register_codec
        from numcodecs.abc import Codec
        import imagecodecs
        
        # 详细检查JPEGXL支持
        jpegxl_attr = imagecodecs.JPEGXL
        is_stub = str(jpegxl_attr).find('STUB') != -1 or str(type(jpegxl_attr)).find('STUB') != -1
        print(f"Checking imagecodecs: version={getattr(imagecodecs, '__version__', 'unknown')}, JPEGXL={jpegxl_attr}")
        
        if is_stub or not jpegxl_attr:
            # 提供更详细的诊断信息
            if is_stub:
                print(f"⚠ imagecodecs.JPEGXL is a STUB (not compiled with JPEG-XL support)")
                print(f"   You need to install libjxl first, then reinstall imagecodecs:")
                print(f"   1. conda install -c conda-forge libjxl")
                print(f"   2. conda install -c conda-forge imagecodecs --force-reinstall")
                print(f"   Or: conda install -c conda-forge libjxl imagecodecs --force-reinstall")
            else:
                print(f"⚠ imagecodecs.JPEGXL is False or not available")
                print(f"   This means imagecodecs was installed but without JPEG-XL support.")
                print(f"   Try: conda install -c conda-forge libjxl imagecodecs --force-reinstall")
            raise ImportError("imagecodecs.JPEGXL not available")
        
        # 定义JpegXl codec类
        class JpegXl(Codec):
            """JPEG XL codec for numcodecs."""
            codec_id = "imagecodecs_jpegxl"
            
            def __init__(
                self,
                level=None,
                effort=None,
                distance=None,
                lossless=None,
                decodingspeed=None,
                photometric=None,
                planar=None,
                usecontainer=None,
                index=None,
                keeporientation=None,
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
        # 验证注册成功
        get_codec({"id": "imagecodecs_jpegxl"})
        print("✓ Registered imagecodecs_jpegxl codec directly")
        return True
        
    except ImportError as e:
        print(f"⚠ Warning: Could not import imagecodecs: {e}")
        print(f"   The UMI dataset uses JPEG-XL compression which requires imagecodecs with JPEG-XL support.")
        print(f"   Please install it with: pip install imagecodecs")
        print(f"   Note: You may need to install libjxl system library first:")
        print(f"     - Ubuntu/Debian: sudo apt-get install libjxl-dev")
        print(f"     - Or reinstall imagecodecs: pip install --force-reinstall --no-cache-dir imagecodecs")
        return False
    except Exception as e:
        # Check if imagecodecs is installed but JPEGXL is not available
        try:
            import imagecodecs
            jpegxl_attr = imagecodecs.JPEGXL
            is_stub = str(jpegxl_attr).find('STUB') != -1 or str(type(jpegxl_attr)).find('STUB') != -1
            print(f"Debug: imagecodecs imported, version={getattr(imagecodecs, '__version__', 'unknown')}")
            print(f"Debug: imagecodecs.JPEGXL = {jpegxl_attr} (is_stub={is_stub})")
            if is_stub or not jpegxl_attr:
                if is_stub:
                    print(f"⚠ Warning: imagecodecs is installed but JPEG-XL is a STUB (not compiled with support).")
                    print(f"   You need to install libjxl first, then reinstall imagecodecs:")
                    print(f"   1. conda install -c conda-forge libjxl")
                    print(f"   2. conda install -c conda-forge imagecodecs --force-reinstall")
                    print(f"   Or: conda install -c conda-forge libjxl imagecodecs --force-reinstall")
                else:
                    print(f"⚠ Warning: imagecodecs is installed but JPEG-XL support is not available.")
                    print(f"   This usually means libjxl system library is missing or imagecodecs was")
                    print(f"   installed without JPEG-XL support.")
                    print(f"   Solutions:")
                    print(f"     1. Install libjxl: conda install -c conda-forge libjxl")
                    print(f"     2. Reinstall imagecodecs: conda install -c conda-forge imagecodecs --force-reinstall")
                    print(f"     3. Or both: conda install -c conda-forge libjxl imagecodecs --force-reinstall")
        except Exception as import_err:
            print(f"Debug: Could not import imagecodecs for check: {import_err}")
        print(f"⚠ Warning: Could not register imagecodecs_jpegxl codec: {e}")
        import traceback
        traceback.print_exc()
        return False

# 在导入zarr之前注册codec
_codec_registered = _register_jpegxl_codec()


def _axis_angle_to_rot6d(axis_angle):
    """
    Convert axis-angle (3D) to rotation 6D representation.
    axis_angle: (..., 3) numpy array
    Returns: (..., 6) numpy array
    """
    # Convert axis-angle to rotation matrix
    rot = st.Rotation.from_rotvec(axis_angle)
    rot_mat = rot.as_matrix()  # (..., 3, 3)
    
    # Extract first two rows as 6D representation
    batch_dim = rot_mat.shape[:-2]
    rot6d = rot_mat[..., :2, :].reshape(batch_dim + (6,))
    return rot6d


def _build_action_from_robot_data(eef_pos, eef_rot_axis_angle, gripper_width, start_pose=None):
    """
    Build 10D action from robot data.
    Args:
        eef_pos: (3,) - end effector position
        eef_rot_axis_angle: (3,) - end effector rotation (axis-angle)
        gripper_width: (1,) - gripper width
        start_pose: (6,) optional - episode start pose [pos(3), rot_axis_angle(3)]
    Returns:
        action: (10,) - [pos(3), rot_6d(6), gripper(1)]
    """
    # Convert axis-angle to 6D rotation
    rot6d = _axis_angle_to_rot6d(eef_rot_axis_angle.reshape(1, 3))[0]  # (6,)
    
    # If start_pose provided, compute relative action
    if start_pose is not None:
        start_pos = start_pose[:3]
        start_rot_axis_angle = start_pose[3:]
        
        # Relative position
        rel_pos = eef_pos - start_pos
        
        # Relative rotation: start_rot^-1 * current_rot
        start_rot = st.Rotation.from_rotvec(start_rot_axis_angle)
        current_rot = st.Rotation.from_rotvec(eef_rot_axis_angle)
        rel_rot = start_rot.inv() * current_rot
        rel_rot_mat = rel_rot.as_matrix()
        rel_rot6d = rel_rot_mat[:2, :].reshape(6)
        
        action = np.concatenate([rel_pos, rel_rot6d, gripper_width])
    else:
        # Absolute action
        action = np.concatenate([eef_pos, rot6d, gripper_width])
    
    return action.astype(np.float32)


class UmiVideoDataset(Dataset):

    def __init__(
        self,
        dataset_root_dir: str,
        max_condition_frames: int = 2,
        image_size: int = 256,
        split: str = 'train',  # 'train' or 'val'
        dataset_names: Optional[List[str]] = None,  # 数据集名称列表，如 ['cup_arrangement_0', 'towel_folding_0', 'mouse_arrangement_0']
        used_episode_indices_file: Optional[str] = None,  # JSON文件，指定使用的episode索引
        dataset_configs: Optional[Dict[str, Dict[str, Any]]] = None,  # 数据集配置（mask_mirror等）
        **kwargs
    ):
        # Validate and normalize dataset_root_dir
        if not dataset_root_dir:
            raise ValueError(f"dataset_root_dir cannot be empty or None. Received: {repr(dataset_root_dir)}")
        
        dataset_root_dir = os.path.expanduser(dataset_root_dir)
        dataset_root_dir = os.path.normpath(dataset_root_dir)
        dataset_root_dir = os.path.abspath(dataset_root_dir)
        
        if not os.path.exists(dataset_root_dir):
            raise ValueError(f"dataset_root_dir does not exist: {dataset_root_dir}")
        
        if not os.path.isdir(dataset_root_dir):
            raise ValueError(f"dataset_root_dir is not a directory: {dataset_root_dir}")
        
        self.dataset_root_dir = dataset_root_dir
        print(f"Dataset root directory: {self.dataset_root_dir}")
        self.max_condition_frames = max_condition_frames
        self.image_size = image_size
        self.split = split
        
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
            if not dataset_name or not dataset_name.strip():
                print(f"Warning: Empty dataset name, skipping")
                continue
                
            if not dataset_root_dir or not dataset_root_dir.strip():
                raise ValueError(f"dataset_root_dir is empty or None. Provided value: {repr(dataset_root_dir)}")
            
            zarr_path = os.path.join(dataset_root_dir, dataset_name + '.zarr')
            
            # 转换为绝对路径以便调试
            zarr_path = os.path.abspath(zarr_path)
            
            if not zarr_path or not zarr_path.strip():
                raise ValueError(f"Constructed zarr_path is empty. dataset_root_dir={repr(dataset_root_dir)}, dataset_name={repr(dataset_name)}")
            
            if not os.path.exists(zarr_path):
                print(f"Warning: {zarr_path} does not exist, skipping dataset {dataset_name}")
                print(f"  dataset_root_dir: {os.path.abspath(dataset_root_dir)}")
                print(f"  dataset_name: {dataset_name}")
                continue
            
            if not os.path.isdir(zarr_path):
                print(f"Warning: {zarr_path} exists but is not a directory, skipping dataset {dataset_name}")
                continue
            
                print(f"Loading dataset: {dataset_name} from {zarr_path}")
            try:
                # Expand user path and normalize
                zarr_path = os.path.expanduser(zarr_path)
                zarr_path = os.path.normpath(zarr_path)
                zarr_path = os.path.abspath(zarr_path)
                
                # Verify it's a directory
                if not os.path.isdir(zarr_path):
                    print(f"Error: {zarr_path} is not a directory")
                    continue
                
                # Check directory structure
                dir_contents = os.listdir(zarr_path)
                has_zarray_files = any(f.endswith('.zarray') or f.endswith('.zgroup') for f in dir_contents)
                has_zarr_groups = 'data' in dir_contents or 'meta' in dir_contents
                
                if not has_zarray_files and not has_zarr_groups:
                    print(f"Warning: {zarr_path} does not appear to be a valid zarr store")
                    print(f"  Directory contents: {dir_contents[:10]}")
                    continue
                
                # Check zarr structure
                zgroup_file = os.path.join(zarr_path, '.zgroup')
                has_root_zgroup = os.path.exists(zgroup_file)
                data_path = os.path.join(zarr_path, 'data')
                meta_path = os.path.join(zarr_path, 'meta')
                
                print(f"  Checking zarr structure at: {zarr_path}")
                print(f"    Root .zgroup exists: {has_root_zgroup}")
                print(f"    Data directory exists: {os.path.isdir(data_path)}")
                print(f"    Meta directory exists: {os.path.isdir(meta_path)}")
                
                # Try different methods to open zarr store
                zarr_store = None
                try:
                    # Method 1: Standard zarr.open() - works if root .zgroup exists
                    zarr_store = zarr.open(zarr_path, mode='r')
                    print(f"  ✓ Successfully opened with zarr.open()")
                except Exception as e1:
                    print(f"  ✗ zarr.open() failed: {e1}")
                    try:
                        # Method 2: Use DirectoryStore and zarr.group() 
                        # This can work even without root .zgroup if data/meta are accessible
                        store = zarr.DirectoryStore(zarr_path)
                        # zarr.group() will try to open existing group or create new one
                        # But in read mode, we need to be careful
                        # Try opening as a group first
                        zarr_store = zarr.group(store=store)
                        # Verify it can access data and meta
                        if 'data' not in zarr_store:
                            raise KeyError("'data' not found in zarr group")
                        if 'meta' not in zarr_store:
                            raise KeyError("'meta' not found in zarr group")
                        print(f"  ✓ Successfully opened with zarr.group()")
                    except Exception as e2:
                        print(f"  ✗ zarr.group() failed: {e2}")
                        # If both methods fail, the zarr store might be incomplete
                        # Check if we need to create a root .zgroup file
                        if not has_root_zgroup and os.path.isdir(data_path) and os.path.isdir(meta_path):
                            raise RuntimeError(
                                f"Zarr store at {zarr_path} appears to be missing a root .zgroup file. "
                                f"The data/ and meta/ directories exist, but zarr cannot open the store "
                                f"without a root group marker. You may need to create a .zgroup file "
                                f"in the root directory, or the zarr store may be incomplete. "
                                f"Original errors: zarr.open()={e1}, zarr.group()={e2}"
                            )
                        else:
                            raise RuntimeError(
                                f"Failed to open zarr store at {zarr_path}. "
                                f"Errors: zarr.open()={e1}, zarr.group()={e2}"
                            )
                
                if zarr_store is None:
                    raise ValueError(f"zarr.open() returned None for path: {zarr_path}")
                    
            except Exception as e:
                print(f"Error opening zarr store at {zarr_path}: {e}")
                print(f"  Path type: {type(zarr_path)}, Path value: {repr(zarr_path)}")
                print(f"  Path exists: {os.path.exists(zarr_path)}")
                print(f"  Is directory: {os.path.isdir(zarr_path) if os.path.exists(zarr_path) else False}")
                import traceback
                traceback.print_exc()
                raise
            
            # 检查数据格式
            if 'data' not in zarr_store or 'camera0_rgb' not in zarr_store['data']:
                print(f"Warning: {zarr_path} does not have data/camera0_rgb, skipping")
                continue
            
            # 在访问zarr数组之前，确保codec已注册
            try:
                from numcodecs.registry import get_codec
                get_codec({"id": "imagecodecs_jpegxl"})
            except (ValueError, TypeError):
                # 提供详细的错误信息和解决方案
                error_msg = (
                    f"imagecodecs_jpegxl codec is not registered but is required to read the UMI dataset.\n"
                    f"\n"
                    f"The UMI dataset uses JPEG-XL compression. You need imagecodecs with JPEG-XL support.\n"
                    f"\n"
                    f"Solutions:\n"
                    f"1. Install libjxl system library first:\n"
                    f"   Ubuntu/Debian: sudo apt-get install libjxl-dev libjxl-tools\n"
                    f"   Then reinstall imagecodecs: pip install --force-reinstall --no-cache-dir imagecodecs\n"
                    f"\n"
                    f"2. Or use conda (recommended):\n"
                    f"   conda install -c conda-forge imagecodecs\n"
                    f"\n"
                    f"3. Check if imagecodecs has JPEG-XL support:\n"
                    f"   python -c 'import imagecodecs; print(imagecodecs.JPEGXL)'\n"
                    f"   Should print True, not False or raise an error.\n"
                )
                raise RuntimeError(error_msg)
            
            images = zarr_store['data']['camera0_rgb']  # (N, 224, 224, 3) uint8
            
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
            
            # 尝试加载action数据（如果存在预处理的action）
            actions = None
            if 'data' in zarr_store and 'action' in zarr_store['data']:
                actions = zarr_store['data']['action']  # (N, 10) float32
                print(f"  Found preprocessed action data in {dataset_name}")
            else:
                # 从原始数据构建action: robot0_eef_pos, robot0_eef_rot_axis_angle, robot0_gripper_width
                if 'data' in zarr_store:
                    has_eef_pos = 'robot0_eef_pos' in zarr_store['data']
                    has_eef_rot = 'robot0_eef_rot_axis_angle' in zarr_store['data']
                    has_gripper = 'robot0_gripper_width' in zarr_store['data']
                    
                    if has_eef_pos and has_eef_rot and has_gripper:
                        print(f"  Building action from raw data: robot0_eef_pos, robot0_eef_rot_axis_angle, robot0_gripper_width")
                        # 将在__getitem__中动态构建，这里只标记
                        actions = 'build_from_raw'  # 标记需要构建
                    else:
                        print(f"  Warning: Missing robot data in {dataset_name}")
                        print(f"    robot0_eef_pos: {has_eef_pos}, robot0_eef_rot_axis_angle: {has_eef_rot}, robot0_gripper_width: {has_gripper}")
                else:
                    print(f"  Warning: No action data found in {dataset_name}, action prediction will be disabled")
            
            # 加载原始robot数据（用于构建action）
            robot_data = {}
            if 'data' in zarr_store:
                if 'robot0_eef_pos' in zarr_store['data']:
                    robot_data['eef_pos'] = zarr_store['data']['robot0_eef_pos']  # (N, 3)
                if 'robot0_eef_rot_axis_angle' in zarr_store['data']:
                    robot_data['eef_rot_axis_angle'] = zarr_store['data']['robot0_eef_rot_axis_angle']  # (N, 3)
                if 'robot0_gripper_width' in zarr_store['data']:
                    robot_data['gripper_width'] = zarr_store['data']['robot0_gripper_width']  # (N, 1)
                if 'robot0_demo_start_pose' in zarr_store['data']:
                    robot_data['demo_start_pose'] = zarr_store['data']['robot0_demo_start_pose']  # (episodes, 6)
            
            self.zarr_stores.append({
                'store': zarr_store,
                'images': images,
                'actions': actions,  # 预处理的action或'build_from_raw'
                'robot_data': robot_data,  # 原始robot数据
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
        """创建所有有效的 (dataset_idx, episode_idx, frame_idx) 索引"""
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
        actions = zarr_data.get('actions', None)  # 获取action数据（如果存在）
        robot_data = zarr_data.get('robot_data', {})
        episode_ends = zarr_data['episode_ends']
        episode_starts = [0] + list(episode_ends[:-1])
        ep_start = episode_starts[ep_idx]
        
        # 获取条件帧
        condition_frames = []
        for i in range(self.max_condition_frames):
            cond_frame_idx = frame_idx - self.max_condition_frames + i
            cond_frame = images[cond_frame_idx]  # (H, W, 3) uint8
            condition_frames.append(cond_frame)
        condition_frames = np.stack(condition_frames)  # (max_condition_frames, H, W, 3)
        
        # 获取目标帧
        target_frame = images[frame_idx]  # (H, W, 3) uint8
        
        # 获取对应的action（如果存在）
        action = None
        if actions is not None:
            if isinstance(actions, str) and actions == 'build_from_raw':
                # 从原始数据构建action
                if 'eef_pos' in robot_data and 'eef_rot_axis_angle' in robot_data and 'gripper_width' in robot_data:
                    eef_pos = robot_data['eef_pos'][frame_idx]  # (3,)
                    eef_rot_axis_angle = robot_data['eef_rot_axis_angle'][frame_idx]  # (3,)
                    gripper_width = robot_data['gripper_width'][frame_idx]  # (1,)
                    
                    # 获取episode start pose（如果存在）
                    start_pose = None
                    if 'demo_start_pose' in robot_data:
                        start_pose = robot_data['demo_start_pose'][ep_idx]  # (6,)
                    
                    action = _build_action_from_robot_data(
                        eef_pos, eef_rot_axis_angle, gripper_width, start_pose
                    )
                    action = torch.from_numpy(action).float()
            else:
                # 使用预处理的action
                action = actions[frame_idx]  # (10,) float32
                action = torch.from_numpy(action).float()
        
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
        
        # 如果action数据存在，添加到结果中
        if action is not None:
            result['action'] = action  # (10,)

        return result