"""
快速诊断 PushT 数据加载问题
用法: python debug_pusht_data.py --data_path /workspace/pusht_data/pusht --dataset_names pusht_cchi_v7_replay
"""
import zarr
import numpy as np
import argparse
import torch
from PIL import Image

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset_names', type=str, default='pusht_cchi_v7_replay')
    args = parser.parse_args()

    zarr_path = f"{args.data_path}/{args.dataset_names}.zarr"
    print(f"Opening zarr: {zarr_path}")
    store = zarr.open(zarr_path, mode='r')

    # 1. 检查 zarr 结构
    print("\n=== Zarr structure ===")
    print(f"Top-level keys: {list(store.keys())}")
    if 'data' in store:
        print(f"data/ keys: {list(store['data'].keys())}")
    if 'meta' in store:
        print(f"meta/ keys: {list(store['meta'].keys())}")

    # 2. 检查图像数组
    if 'data' in store and 'img' in store['data']:
        images = store['data']['img']
        print(f"\n=== Image array info ===")
        print(f"Shape: {images.shape}")
        print(f"Dtype: {images.dtype}")
        print(f"Chunks: {images.chunks}")

        # 读取第一帧
        frame0 = np.array(images[0])
        print(f"\n=== First frame (images[0]) ===")
        print(f"Shape: {frame0.shape}")
        print(f"Dtype: {frame0.dtype}")
        print(f"Min: {frame0.min()}, Max: {frame0.max()}, Mean: {frame0.mean():.2f}")
        print(f"Value range check:")
        if frame0.dtype == np.uint8:
            print(f"  ✓ uint8 [0, 255] - normalization /127.5-1.0 is correct")
        elif frame0.dtype in (np.float32, np.float64):
            if frame0.max() <= 1.0:
                print(f"  ✗ float [0, 1] - normalization /127.5-1.0 is WRONG!")
                print(f"    Should use: *2.0 - 1.0")
            elif frame0.max() <= 255.0:
                print(f"  ⚠ float [0, 255] - normalization /127.5-1.0 works but inefficient")
            else:
                print(f"  ? Unknown range, check manually")
        else:
            print(f"  ? Unknown dtype: {frame0.dtype}")

        # 模拟当前归一化
        frame_float = torch.from_numpy(frame0).float()
        frame_norm = frame_float / 127.5 - 1.0
        print(f"\n=== After current normalization (/127.5 - 1.0) ===")
        print(f"Min: {frame_norm.min():.4f}, Max: {frame_norm.max():.4f}, Mean: {frame_norm.mean():.4f}")
        if frame_norm.min() < -1.1 or frame_norm.max() > 1.1:
            print(f"  ✗ Values out of [-1, 1] range!")
        elif frame_norm.max() - frame_norm.min() < 0.1:
            print(f"  ✗ Almost no contrast! All values ~{frame_norm.mean():.4f}")
            print(f"    This explains gray output - model sees constant input")
        else:
            print(f"  ✓ Values look reasonable")

        # 保存原始帧为图片
        if frame0.dtype == np.uint8 and len(frame0.shape) == 3:
            Image.fromarray(frame0).save("debug_pusht_raw_frame0.png")
            print(f"\nSaved raw frame to: debug_pusht_raw_frame0.png")
        elif frame0.dtype in (np.float32, np.float64) and frame0.max() <= 1.0:
            img = (frame0 * 255).astype(np.uint8)
            Image.fromarray(img).save("debug_pusht_raw_frame0.png")
            print(f"\nSaved raw frame (float→uint8) to: debug_pusht_raw_frame0.png")

        # 读取多帧检查一致性
        print(f"\n=== Sampling more frames ===")
        for idx in [0, 100, 1000, 10000]:
            if idx < images.shape[0]:
                f = np.array(images[idx])
                print(f"  Frame {idx}: dtype={f.dtype}, min={f.min()}, max={f.max()}, mean={f.mean():.2f}")

    # 3. 检查 episode_ends
    if 'meta' in store and 'episode_ends' in store['meta']:
        ep_ends = store['meta']['episode_ends'][:]
        print(f"\n=== Episode info ===")
        print(f"Number of episodes: {len(ep_ends)}")
        print(f"First 5 episode_ends: {ep_ends[:5]}")
        print(f"Total frames: {ep_ends[-1]}")
        ep_lengths = np.diff(np.concatenate([[0], ep_ends]))
        print(f"Episode lengths: min={ep_lengths.min()}, max={ep_lengths.max()}, mean={ep_lengths.mean():.1f}")

if __name__ == '__main__':
    main()
