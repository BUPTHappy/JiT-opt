#!/usr/bin/env python3
"""
测试UMI Multi-Task数据集加载
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dataset.umi_video_dataset import UmiVideoDataset
import torch
from torch.utils.data import DataLoader

def test_dataset():
    # 数据路径（根据你的实际路径修改）
    data_path = "/workspace/data/umi"  # 修改为你的实际路径
    
    # 数据集名称
    dataset_names = [
        'cup_arrangement_0',
        'towel_folding_0',
        'mouse_arrangement_0'
    ]
    
    print("=" * 60)
    print("Testing UMI Multi-Task Dataset Loading")
    print("=" * 60)
    
    try:
        print(f"\n1. Creating dataset...")
        print(f"   Data path: {data_path}")
        print(f"   Dataset names: {dataset_names}")
        
        dataset = UmiVideoDataset(
            dataset_root_dir=data_path,
            dataset_names=dataset_names,
            max_condition_frames=2,
            image_size=256,
            split='train'
        )
        
        print(f"\n✓ Dataset created successfully!")
        print(f"   Total samples: {len(dataset):,}")
        
        # 测试获取单个样本
        print(f"\n2. Testing single sample...")
        sample = dataset[0]
        print(f"   Condition frames shape: {sample['condition_frames'].shape}")
        print(f"   Target frame shape: {sample['target_frame'].shape}")
        print(f"   Condition frames range: [{sample['condition_frames'].min():.2f}, {sample['condition_frames'].max():.2f}]")
        print(f"   Target frame range: [{sample['target_frame'].min():.2f}, {sample['target_frame'].max():.2f}]")
        
        # 测试DataLoader
        print(f"\n3. Testing DataLoader...")
        dataloader = DataLoader(
            dataset, 
            batch_size=4, 
            shuffle=False, 
            num_workers=0,  # 先用0避免多进程问题
            pin_memory=False
        )
        
        batch = next(iter(dataloader))
        print(f"   Batch condition_frames shape: {batch['condition_frames'].shape}")
        print(f"   Batch target_frame shape: {batch['target_frame'].shape}")
        print(f"   Expected: condition_frames (4, 2, 3, 256, 256), target_frame (4, 3, 256, 256)")
        
        # 验证形状
        assert batch['condition_frames'].shape == (4, 2, 3, 256, 256), \
            f"Wrong condition_frames shape: {batch['condition_frames'].shape}"
        assert batch['target_frame'].shape == (4, 3, 256, 256), \
            f"Wrong target_frame shape: {batch['target_frame'].shape}"
        
        print(f"\n✓ All tests passed!")
        print(f"\n4. Dataset statistics:")
        print(f"   Total samples: {len(dataset):,}")
        print(f"   Samples per batch (batch_size=4): {len(dataloader)} batches")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_dataset()
    sys.exit(0 if success else 1)
