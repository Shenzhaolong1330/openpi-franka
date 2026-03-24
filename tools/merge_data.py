#!/usr/bin/env python3
"""Merge multiple LeRobot datasets into one."""

import os
os.environ["HF_HUB_OFFLINE"] = "1"  # 禁用网络请求
os.environ["HF_HOME"] = "/vepfs-mlp2/c20250510/250303034/workspace/.cache/huggingface"  # 设置缓存目录
os.environ["HF_DATASETS_CACHE"] = "/vepfs-mlp2/c20250510/250303034/workspace/.cache/huggingface/datasets"

from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets

# 数据集根目录
DATA_ROOT = Path("/vepfs-mlp2/c20250510/250303034/workspace/data/dual_dobot_010/dual_dobot")

# 要合并的数据集名称列表
DATASET_NAMES = [
    "milk_shake_3_16_step2_20260316_v02",
    "milk_shake_3_16_step3_20260316_v01",
    "milk_shake_3_16_step3_20260316_v02",
    "milk_shake_3_16_step4_20260316_v01",
    "milk_shake_3_16_step5_20260316_v01",
    "milk_shake_3_18_step1_20260318_v01",
    "milk_shake_3_18_step1_20260318_v02",
    "milk_shake_3_19_step1_20260319_v01",
    "milk_shake_3_19_step1_20260319_v02",
    "milk_shake_3_19_step1_20260319_v03",
    "milk_shake_3_19_step1_20260319_v04",
    "milk_shake_3_19_step1_20260319_v05",
    "milk_shake_3_19_step1_20260319_v06",
    "milk_shake_3_19_step1_20260319_v07",
    "milk_shake_3_19_step2_20260319_v01",
    "milk_shake_3_19_step2_20260319_v02",
    "milk_shake_3_19_step2_20260319_v03",
    "milk_shake_3_19_step2_20260319_v04",
    "milk_shake_3_19_step2_20260319_v05",
    "milk_shake_3_19_step2_20260319_v06",
    "milk_shake_3_19_step3_20260319_v01",
    "milk_shake_3_19_step3_20260319_v02",
    "milk_shake_3_19_step3_20260319_v01",
    "milk_shake_3_19_step3_20260319_v02",
    "milk_shake_3_19_step3_20260319_v03",
    "milk_shake_3_19_step3_20260319_v04",
    "milk_shake_3_19_step3_20260319_v05",
    "milk_shake_3_19_step3_20260319_v06",
    "milk_shake_3_19_step4_20260319_v02",
    "milk_shake_3_19_step4_20260319_v03",
    "milk_shake_3_19_step4_20260319_v04",
    "milk_shake_3_19_step4_20260319_v05",
    "milk_shake_3_19_step4_20260319_v06",
    "milk_shake_3_19_step4_20260319_v07",
    "milk_shake_3_19_step4_20260319_v08",
    "milk_shake_3_19_step5_20260319_v01",
    "milk_shake_3_19_step5_20260319_v02",
    "milk_shake_3_19_step5_20260319_v03",
]

# 输出数据集名称
OUTPUT_NAME = "milk_shake_merged"

def main():
    print(f"Loading {len(DATASET_NAMES)} datasets...")
    
    # 加载所有数据集
    datasets = []
    for name in DATASET_NAMES:
        dataset_path = DATA_ROOT / name
        if not dataset_path.exists():
            print(f"Warning: {name} not found, skipping...")
            continue
        print(f"Loading: {name}")
        try:
            ds = LeRobotDataset(repo_id=name, root=dataset_path)
            datasets.append(ds)
        except Exception as e:
            print(f"Error loading {name}: {e}")
            print(f"Skipping {name}...")
            continue
    
    print(f"\nMerging {len(datasets)} datasets...")
    
    # 合并数据集
    merged = merge_datasets(
        datasets=datasets,
        output_repo_id=OUTPUT_NAME,
        output_dir=DATA_ROOT / OUTPUT_NAME,
    )
    
    print(f"\nMerged dataset created:")
    print(f"  - Total episodes: {merged.num_episodes}")
    print(f"  - Total frames: {merged.num_frames}")
    print(f"  - Output path: {DATA_ROOT / OUTPUT_NAME}")

if __name__ == "__main__":
    main()