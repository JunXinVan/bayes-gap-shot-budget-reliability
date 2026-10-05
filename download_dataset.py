#!/usr/bin/env python3
"""下载 ManyICLBench 数据集（使用 HF-Mirror）"""

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from datasets import load_dataset

print("📥 下载 ManyICLBench 数据集 (ARC-Challenge, seed0)...")
ds = load_dataset(
    "launch/ManyICLBench",
    "ARC-Challenge",
    split="seed0",
    trust_remote_code=True
)
print(f"✅ 数据集下载完成！")
print(f"   样本数: {len(ds)}")
print(f"   字段: {list(ds[0].keys())[:10]}")
