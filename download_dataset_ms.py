#!/usr/bin/env python3
"""尝试从 ModelScope 下载 ManyICLBench 数据集"""

from modelscope.msdatasets import MsDataset

try:
    print("📥 尝试从 ModelScope 下载 ManyICLBench...")
    # ManyICLBench 在 ModelScope 上的名称可能不同
    ds = MsDataset.load(
        'launch/ManyICLBench',
        subset_name='ARC-Challenge',
        split='seed0'
    )
    print(f"✅ 数据集下载完成！样本数: {len(ds)}")
except Exception as e:
    print(f"⚠️ ModelScope 下载失败: {e}")
    print("建议：使用本地 JSONL 数据进行测试")
