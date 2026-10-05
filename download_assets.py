#!/usr/bin/env python3
"""
预下载模型和数据集脚本（适用于国内无代理环境）
使用 ModelScope 镜像下载模型
"""

import os
import sys
from pathlib import Path

# 设置缓存目录
HF_HOME = Path("/root/autodl-tmp/hf_cache")
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_DATASETS_CACHE"] = str(HF_HOME / "datasets")
os.environ["TRANSFORMERS_CACHE"] = str(HF_HOME / "transformers")
os.environ["MODELSCOPE_CACHE"] = str(HF_HOME / "modelscope")

def download_model_modelscope(model_id: str, local_dir: str = None):
    """使用 ModelScope 下载模型"""
    try:
        from modelscope import snapshot_download
        print(f"\n{'='*60}")
        print(f"📥 下载模型: {model_id}")
        print(f"{'='*60}")
        
        # ModelScope 上的模型 ID 映射
        ms_model_id = model_id
        if model_id.startswith("Qwen/"):
            ms_model_id = model_id  # ModelScope 上 Qwen 模型命名一致
        
        cache_dir = snapshot_download(
            ms_model_id,
            cache_dir=os.environ["MODELSCOPE_CACHE"],
            local_dir=local_dir,
        )
        print(f"✅ 模型下载完成: {cache_dir}")
        return cache_dir
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def download_model_hf_mirror(model_id: str):
    """使用 HF-Mirror 下载模型（备选）"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"\n{'='*60}")
        print(f"📥 使用 HF-Mirror 下载: {model_id}")
        print(f"{'='*60}")
        
        # 使用 hf-mirror.com 镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype="auto",
            device_map="cpu"  # 先下载到 CPU，不加载
        )
        print(f"✅ 模型下载完成")
        return True
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def download_dataset_hf_mirror(dataset_name: str, config_name: str = None, split: str = None):
    """使用 HF-Mirror 下载数据集"""
    try:
        from datasets import load_dataset
        print(f"\n{'='*60}")
        print(f"📥 下载数据集: {dataset_name}" + (f" ({config_name})" if config_name else ""))
        print(f"{'='*60}")
        
        # 使用 hf-mirror.com 镜像
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        ds = load_dataset(
            dataset_name,
            config_name,
            split=split,
            trust_remote_code=True
        )
        print(f"✅ 数据集下载完成")
        if hasattr(ds, '__len__'):
            print(f"   样本数: {len(ds)}")
        return ds
    except Exception as e:
        print(f"❌ 下载失败: {e}")
        return None

def download_manyiclbench():
    """下载 ManyICLBench 数据集"""
    # ManyICLBench 在 Hugging Face 上
    return download_dataset_hf_mirror(
        "launch/ManyICLBench",
        config_name="ARC-Challenge",
        split="seed0"
    )

def main():
    print("🚀 开始预下载模型和数据集")
    print(f"📁 缓存目录: {HF_HOME}")
    
    # 创建缓存目录
    HF_HOME.mkdir(parents=True, exist_ok=True)
    (HF_HOME / "modelscope").mkdir(exist_ok=True)
    (HF_HOME / "datasets").mkdir(exist_ok=True)
    (HF_HOME / "transformers").mkdir(exist_ok=True)
    
    results = {}
    
    # 1. 下载 Qwen2.5-7B 模型 (ModelScope)
    print("\n" + "="*60)
    print("步骤 1/2: 下载 Qwen2.5-7B-Instruct 模型")
    print("="*60)
    
    model_cache = download_model_modelscope("Qwen/Qwen2.5-7B-Instruct")
    results["Qwen2.5-7B-Instruct"] = model_cache is not None
    
    # 如果 ModelScope 失败，尝试 HF-Mirror
    if not results["Qwen2.5-7B-Instruct"]:
        print("\n⚠️ ModelScope 下载失败，尝试 HF-Mirror...")
        results["Qwen2.5-7B-Instruct"] = download_model_hf_mirror("Qwen/Qwen2.5-7B-Instruct")
    
    # 2. 下载 ManyICLBench 数据集 (HF-Mirror)
    print("\n" + "="*60)
    print("步骤 2/2: 下载 ManyICLBench 数据集")
    print("="*60)
    
    ds = download_manyiclbench()
    results["ManyICLBench-ARC-Challenge"] = ds is not None
    
    # 总结
    print("\n" + "="*60)
    print("📊 下载结果总结")
    print("="*60)
    for name, success in results.items():
        status = "✅ 成功" if success else "❌ 失败"
        print(f"{status}: {name}")
    
    # 检查缓存目录大小
    print(f"\n📁 缓存目录使用情况:")
    if HF_HOME.exists():
        import subprocess
        result = subprocess.run(
            ["du", "-sh", str(HF_HOME)],
            capture_output=True,
            text=True
        )
        print(f"   {result.stdout.strip()}")
    
    all_success = all(results.values())
    if all_success:
        print("\n🎉 所有资源下载成功！可以开始运行实验了。")
        return 0
    else:
        print("\n⚠️ 部分资源下载失败，请检查错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
