"""
Tiny Transformer on REAL Datasets (ARC, HellaSwag)
NOT synthetic data - uses actual benchmark datasets.

Design:
- Model: Tiny Transformer (37K params, 2 layers)
- Training: Use k <= k_train demonstrations, predict answer index
- Evaluation: Test with k > k_train to observe OverPrompting
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from overprompting_exp.tasks.manyiclbench import ManyICLBenchConfig, ManyICLBenchTask


class TinyTransformer(nn.Module):
    """Transformer for real-data ICL; size configurable via constructor args."""
    
    def __init__(self, vocab_size=256, d_model=64, n_layers=4, n_heads=4, max_len=512, dropout=0.1, n_classes=4):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, n_classes)
        
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def char_tokenize(text: str, vocab_size: int = 256) -> List[int]:
    """Simple character-level tokenization."""
    tokens = [ord(c) % vocab_size for c in text[:400]]
    return tokens


def load_arc_dataset(n_examples: int = 200, seed: int = 42):
    """Load REAL ARC-Challenge dataset."""
    cfg = ManyICLBenchConfig(
        task_name="ARC-Challenge",
        split="seed0",
        prefix_key="8k",
        max_examples=n_examples,
        candidates=["A", "B", "C", "D"]
    )
    
    task = ManyICLBenchTask(cfg, seed=seed)
    task.load()
    
    batches = list(task.iter_batches())
    batch = batches[0]
    
    dataset = []
    for i, ex in enumerate(batch.examples):
        demos = [
            {'question': d.input_text, 'answer': d.output_text}
            for d in batch.demo_pools[i]
        ]
        dataset.append({
            'question': ex.input_text,
            'gold': ex.gold,  # e.g., "A"
            'gold_idx': ord(ex.gold) - ord('A'),  # 0, 1, 2, or 3
            'candidates': ex.candidates,
            'demonstrations': demos
        })
    
    return dataset


def build_prompt(example: Dict, k: int) -> str:
    """Build prompt with k demonstrations."""
    if k == 0:
        return f"Question: {example['question']}\nAnswer:"
    
    demos = example['demonstrations'][:k]
    demo_texts = [f"Question: {d['question']}\nAnswer: {d['answer']}" for d in demos]
    prompt = "\n\n".join(demo_texts)
    prompt += f"\n\nQuestion: {example['question']}\nAnswer:"
    return prompt


def train_model(
    dataset: List[Dict],
    k_train: int = 4,
    epochs: int = 100,
    device: str = "cuda",
    model_config: Dict | None = None,
):
    """Train Transformer on real ARC data. model_config: vocab_size, d_model, n_layers, n_heads, max_len, dropout."""
    if model_config is None:
        model_config = {}
    model = TinyTransformer(
        vocab_size=model_config.get("vocab_size", 256),
        d_model=model_config.get("d_model", 64),
        n_layers=model_config.get("n_layers", 4),
        n_heads=model_config.get("n_heads", 4),
        max_len=model_config.get("max_len", 512),
        dropout=model_config.get("dropout", 0.1),
        n_classes=model_config.get("n_classes", 4),
    ).to(device)
    print(f"Model: {model.count_parameters():,} parameters")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-3, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss()
    
    # Use subset for training
    train_size = min(150, len(dataset))
    train_data = dataset[:train_size]
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        np.random.shuffle(train_data)
        
        for example in train_data:
            # Random k for training (0 to k_train)
            k = np.random.randint(0, k_train + 1)
            prompt = build_prompt(example, k)
            tokens = char_tokenize(prompt)
            
            if len(tokens) < 5:
                continue
            
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            label = torch.tensor([example['gold_idx']], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            output = model(input_tensor)
            
            # Predict from last position
            loss = criterion(output[:, -1, :], label)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pred = output[:, -1, :].argmax(dim=-1)
            correct += (pred == label).sum().item()
            total += 1
        
        if epoch % 10 == 0:
            acc = correct / max(total, 1)
            print(f"Epoch {epoch:3d}: Loss={total_loss/max(total,1):.4f}, Acc={acc:.4f}")
    
    return model


def evaluate_model(model: nn.Module, dataset: List[Dict], k: int, device: str = "cuda") -> float:
    """Evaluate with exactly k demonstrations."""
    model.eval()
    correct = 0
    total = 0
    
    # Use held-out test set
    test_data = dataset[150:200]
    
    with torch.no_grad():
        for example in test_data:
            prompt = build_prompt(example, k)
            tokens = char_tokenize(prompt)
            
            if len(tokens) < 5:
                continue
            
            input_tensor = torch.tensor([tokens], dtype=torch.long).to(device)
            label = example['gold_idx']
            
            output = model(input_tensor)
            pred = output[:, -1, :].argmax(dim=-1).item()
            
            correct += (pred == label)
            total += 1
    
    return correct / max(total, 1)


def run_experiment(
    k_train: int = 4,
    test_k_values: List[int] = None,
    output_dir: str = "results/tiny_transformer_real",
    dataset: List[Dict] | None = None,
    n_examples: int = 200,
    model_config: Dict | None = None,
    epochs: int = 50,
):
    """Run complete experiment on REAL ARC data.

    If dataset is provided, it is used as-is (no reload). Otherwise load via load_arc_dataset(n_examples).
    model_config: optional dict with d_model, n_layers, n_heads, vocab_size, max_len, dropout, n_classes.
    Fails loudly if real data cannot be loaded; no synthetic fallback.
    """
    if test_k_values is None:
        test_k_values = [0, 1, 2, 4, 8, 16]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 70)
    print("TINY TRANSFORMER ON REAL ARC-CHALLENGE DATA")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Training k: 0-{k_train}")
    print(f"Test k values: {test_k_values}")
    print("=" * 70)
    
    # Load real ARC dataset (or use provided dataset)
    if dataset is not None:
        print("\nUsing provided REAL ARC-Challenge dataset (no reload).")
    else:
        print("\nLoading REAL ARC-Challenge dataset...")
        dataset = load_arc_dataset(n_examples=n_examples)
    print(f"Loaded {len(dataset)} examples with demonstrations")
    print(f"Example: {dataset[0]['question'][:80]}...")
    print(f"Gold answer: {dataset[0]['gold']}")
    print(f"Number of demos available: {len(dataset[0]['demonstrations'])}")
    
    # Train
    print(f"\nTraining on k <= {k_train}...")
    model = train_model(
        dataset,
        k_train=k_train,
        epochs=epochs,
        device=device,
        model_config=model_config,
    )
    
    # Evaluate
    print("\n" + "=" * 70)
    print("EVALUATION ON HELD-OUT TEST SET")
    print("=" * 70)
    
    results = []
    for k in test_k_values:
        acc = evaluate_model(model, dataset, k=k, device=device)
        results.append({"k": k, "accuracy": acc, "train_k": k_train})
        print(f"k={k:2d}: {acc*100:5.2f}%")
    
    # Save
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"arc_real_k{k_train}_results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    torch.save(model.state_dict(), output_path / "arc_real_model.pt")
    
    print(f"\nResults saved to {results_file}")
    return results


if __name__ == "__main__":
    run_experiment(k_train=4, test_k_values=[0, 1, 2, 4, 8, 12, 16, 24])
