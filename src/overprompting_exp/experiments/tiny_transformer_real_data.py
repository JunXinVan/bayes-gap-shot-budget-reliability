"""
Tiny Transformer on REAL Datasets (ARC, HellaSwag)
NOT synthetic data - uses actual benchmark datasets.

This recreates the paper's Figure 2 experiment but with real data.
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
import yaml
from typing import Dict, List, Tuple

class TinyTransformer(nn.Module):
    """Tiny Transformer: 37K parameters, 2 layers."""
    
    def __init__(self, vocab_size=1000, d_model=32, n_layers=2, n_heads=2, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output(x)
    
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def load_real_dataset(dataset_name: str, n_examples: int = 200, seed: int = 42):
    """
    Load REAL dataset (NOT synthetic).
    Uses the ManyICLBench dataset loader from the main experiments.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from overprompting_exp.tasks.manyiclbench import ManyICLBenchConfig, ManyICLBenchTask
    
    # Map dataset name to ManyICLBench task name
    task_map = {
        "arc_challenge": "ARC-Challenge",
        "hellaswag": "Hellaswag",
        "arc_easy": "ARC-Easy",
    }
    
    task_name = task_map.get(dataset_name, dataset_name)
    
    cfg = ManyICLBenchConfig(
        task_name=task_name,
        split="seed0",
        prefix_key="8k",  # Use 8k context
        max_examples=n_examples,
        candidates=["A", "B", "C", "D"] if "ARC" in task_name else None
    )
    
    task = ManyICLBenchTask(cfg, seed=seed)
    task.load()
    
    # Extract examples and demo pool
    batches = list(task.iter_batches())
    if not batches:
        raise ValueError(f"No data loaded for {dataset_name}")
    
    batch = batches[0]
    
    # Convert to simple format
    dataset = []
    for i, ex in enumerate(batch.examples):
        dataset.append({
            'question': ex.input_text,
            'gold': ex.gold,
            'candidates': ex.candidates,
            'example_id': ex.example_id,
            'demonstrations': [
                {'question': d.input_text, 'answer': d.output_text}
                for d in batch.demo_pools[i]
            ]
        })
    
    return dataset, task


def create_in_context_example(example: Dict, k: int, tokenizer=None) -> Tuple[torch.Tensor, int]:
    """
    Create in-context learning example with k demonstrations.
    Returns (input_ids, label_idx).
    """
    # Simple tokenization (character-level for simplicity)
    def simple_tokenize(text: str, vocab_size: int = 1000) -> List[int]:
        # Map characters to indices
        tokens = []
        for c in text[:500]:  # Limit length
            tokens.append(ord(c) % vocab_size)
        return tokens
    
    # Build prompt with k demonstrations
    if k == 0:
        prompt = f"Question: {example['question']}\nAnswer:"
    else:
        # Use k demonstrations from the example
        demonstrations = example.get('demonstrations', [])
        demo_texts = []
        for i in range(min(k, len(demonstrations))):
            demo = demonstrations[i]
            demo_texts.append(f"Question: {demo['question']}\nAnswer: {demo['answer']}")
        
        prompt = "\n\n".join(demo_texts)
        prompt += f"\n\nQuestion: {example['question']}\nAnswer:"
    
    # Tokenize
    input_ids = simple_tokenize(prompt)
    input_tensor = torch.tensor(input_ids, dtype=torch.long)
    
    # Label
    label_idx = example.get('label_idx', 0)
    
    return input_tensor, label_idx


def train_on_real_data(
    dataset_name: str = "arc_challenge",
    k_train: int = 4,
    n_epochs: int = 50,
    lr: float = 1e-3,
    batch_size: int = 16,
    device: str = "cuda"
):
    """
    Train Tiny Transformer on REAL dataset.
    """
    print(f"Loading REAL dataset: {dataset_name}")
    print("=" * 60)
    
    # Load real dataset
    dataset, backend = load_real_dataset(dataset_name, n_examples=200)
    print(f"Loaded {len(dataset)} examples")
    
    # Create model
    model = TinyTransformer(
        vocab_size=1000,
        d_model=32,
        n_layers=2,
        n_heads=2,
        max_len=512
    ).to(device)
    
    print(f"Model parameters: {model.count_parameters():,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Training
    model.train()
    for epoch in range(n_epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        # Sample batch
        indices = np.random.choice(len(dataset), size=batch_size, replace=False)
        
        for idx in indices:
            example = dataset[idx]
            input_tensor, label_idx = create_in_context_example(example, k=k_train)
            
            if len(input_tensor) == 0:
                continue
                
            input_tensor = input_tensor.unsqueeze(0).to(device)
            label_tensor = torch.tensor([label_idx], dtype=torch.long).to(device)
            
            optimizer.zero_grad()
            output = model(input_tensor)
            
            # Use last position for prediction
            loss = criterion(output[:, -1, :], label_tensor)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output[:, -1, :].argmax(dim=-1)
            correct += (pred == label_tensor).sum().item()
            total += 1
        
        if epoch % 10 == 0:
            acc = correct / max(total, 1)
            print(f"Epoch {epoch:3d}: Loss={total_loss/max(total,1):.4f}, Acc={acc:.4f}")
    
    return model, backend, dataset


def evaluate_k_shot_real(
    model: nn.Module,
    backend,
    dataset,
    k_test: int,
    n_samples: int = 200,
    device: str = "cuda"
) -> float:
    """
    Evaluate on REAL data with k-shot demonstrations.
    """
    model.eval()
    correct = 0
    total = 0
    
    # Sample test examples
    test_indices = np.random.choice(len(dataset), size=min(n_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for idx in test_indices:
            example = dataset[idx]
            input_tensor, label_idx = create_in_context_example(example, k=k_test)
            
            if len(input_tensor) == 0:
                continue
                
            input_tensor = input_tensor.unsqueeze(0).to(device)
            label_tensor = torch.tensor([label_idx], dtype=torch.long).to(device)
            
            output = model(input_tensor)
            pred = output[:, -1, :].argmax(dim=-1)
            correct += (pred == label_tensor).sum().item()
            total += 1
    
    return correct / max(total, 1)


def run_experiment(
    dataset_name: str = "arc_challenge",
    k_train: int = 4,
    test_k_values: List[int] = None,
    output_dir: str = "results/tiny_transformer_real"
):
    """
    Run full experiment on REAL dataset.
    """
    if test_k_values is None:
        test_k_values = [0, 1, 2, 4, 8, 16]
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"TINY TRANSFORMER ON REAL DATA: {dataset_name}")
    print(f"{'='*60}")
    print(f"Training k: {k_train}")
    print(f"Test k values: {test_k_values}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Train
    model, backend, dataset = train_on_real_data(
        dataset_name=dataset_name,
        k_train=k_train,
        n_epochs=50,
        device=device
    )
    
    # Evaluate
    print(f"\n{'='*60}")
    print("EVALUATION ON REAL DATA")
    print(f"{'='*60}")
    
    results = []
    for k in test_k_values:
        acc = evaluate_k_shot_real(model, backend, dataset, k_test=k, device=device)
        results.append({
            "k": k,
            "accuracy": acc,
            "dataset": dataset_name,
            "train_k": k_train
        })
        print(f"k={k:2d}: Accuracy={acc:.4f} ({acc*100:.2f}%)")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / f"{dataset_name}_k{k_train}_results.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Save model
    torch.save(model.state_dict(), output_path / f"{dataset_name}_model.pt")
    
    print(f"\nResults saved to {results_file}")
    
    return results


if __name__ == "__main__":
    # Run on real ARC dataset
    run_experiment(
        dataset_name="arc_challenge",
        k_train=4,
        test_k_values=[0, 1, 2, 4, 8, 16, 24],
        output_dir="results/tiny_transformer_real"
    )
