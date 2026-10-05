"""
Formal Transformer Experiment Framework for Paper
Systematic hyperparameter search for optimal OverPrompting demonstration
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple, Optional
import torch.utils.data as data
from dataclasses import dataclass
import itertools


@dataclass
class ModelConfig:
    """Transformer model configuration."""
    d_model: int
    n_layers: int
    n_heads: int
    vocab_size: int = 100
    max_len: int = 200
    
    @property
    def name(self) -> str:
        return f"d{self.d_model}_L{self.n_layers}_h{self.n_heads}"
    
    @property
    def param_count(self) -> int:
        """Estimate parameter count."""
        d = self.d_model
        h = self.n_heads
        l = self.n_layers
        v = self.vocab_size
        # Embedding + Positional + Transformer layers + Output
        embedding = v * d + self.max_len * d
        per_layer = 4 * d * d + 2 * d * (4 * d)  # Self-attn + FFN
        output = d * v
        return embedding + l * per_layer + output


class FormalTransformer(nn.Module):
    """Production-quality Transformer for paper experiments."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.max_len, config.d_model) * 0.02)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layers)
        self.output = nn.Linear(config.d_model, config.vocab_size)
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output(x)


class AssociativeRecallTask:
    """Associative Recall task for In-Context Learning."""
    
    def __init__(self, vocab_size=100, key_range=(0, 50), value_range=(50, 100), seed=42):
        self.vocab_size = vocab_size
        self.key_range = key_range
        self.value_range = value_range
        self.rng = np.random.RandomState(seed)
        
    def generate_batch(self, k_pairs: int, batch_size: int, max_len: int = 200) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate a batch of sequences."""
        X, Y = [], []
        
        for _ in range(batch_size):
            if k_pairs <= 0:
                # k=0: No examples, random query
                seq = torch.zeros(max_len, dtype=torch.long)
                seq[0] = self.rng.randint(self.key_range[0], self.key_range[1])
                target = self.rng.randint(self.value_range[0], self.value_range[1])
            else:
                # Generate k key-value pairs
                keys = self.rng.randint(self.key_range[0], self.key_range[1], size=k_pairs)
                vals = self.rng.randint(self.value_range[0], self.value_range[1], size=k_pairs)
                query_idx = self.rng.randint(0, k_pairs)
                
                # Build sequence: k1 v1 k2 v2 ... query
                seq_len = 2 * k_pairs + 1
                seq = torch.zeros(max(seq_len, max_len), dtype=torch.long)
                
                for i in range(k_pairs):
                    seq[2*i] = keys[i]
                    seq[2*i + 1] = vals[i]
                seq[2*k_pairs] = keys[query_idx]
                
                target = vals[query_idx]
                
                # Pad to max_len
                if len(seq) < max_len:
                    padded = torch.zeros(max_len, dtype=torch.long)
                    padded[:len(seq)] = seq
                    seq = padded
                else:
                    seq = seq[:max_len]
            
            X.append(seq)
            Y.append(target)
        
        return torch.stack(X), torch.tensor(Y, dtype=torch.long)


@dataclass
class ExperimentResult:
    """Structured experiment result."""
    config: ModelConfig
    train_k: int
    k_values: List[int]
    accuracies: List[float]
    peak_k: int
    peak_acc: float
    op_severity: float  # OverPrompting severity
    random_baseline: float
    
    def to_dict(self) -> Dict:
        return {
            'model': self.config.name,
            'params': self.config.param_count,
            'd_model': self.config.d_model,
            'n_layers': self.config.n_layers,
            'n_heads': self.config.n_heads,
            'train_k': self.train_k,
            'k_values': self.k_values,
            'accuracies': self.accuracies,
            'peak_k': self.peak_k,
            'peak_acc': self.peak_acc,
            'op_severity': self.op_severity,
            'random_baseline': self.random_baseline,
        }


class FormalExperimentRunner:
    """Production experiment runner."""
    
    def __init__(self, output_dir: str = "results/formal_transformer"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def train_model(self, model: nn.Module, train_k: int, 
                    n_samples: int = 2000, epochs: int = 40, 
                    batch_size: int = 64, lr: float = 1e-3,
                    max_len: int = 200) -> nn.Module:
        """Train model on k-shot task."""
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        criterion = nn.CrossEntropyLoss()
        
        task = AssociativeRecallTask(seed=42)
        
        for epoch in range(epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            # Training batches
            n_batches = n_samples // batch_size
            for _ in range(n_batches):
                X, Y = task.generate_batch(train_k, batch_size, max_len)
                X, Y = X.to(self.device), Y.to(self.device)
                
                optimizer.zero_grad()
                output = model(X)
                logits = output[:, -1, :]  # Last position
                loss = criterion(logits, Y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item() * len(X)
                correct += (logits.argmax(-1) == Y).sum().item()
                total += len(X)
            
            scheduler.step()
            
            if epoch % 10 == 0:
                print(f"    Epoch {epoch:2d}: Loss={total_loss/total:.4f}, Acc={correct/total:.4f}")
        
        return model
    
    def evaluate_model(self, model: nn.Module, k: int, 
                       n_samples: int = 500, batch_size: int = 64,
                       max_len: int = 200) -> float:
        """Evaluate model on k-shot task."""
        model.eval()
        task = AssociativeRecallTask(seed=123)  # Different seed from training
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            n_batches = n_samples // batch_size
            for _ in range(n_batches):
                X, Y = task.generate_batch(k, batch_size, max_len)
                X, Y = X.to(self.device), Y.to(self.device)
                output = model(X)
                logits = output[:, -1, :]
                correct += (logits.argmax(-1) == Y).sum().item()
                total += len(X)
        
        return correct / total
    
    def run_experiment(self, model_config: ModelConfig, train_k: int,
                       test_k_values: List[int],
                       n_train: int = 2000,
                       n_test: int = 500,
                       epochs: int = 40) -> ExperimentResult:
        """Run single experiment."""
        print(f"\n  Model: {model_config.name} ({model_config.param_count:,} params)")
        print(f"  Training on k={train_k}, Testing on k={test_k_values}")
        
        # Create and train model
        model = FormalTransformer(model_config)
        model = self.train_model(model, train_k, n_samples=n_train, epochs=epochs, 
                                  max_len=model_config.max_len)
        
        # Evaluate on different k values
        accuracies = []
        for k in test_k_values:
            acc = self.evaluate_model(model, k, n_samples=n_test, 
                                       max_len=model_config.max_len)
            accuracies.append(acc)
            print(f"    k={k:2d}: Acc={acc:.4f}")
        
        # Compute metrics
        peak_idx = np.argmax(accuracies)
        peak_k = test_k_values[peak_idx]
        peak_acc = accuracies[peak_idx]
        random_baseline = accuracies[0] if test_k_values[0] == 0 else 1.0 / 50
        final_acc = accuracies[-1]
        op_severity = (peak_acc - final_acc) / peak_acc if peak_acc > 0 else 0
        
        result = ExperimentResult(
            config=model_config,
            train_k=train_k,
            k_values=test_k_values,
            accuracies=accuracies,
            peak_k=peak_k,
            peak_acc=peak_acc,
            op_severity=op_severity,
            random_baseline=random_baseline
        )
        
        return result
    
    def save_result(self, result: ExperimentResult, exp_name: str):
        """Save experiment result."""
        exp_dir = self.output_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(exp_dir / 'result.json', 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
        
        # Save metrics in jsonl format
        with open(exp_dir / 'metrics.jsonl', 'w') as f:
            for k, acc in zip(result.k_values, result.accuracies):
                f.write(json.dumps({'k': k, 'accuracy': acc}) + '\n')
        
        print(f"  Saved to {exp_dir}")


def grid_search(output_dir: str = "results/formal_transformer"):
    """
    Grid search for optimal OverPrompting configuration.
    This is the MAIN function for formal experiments.
    """
    runner = FormalExperimentRunner(output_dir)
    
    # Define search space (systematic exploration)
    search_configs = [
        # Small models (underfitting regime)
        ModelConfig(d_model=32, n_layers=1, n_heads=2),
        ModelConfig(d_model=32, n_layers=2, n_heads=2),
        
        # Medium models (balanced)
        ModelConfig(d_model=64, n_layers=2, n_heads=2),
        ModelConfig(d_model=64, n_layers=4, n_heads=4),
        
        # Large models (expressive but may overfit)
        ModelConfig(d_model=128, n_layers=2, n_heads=4),
        ModelConfig(d_model=128, n_layers=4, n_heads=4),
        ModelConfig(d_model=128, n_layers=6, n_heads=4),
        
        # Very large (for comparison)
        ModelConfig(d_model=256, n_layers=4, n_heads=8),
    ]
    
    train_k_values = [4, 8, 12]  # Different training lengths
    test_k_values = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64]
    
    all_results = []
    
    print("=" * 70)
    print("FORMAL TRANSFORMER EXPERIMENTS - GRID SEARCH")
    print("=" * 70)
    print(f"Device: {runner.device}")
    print(f"Testing {len(search_configs)} model configs × {len(train_k_values)} train_k = {len(search_configs) * len(train_k_values)} experiments")
    print()
    
    for i, model_config in enumerate(search_configs, 1):
        for j, train_k in enumerate(train_k_values, 1):
            exp_id = f"exp{i}_{j}"
            exp_name = f"{model_config.name}_train{train_k}"
            
            print(f"\n[{exp_id}] {exp_name}")
            print("-" * 50)
            
            try:
                result = runner.run_experiment(
                    model_config=model_config,
                    train_k=train_k,
                    test_k_values=test_k_values,
                    n_train=2000,
                    n_test=500,
                    epochs=40
                )
                runner.save_result(result, exp_name)
                all_results.append(result)
                
                # Print summary
                print(f"\n  Peak: k={result.peak_k}, Acc={result.peak_acc:.2%}")
                print(f"  OP Severity: {result.op_severity:.1%}")
                
            except Exception as e:
                print(f"  ERROR: {e}")
                import traceback
                traceback.print_exc()
    
    # Save overall summary
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)
    
    # Rank by OverPrompting severity
    print("\nTop 5 by OP Severity:")
    sorted_by_op = sorted(all_results, key=lambda x: x.op_severity, reverse=True)
    for i, r in enumerate(sorted_by_op[:5], 1):
        print(f"  {i}. {r.config.name} (train_k={r.train_k}): "
              f"OP={r.op_severity:.1%}, Peak={r.peak_acc:.1%}@k={r.peak_k}")
    
    # Rank by peak accuracy
    print("\nTop 5 by Peak Accuracy:")
    sorted_by_peak = sorted(all_results, key=lambda x: x.peak_acc, reverse=True)
    for i, r in enumerate(sorted_by_peak[:5], 1):
        print(f"  {i}. {r.config.name} (train_k={r.train_k}): "
              f"Peak={r.peak_acc:.1%}@k={r.peak_k}, OP={r.op_severity:.1%}")
    
    # Save summary
    summary = {
        'total_experiments': len(all_results),
        'by_op_severity': [r.to_dict() for r in sorted_by_op],
        'by_peak_accuracy': [r.to_dict() for r in sorted_by_peak],
    }
    
    with open(Path(output_dir) / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to {output_dir}/summary.json")
    
    return all_results


if __name__ == '__main__':
    grid_search()
