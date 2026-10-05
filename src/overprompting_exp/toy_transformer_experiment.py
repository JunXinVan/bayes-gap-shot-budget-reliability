"""
Tiny Transformer Experiment Framework
Based on Paper Section 6.1 / Appendix D.1
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json
from typing import List, Dict, Tuple
import torch.utils.data as data


class TinyTransformer(nn.Module):
    """Configurable small Transformer for controlled experiments."""
    
    def __init__(self, vocab_size=100, d_model=64, n_layers=2, n_heads=2, max_len=130):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, max_len, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads, 
            dim_feedforward=d_model*4, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output(x)


class AssociativeRecallTask:
    """Associative Recall task: k1 v1 k2 v2 ... query -> answer"""
    
    def __init__(self, vocab_size=100, key_range=(0, 50), value_range=(50, 100)):
        self.vocab_size = vocab_size
        self.key_range = key_range
        self.value_range = value_range
        
    def generate_sequence(self, k_pairs: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate one sequence with k_pairs key-value pairs."""
        if k_pairs <= 0:
            # k=0: random query, random target
            seq = torch.zeros(20, dtype=torch.long)
            seq[0] = torch.randint(self.key_range[0], self.key_range[1], (1,)).item()
            target = torch.randint(self.value_range[0], self.value_range[1], (1,)).item()
            return seq, torch.tensor(target, dtype=torch.long)
        
        keys = torch.randint(self.key_range[0], self.key_range[1], (k_pairs,))
        vals = torch.randint(self.value_range[0], self.value_range[1], (k_pairs,))
        query_idx = torch.randint(0, k_pairs, (1,)).item()
        
        seq_len = 2 * k_pairs + 1  # k1 v1 k2 v2 ... query
        seq = torch.zeros(max(seq_len, 20), dtype=torch.long)
        for i in range(k_pairs):
            seq[2*i] = keys[i]
            seq[2*i + 1] = vals[i]
        seq[2*k_pairs] = keys[query_idx]  # query at the end
        
        return seq, vals[query_idx]
    
    def generate_dataset(self, n_samples: int, k_pairs: int, max_len: int = 130) -> data.Dataset:
        """Generate dataset for training/evaluation."""
        X, Y = [], []
        for _ in range(n_samples):
            seq, target = self.generate_sequence(k_pairs)
            if len(seq) < max_len:
                padded = torch.zeros(max_len, dtype=torch.long)
                padded[:len(seq)] = seq
                X.append(padded)
            else:
                X.append(seq[:max_len])
            Y.append(target if isinstance(target, torch.Tensor) else torch.tensor(target, dtype=torch.long))
        return data.TensorDataset(torch.stack(X), torch.stack(Y))


class TinyTransformerExperiment:
    """Main experiment runner for Tiny Transformer."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model config
        self.vocab_size = config.get('vocab_size', 100)
        self.d_model = config.get('d_model', 64)
        self.n_layers = config.get('n_layers', 2)
        self.n_heads = config.get('n_heads', 2)
        self.max_len = config.get('max_len', 130)
        
        # Training config
        self.train_k = config.get('train_k', 8)
        self.n_train_samples = config.get('n_train_samples', 1000)
        self.n_epochs = config.get('n_epochs', 30)
        self.batch_size = config.get('batch_size', 50)
        self.lr = config.get('lr', 1e-3)
        
        # Evaluation config
        self.test_k_values = config.get('test_k_values', [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64])
        self.n_test_samples = config.get('n_test_samples', 200)
        
        # Output
        self.output_dir = Path(config.get('output_dir', 'results/toy_transformer'))
        self.name = config.get('name', 'default')
        
    def train(self, model: nn.Module, train_loader: data.DataLoader) -> nn.Module:
        """Train the model."""
        model.to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(self.n_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                output = model(batch_x)
                
                # Get prediction at last position
                logits = output[:, -1, :]
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(batch_x)
                correct += (logits.argmax(-1) == batch_y).sum().item()
                total += len(batch_x)
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Loss={total_loss/total:.4f}, Acc={correct/total:.4f}")
                
        return model
    
    def evaluate(self, model: nn.Module, k: int) -> float:
        """Evaluate model on k-shot task."""
        model.eval()
        task = AssociativeRecallTask(self.vocab_size)
        test_dataset = task.generate_dataset(self.n_test_samples, k, self.max_len)
        test_loader = data.DataLoader(test_dataset, batch_size=self.batch_size)
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_x, batch_y in test_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = model(batch_x)
                logits = output[:, -1, :]
                correct += (logits.argmax(-1) == batch_y).sum().item()
                total += len(batch_x)
        
        return correct / total if total > 0 else 0.0
    
    def run(self) -> Dict:
        """Run full experiment."""
        print(f"\n{'='*70}")
        print(f"Tiny Transformer Experiment: {self.name}")
        print(f"{'='*70}")
        print(f"Model: d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}")
        print(f"Train: k<={self.train_k}, {self.n_train_samples} samples, {self.n_epochs} epochs")
        print(f"Test: k in {self.test_k_values}")
        print(f"Device: {self.device}")
        
        # Create model
        model = TinyTransformer(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            n_layers=self.n_layers,
            n_heads=self.n_heads,
            max_len=self.max_len
        )
        
        # Generate training data (only k <= train_k)
        print(f"\n[1/3] Generating training data (k<={self.train_k})...")
        task = AssociativeRecallTask(self.vocab_size)
        train_dataset = task.generate_dataset(self.n_train_samples, self.train_k, self.max_len)
        train_loader = data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Train
        print(f"\n[2/3] Training...")
        model = self.train(model, train_loader)
        
        # Evaluate on different k values
        print(f"\n[3/3] Evaluating on different k values...")
        results = []
        for k in self.test_k_values:
            acc = self.evaluate(model, k)
            results.append({'k': k, 'accuracy': acc})
            print(f"  k={k:2d}: Acc={acc:.4f}")
        
        # Save results
        exp_dir = self.output_dir / self.name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        with open(exp_dir / 'metrics.jsonl', 'w') as f:
            for r in results:
                f.write(json.dumps(r) + '\n')
        
        # Save config
        with open(exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save model
        torch.save(model.state_dict(), exp_dir / 'model.pt')
        
        print(f"\nResults saved to {exp_dir}")
        
        return {
            'results': results,
            'config': self.config,
            'output_dir': str(exp_dir)
        }


def run_experiment_from_config(config_path: str):
    """Run experiment from YAML/JSON config file."""
    import yaml
    
    config_path = Path(config_path)
    if config_path.suffix == '.yaml' or config_path.suffix == '.yml':
        with open(config_path) as f:
            config = yaml.safe_load(f)
    else:
        with open(config_path) as f:
            config = json.load(f)
    
    exp = TinyTransformerExperiment(config)
    return exp.run()


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        run_experiment_from_config(sys.argv[1])
    else:
        # Default experiment
        config = {
            'name': 'baseline',
            'd_model': 64,
            'n_layers': 2,
            'n_heads': 2,
            'train_k': 8,
            'test_k_values': [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64],
        }
        exp = TinyTransformerExperiment(config)
        exp.run()
