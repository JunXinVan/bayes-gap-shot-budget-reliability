"""
Tiny Transformer + Associative Recall for controlled causal validation.
This recreates the paper's Figure 2 experiment (Section 6.1, Appendix D.1).
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import json

class TinyTransformer(nn.Module):
    """2-4 layer transformer for associative recall."""
    
    def __init__(self, vocab_size=100, d_model=64, n_layers=2, n_heads=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 128, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        # x: (batch, seq_len)
        x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output(x)


def generate_associative_recall_data(n_samples=1000, seq_len=20, vocab_size=100, k_train=8):
    """
    Generate associative recall data.
    Format: [key_1, val_1, key_2, val_2, ..., key_n, val_n, query_key, ?]
    Model needs to predict the corresponding value.
    """
    data = []
    for _ in range(n_samples):
        # Random key-value pairs
        keys = torch.randint(0, vocab_size // 2, (k_train,))
        vals = torch.randint(vocab_size // 2, vocab_size, (k_train,))
        
        # Query is one of the keys
        query_idx = torch.randint(0, k_train, (1,)).item()
        query_key = keys[query_idx]
        target_val = vals[query_idx]
        
        # Sequence: key_1, val_1, key_2, val_2, ..., query_key
        seq = torch.zeros(seq_len, dtype=torch.long)
        for i in range(min(k_train, seq_len // 2 - 1)):
            seq[2*i] = keys[i]
            seq[2*i + 1] = vals[i]
        seq[-2] = query_key
        seq[-1] = 0  # placeholder
        
        data.append((seq, target_val))
    
    return data


def train_tiny_transformer(k_train=8, epochs=100):
    """Train on k <= k_train examples."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TinyTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    # Training data
    train_data = generate_associative_recall_data(n_samples=5000, k_train=k_train)
    
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        
        for seq, target in train_data:
            seq = seq.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            
            optimizer.zero_grad()
            output = model(seq)
            # Predict last position
            loss = criterion(output[:, -1, :], target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pred = output[:, -1, :].argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += 1
        
        if epoch % 20 == 0:
            acc = correct / total
            print(f"Epoch {epoch}: Loss={total_loss/len(train_data):.4f}, Acc={acc:.4f}")
    
    return model


def evaluate_k_shot(model, k_test, n_samples=200):
    """Evaluate model on k-shot examples."""
    device = next(model.parameters()).device
    model.eval()
    
    test_data = generate_associative_recall_data(n_samples=n_samples, k_train=k_test)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for seq, target in test_data:
            seq = seq.unsqueeze(0).to(device)
            target = target.unsqueeze(0).to(device)
            
            output = model(seq)
            pred = output[:, -1, :].argmax(dim=-1)
            correct += (pred == target).sum().item()
            total += 1
    
    return correct / total


def run_toy_experiment(k_train=8, k_max=64, output_dir="results/toy_transformer"):
    """
    Run the paper's toy experiment.
    
    Expected curve (Figure 2):
    - k=0 to k=8: High accuracy (~100%)
    - k=9 to k=16: Sharp drop (overprompting starts)
    - k=17 to k=64: Continued decline (no recovery)
    """
    print(f"Training Tiny Transformer on k<={k_train}...")
    model = train_tiny_transformer(k_train=k_train, epochs=100)
    
    print(f"\nEvaluating on k=0 to k={k_max}...")
    results = []
    
    for k in range(0, k_max + 1, 2):
        acc = evaluate_k_shot(model, k_test=k, n_samples=200)
        results.append({"k": k, "accuracy": acc})
        print(f"k={k:2d}: Acc={acc:.4f}")
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "metrics.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Save model
    torch.save(model.state_dict(), output_path / "model.pt")
    
    print(f"\nResults saved to {output_path}")
    return results


if __name__ == "__main__":
    # This matches the paper's Figure 2
    run_toy_experiment(k_train=8, k_max=64)
