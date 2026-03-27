"""
Transformer model training for protein sequence fitness prediction.
Proof of concept - works but needs refactoring for production use.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os

class ProteinTransformer(nn.Module):
    def __init__(self):
        super().__init__()

        # Architecture hyperparameters
        vocab_size = 21  # 20 amino acids + padding
        d_model = 128
        nhead = 4
        num_layers = 2
        dim_feedforward = 256
        max_len = 100

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = self._get_positional_encoding(max_len, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Prediction head
        self.fc = nn.Linear(d_model, 1)

    def _get_positional_encoding(self, max_len, d_model):
        pos_enc = torch.zeros(max_len, d_model)
        for pos in range(max_len):
            for i in range(0, d_model, 2):
                pos_enc[pos, i] = np.sin(pos / (10000 ** (i / d_model)))
                if i + 1 < d_model:
                    pos_enc[pos, i + 1] = np.cos(pos / (10000 ** (i / d_model)))
        return pos_enc

    def forward(self, x, mask=None):
        # x: (batch, seq_len)
        embedded = self.embedding(x)

        # Add positional encoding
        seq_len = x.size(1)
        pos_enc = self.pos_encoding[:seq_len, :].unsqueeze(0).to(x.device)
        embedded = embedded + pos_enc

        # Transform
        transformed = self.transformer(embedded, src_key_padding_mask=mask)

        # Pool and predict
        pooled = transformed.mean(dim=1)
        output = self.fc(pooled)

        return output.squeeze(-1)


class ProteinDataset(Dataset):
    def __init__(self, sequences, fitness_scores):
        self.sequences = sequences
        self.fitness_scores = fitness_scores

        # AA to index mapping
        self.aa_to_idx = {
            'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8,
            'K': 9, 'L': 10, 'M': 11, 'N': 12, 'P': 13, 'Q': 14, 'R': 15,
            'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, '-': 0
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        fitness = self.fitness_scores[idx]

        # Tokenize sequence
        tokens = []
        for aa in seq:
            tokens.append(self.aa_to_idx.get(aa, 0))

        # Pad to length 100
        while len(tokens) < 100:
            tokens.append(0)
        tokens = tokens[:100]

        return torch.tensor(tokens), torch.tensor(fitness, dtype=torch.float32)


class Trainer:
    def __init__(self):
        self.model = ProteinTransformer()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        self.train_losses = []
        self.val_losses = []

    def load_data(self, filepath='data/protein_fitness.csv'):
        """Load and prepare data"""
        df = pd.read_csv(filepath)

        # Split data - just use first 80% for train, rest for test
        n = len(df)
        train_size = int(0.8 * n)

        train_df = df[:train_size]
        test_df = df[train_size:]

        # Normalize fitness scores using training mean/std
        train_mean = train_df['fitness'].mean()
        train_std = train_df['fitness'].std()

        train_df['fitness'] = (train_df['fitness'] - train_mean) / train_std
        test_df['fitness'] = (test_df['fitness'] - train_mean) / train_std

        # Create datasets
        self.train_dataset = ProteinDataset(
            train_df['sequence'].tolist(),
            train_df['fitness'].tolist()
        )
        self.test_dataset = ProteinDataset(
            test_df['sequence'].tolist(),
            test_df['fitness'].tolist()
        )

        print(f"Loaded {len(self.train_dataset)} training examples")
        print(f"Loaded {len(self.test_dataset)} test examples")

    def train(self, epochs=50, lr=0.001, batch_size=32):
        """Train the model"""

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()

        best_loss = float('inf')

        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            for sequences, targets in train_loader:
                sequences = sequences.to(self.device)
                targets = targets.to(self.device)

                optimizer.zero_grad()
                predictions = self.model(sequences)
                loss = criterion(predictions, targets)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            self.train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for sequences, targets in test_loader:
                    sequences = sequences.to(self.device)
                    targets = targets.to(self.device)

                    predictions = self.model(sequences)
                    loss = criterion(predictions, targets)
                    val_loss += loss.item()

            val_loss /= len(test_loader)
            self.val_losses.append(val_loss)

            print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

            # Save best model
            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pt')
                print("  -> Saved best model")

        # Plot training curves
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train')
        plt.plot(self.val_losses, label='Validation')
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.legend()
        plt.savefig('training_curves.png')
        plt.close()
        print("Saved training curves to training_curves.png")

    def evaluate(self):
        """Evaluate model on test set"""
        self.model.load_state_dict(torch.load('best_model.pt'))
        self.model.eval()

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=32,
            shuffle=False
        )

        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for sequences, targets in test_loader:
                sequences = sequences.to(self.device)
                predictions = self.model(sequences)

                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(targets.numpy())

        # Calculate metrics
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)

        mse = np.mean((predictions - targets) ** 2)
        mae = np.mean(np.abs(predictions - targets))

        # R^2
        ss_res = np.sum((targets - predictions) ** 2)
        ss_tot = np.sum((targets - np.mean(targets)) ** 2)
        r2 = 1 - (ss_res / ss_tot)

        print(f"\nTest Set Metrics:")
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")

        # Save results
        results = {
            'mse': float(mse),
            'mae': float(mae),
            'r2': float(r2)
        }

        with open('results.json', 'w') as f:
            json.dump(results, f, indent=2)

        # Plot predictions vs actual
        plt.figure(figsize=(8, 8))
        plt.scatter(targets, predictions, alpha=0.5)
        plt.plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--')
        plt.xlabel('Actual Fitness')
        plt.ylabel('Predicted Fitness')
        plt.title('Predictions vs Actual')
        plt.savefig('predictions.png')
        plt.close()
        print("Saved predictions plot to predictions.png")

        return results


if __name__ == '__main__':
    trainer = Trainer()
    trainer.load_data()
    trainer.train(epochs=50)
    trainer.evaluate()
