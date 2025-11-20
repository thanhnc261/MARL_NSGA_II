"""
LSTM Baseline for Portfolio Optimization

Implements an LSTM-based portfolio strategy that predicts future returns
and constructs portfolios based on those predictions.

This represents a standard deep learning approach to financial forecasting
and portfolio construction.

Reference:
    Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory.
    Neural Computation, 9(8), 1735-1780.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class LSTMPredictor(nn.Module):
    """
    LSTM network for predicting asset returns.

    Architecture:
    - Input: sequence of features (lookback x n_features)
    - LSTM layers
    - Output: predicted returns for each asset
    """

    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, n_assets: int = 10):
        """
        Initialize LSTM predictor.

        Args:
            input_size: Feature dimension per asset
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            n_assets: Number of assets
        """
        super(LSTMPredictor, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_assets = n_assets

        # LSTM for feature processing
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )

        # Output layer: predict returns for each asset
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size, n_assets)
        )

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch, sequence, features)

        Returns:
            Predicted returns of shape (batch, n_assets)
        """
        # LSTM forward
        lstm_out, _ = self.lstm(x)

        # Take last timestep output
        last_output = lstm_out[:, -1, :]

        # Predict returns
        predictions = self.fc(last_output)

        return predictions


class PortfolioDataset(Dataset):
    """Dataset for LSTM training."""

    def __init__(self, env, lookback: int = 20):
        """
        Create dataset from environment.

        Args:
            env: Portfolio environment
            lookback: Number of historical timesteps to use
        """
        self.env = env
        self.lookback = lookback

        # Collect all data
        self.sequences = []
        self.targets = []

        obs, _ = env.reset()
        for step in range(len(env.dates) - lookback - 1):
            # Collect sequence of features
            sequence = []
            for _ in range(lookback):
                features = obs['features'].flatten()
                sequence.append(features)

                # Step without action (just to get next state)
                dummy_action = np.zeros(env.n_assets + 1)
                obs, _, terminated, _, _ = env.step(dummy_action)

                if terminated:
                    break

            if len(sequence) == lookback and not terminated:
                # Get next returns as target
                next_date = env.dates[env.current_step + 1]
                next_data = env.df.loc[next_date].reindex(env.tickers).fillna(0)
                target_returns = next_data['ret_1'].values

                self.sequences.append(np.array(sequence))
                self.targets.append(target_returns)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.sequences[idx]),
            torch.FloatTensor(self.targets[idx])
        )


class LSTMBaseline:
    """
    LSTM-based portfolio optimizer.

    Strategy:
    1. Train LSTM to predict future returns
    2. At each timestep, predict returns
    3. Allocate weights proportional to predicted returns (higher predicted return = higher weight)
    """

    def __init__(
        self,
        n_assets: int,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        lookback: int = 20,
        learning_rate: float = 1e-3
    ):
        """
        Initialize LSTM baseline.

        Args:
            n_assets: Number of assets
            n_features: Feature dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            lookback: Historical window size
            learning_rate: Learning rate
        """
        self.n_assets = n_assets
        self.n_features = n_features
        self.lookback = lookback

        # Create model
        self.model = LSTMPredictor(
            input_size=n_features * n_assets,
            hidden_size=hidden_size,
            num_layers=num_layers,
            n_assets=n_assets
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

        # Buffer for storing recent observations
        self.obs_buffer = []

    def train(self, env, epochs: int = 10, batch_size: int = 32):
        """
        Train LSTM on environment data.

        Args:
            env: Training environment
            epochs: Number of training epochs
            batch_size: Batch size
        """
        print(f"\nTraining LSTM ({epochs} epochs)...")

        # Create dataset
        dataset = PortfolioDataset(env, self.lookback)

        if len(dataset) == 0:
            print("Warning: No training data available")
            return

        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        # Training loop
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            num_batches = 0

            for sequences, targets in dataloader:
                # Forward pass
                predictions = self.model(sequences)

                # Calculate loss
                loss = self.criterion(predictions, targets)

                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                num_batches += 1

            avg_loss = total_loss / max(num_batches, 1)
            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")

        print("Training complete.")

    def get_action(self, state, deterministic=False):
        """
        Predict returns and convert to portfolio weights.

        Args:
            state: Current state (features)
            deterministic: Ignored

        Returns:
            Action (log-weights for softmax)
        """
        # Add to buffer
        self.obs_buffer.append(state)

        # Keep only last lookback observations
        if len(self.obs_buffer) > self.lookback:
            self.obs_buffer = self.obs_buffer[-self.lookback:]

        # If not enough history, return equal weights
        if len(self.obs_buffer) < self.lookback:
            equal_weights = np.ones(self.n_assets + 1) / (self.n_assets + 1)
            return np.log(equal_weights + 1e-8) * 100

        # Prepare input sequence
        sequence = np.array(self.obs_buffer)  # (lookback, features)
        sequence_tensor = torch.FloatTensor(sequence).unsqueeze(0)  # (1, lookback, features)

        # Predict returns
        self.model.eval()
        with torch.no_grad():
            predicted_returns = self.model(sequence_tensor).squeeze(0)  # (n_assets,)
            predicted_returns = predicted_returns.numpy()

        # Convert predictions to weights
        # Strategy: Allocate more to assets with higher predicted returns
        # Use softmax on predicted returns

        # Add small cash allocation (5%)
        cash_weight = 0.05

        # Softmax on positive predictions only
        exp_returns = np.exp(predicted_returns - np.max(predicted_returns))  # Numerical stability
        asset_weights = exp_returns / np.sum(exp_returns) * (1 - cash_weight)

        # Combine asset weights and cash
        weights = np.concatenate([asset_weights, [cash_weight]])

        # Return log-weights for environment
        return np.log(weights + 1e-8) * 100


def run_lstm_baseline(env_train, env_val, env_test, epochs: int = 10):
    """
    Run LSTM baseline end-to-end.

    Args:
        env_train: Training environment
        env_val: Validation environment
        env_test: Test environment
        epochs: Training epochs

    Returns:
        Dictionary with test metrics
    """
    print("\n" + "="*80)
    print("Running LSTM Baseline")
    print("="*80)

    n_assets = env_train.n_assets
    n_features = env_train.n_features

    # Create and train LSTM
    lstm = LSTMBaseline(
        n_assets=n_assets,
        n_features=n_features,
        hidden_size=64,
        num_layers=2,
        lookback=20,
        learning_rate=1e-3
    )

    lstm.train(env_train, epochs=epochs, batch_size=32)

    # Evaluate on validation
    print("\nValidating...")
    obs, _ = env_val.reset()
    lstm.obs_buffer = []  # Reset buffer
    terminated = False
    val_returns = []

    while not terminated:
        features = obs['features'].flatten()
        action = lstm.get_action(features)
        obs, rewards, terminated, truncated, _ = env_val.step(action)
        val_returns.append(rewards[0])

    val_sharpe = np.mean(val_returns)
    print(f"Validation Sharpe: {val_sharpe:.4f}")

    # Test on test set
    print("\nTesting...")
    obs, _ = env_test.reset()
    lstm.obs_buffer = []  # Reset buffer
    terminated = False

    while not terminated:
        features = obs['features'].flatten()
        action = lstm.get_action(features, deterministic=True)
        obs, _, terminated, _, _ = env_test.step(action)

    from utils.metrics import calculate_metrics
    test_metrics = calculate_metrics(env_test.portfolio_history)

    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")

    return {
        'model': lstm,
        'test_metrics': test_metrics
    }


if __name__ == "__main__":
    print("LSTM Baseline Module")
    print("Predicts returns using LSTM and allocates weights accordingly.")
