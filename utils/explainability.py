"""
SHAP-based Explainability Module for Portfolio Agents

Implements the explainability score for interpretable reinforcement learning:
E = 1 / (1 + σ_SHAP / σ_SHAP_baseline)

Where:
- σ_SHAP = temporal stability measure (std dev of SHAP values across rollouts)
- σ_SHAP_baseline = stability measure for random policy baseline
- E ∈ [0, 1], higher = more stable = more explainable

Reference:
    Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting
    model predictions. NeurIPS.
"""

import numpy as np
import torch
import shap
from typing import Dict, Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class ExplainabilityScorer:
    """
    Calculates SHAP-based explainability scores for RL agents.

    Algorithm:
    - Run N episodes with same policy on validation data (default: 50)
    - Compute SHAP values (DeepSHAP or KernelSHAP) for every feature/timestep
    - Calculate temporal stability: σ_SHAP = mean std_dev across features
    - Compare to baseline random policy for normalization
    """

    def __init__(
        self,
        n_rollouts: int = 50,
        method: str = 'deep',  # 'deep' or 'kernel'
        batch_size: int = 10,
        cache_baseline: bool = True
    ):
        """
        Initialize explainability scorer.

        Args:
            n_rollouts: Number of episodes to run for SHAP stability (default 50)
            method: SHAP method ('deep' for DeepSHAP, 'kernel' for KernelSHAP)
            batch_size: Batch size for SHAP computation
            cache_baseline: Whether to cache baseline calculation
        """
        self.n_rollouts = n_rollouts
        self.method = method
        self.batch_size = batch_size
        self.cache_baseline = cache_baseline
        self._baseline_sigma = None

    def calculate_shap_values(
        self,
        agent,
        env,
        background_samples: Optional[np.ndarray] = None,
        max_steps: int = 50
    ) -> np.ndarray:
        """
        Calculate SHAP values for agent's policy on environment.

        Args:
            agent: Agent with get_action(state) method
            env: Environment to run episodes
            background_samples: Background data for SHAP (if None, collected from env)
            max_steps: Maximum steps per episode

        Returns:
            SHAP values array of shape (n_samples, n_features)
        """
        # Collect states from one episode for background if not provided
        if background_samples is None:
            background_samples = self._collect_background_samples(agent, env, max_steps)

        # Prepare model wrapper for SHAP
        model_wrapper = self._create_model_wrapper(agent)

        if self.method == 'deep':
            # DeepSHAP for neural network policies
            explainer = shap.DeepExplainer(
                model_wrapper,
                torch.FloatTensor(background_samples)
            )

            # Calculate SHAP values
            shap_values = explainer.shap_values(
                torch.FloatTensor(background_samples),
                check_additivity=False
            )

        elif self.method == 'kernel':
            # KernelSHAP (model-agnostic but slower)
            explainer = shap.KernelExplainer(
                lambda x: model_wrapper(torch.FloatTensor(x)).detach().numpy(),
                background_samples[:100]  # Use subset for efficiency
            )

            shap_values = explainer.shap_values(
                background_samples,
                nsamples=100
            )
        else:
            raise ValueError(f"Unknown SHAP method: {self.method}")

        # Convert to numpy if needed
        if isinstance(shap_values, torch.Tensor):
            shap_values = shap_values.detach().numpy()

        return np.array(shap_values)

    def calculate_temporal_stability(
        self,
        agent,
        env,
        max_steps: int = 50
    ) -> float:
        """
        Calculate temporal stability (σ_SHAP) across multiple rollouts.

        Computes the mean standard deviation of SHAP values across rollouts.

        Args:
            agent: Agent to evaluate
            env: Environment
            max_steps: Steps per episode

        Returns:
            σ_SHAP: Temporal stability measure (lower = more stable)
        """
        all_shap_values = []

        # Run n_rollouts episodes
        for rollout_idx in range(self.n_rollouts):
            # Reset environment with same or similar initial state
            obs, _ = env.reset()

            # Collect trajectory
            states = []
            for step in range(max_steps):
                features = obs['features'].flatten()
                states.append(features)

                # Get action and step
                action = agent.get_action(features, deterministic=True)
                obs, _, terminated, truncated, _ = env.step(action)

                if terminated or truncated:
                    break

            # Calculate SHAP values for this rollout
            states_array = np.array(states)
            if len(states_array) > 0:
                shap_vals = self.calculate_shap_values(
                    agent, env,
                    background_samples=states_array,
                    max_steps=max_steps
                )
                all_shap_values.append(shap_vals)

        # Calculate std dev across rollouts for each feature
        # all_shap_values shape: (n_rollouts, n_timesteps, n_features)
        # We want: mean over features of std_dev across rollouts

        if len(all_shap_values) == 0:
            return float('inf')  # No data = unstable

        # Pad to same length
        max_len = max(arr.shape[0] for arr in all_shap_values)
        padded_shap = []
        for arr in all_shap_values:
            if arr.shape[0] < max_len:
                padding = np.zeros((max_len - arr.shape[0], arr.shape[1]))
                arr = np.vstack([arr, padding])
            padded_shap.append(arr)

        shap_array = np.array(padded_shap)  # (n_rollouts, n_timesteps, n_features)

        # Calculate std across rollouts for each (timestep, feature)
        temporal_std = np.std(shap_array, axis=0)  # (n_timesteps, n_features)

        # Mean over features and timesteps
        sigma_shap = np.mean(temporal_std)

        return float(sigma_shap)

    def calculate_baseline_stability(
        self,
        env,
        obs_dim: int,
        action_dim: int,
        max_steps: int = 50
    ) -> float:
        """
        Calculate baseline stability for random policy.

        Args:
            env: Environment
            obs_dim: Observation dimension
            action_dim: Action dimension
            max_steps: Steps per episode

        Returns:
            σ_SHAP_baseline: Baseline stability measure
        """
        if self.cache_baseline and self._baseline_sigma is not None:
            return self._baseline_sigma

        # Create random policy
        class RandomAgent:
            def __init__(self, action_dim):
                self.action_dim = action_dim

            def get_action(self, state, deterministic=False):
                return np.random.randn(self.action_dim)

        random_agent = RandomAgent(action_dim)

        # Calculate stability for random agent
        sigma_baseline = self.calculate_temporal_stability(
            random_agent, env, max_steps
        )

        if self.cache_baseline:
            self._baseline_sigma = sigma_baseline

        return sigma_baseline

    def calculate_explainability_score(
        self,
        agent,
        env,
        obs_dim: int,
        action_dim: int,
        max_steps: int = 50
    ) -> float:
        """
        Calculate final explainability score E.

        Formula:
        E = 1 / (1 + σ_SHAP / σ_SHAP_baseline)

        Returns:
            E ∈ [0, 1], higher = more explainable
        """
        # Calculate temporal stability for agent
        sigma_shap = self.calculate_temporal_stability(agent, env, max_steps)

        # Calculate baseline
        sigma_baseline = self.calculate_baseline_stability(
            env, obs_dim, action_dim, max_steps
        )

        # Avoid division by zero
        if sigma_baseline < 1e-8:
            sigma_baseline = 1e-8

        # Calculate explainability score
        E = 1.0 / (1.0 + sigma_shap / sigma_baseline)

        # Ensure bounded [0, 1]
        E = np.clip(E, 0.0, 1.0)

        return float(E)

    def _collect_background_samples(
        self,
        agent,
        env,
        max_steps: int
    ) -> np.ndarray:
        """Collect background samples for SHAP from one episode."""
        obs, _ = env.reset()
        samples = []

        for _ in range(max_steps):
            features = obs['features'].flatten()
            samples.append(features)

            action = agent.get_action(features, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                break

        return np.array(samples)

    def _create_model_wrapper(self, agent):
        """
        Create a wrapper function for SHAP that outputs action probabilities.

        For continuous actions, we output the raw logits/scores.
        """
        def model_fn(x):
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)

            # Get model output
            with torch.no_grad():
                if hasattr(agent, 'model'):
                    logits, _ = agent.model(x)
                else:
                    # Fallback for simple agents
                    logits = torch.zeros((x.shape[0], agent.action_dim))

            return logits

        return model_fn


def calculate_step_explainability(
    portfolio_history: list,
    weight_history: list,
    window_size: int = 20
) -> float:
    """
    Simplified explainability score for step-wise reward calculation.

    Uses weight consistency as a proxy for explainability:
    - More consistent weights = more explainable
    - Based on weight changes over rolling window

    Args:
        portfolio_history: List of portfolio states
        weight_history: List of weight decisions
        window_size: Window for consistency check

    Returns:
        Explainability proxy score [0, 1]
    """
    if len(weight_history) < window_size:
        return 0.5  # Neutral score if insufficient data

    # Get recent weights
    recent_weights = np.array(weight_history[-window_size:])

    # Calculate consistency: 1 - normalized std of weight changes
    weight_changes = np.diff(recent_weights, axis=0)
    std_changes = np.std(weight_changes, axis=0)
    mean_std = np.mean(std_changes)

    # Normalize to [0, 1]: lower std = higher explainability
    # Assume std > 0.5 is very unstable, std < 0.05 is very stable
    consistency = 1.0 - np.clip(mean_std / 0.5, 0, 1)

    return float(consistency)


if __name__ == "__main__":
    # Example usage
    print("Explainability module loaded successfully.")
    print("To calculate explainability score:")
    print("  scorer = ExplainabilityScorer(n_rollouts=50)")
    print("  E = scorer.calculate_explainability_score(agent, env, obs_dim, action_dim)")
