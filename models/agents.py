"""
Multi-Agent Reinforcement Learning Agents for Portfolio Optimization

This module implements specialized agents for the E-NSGA-II + X-MARL framework:
  - ReturnAgent: Maximizes portfolio returns
  - RiskAgent: Minimizes tail risk (CVaR)
  - ExplainAgent: Maximizes decision explainability

Architecture:
  - Deeper networks (3 layers: 256 -> 128 -> 64)
  - LayerNorm for training stability
  - Specialized reward heads for each objective
  - PPO-compatible actor-critic structure

Author: E-NSGA-II + X-MARL Research Team
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """
    Deep Actor-Critic network for continuous action spaces.

    Architecture:
      Input -> [256] -> LayerNorm -> ReLU -> Dropout
            -> [128] -> LayerNorm -> ReLU
            -> [64]  -> ReLU
            -> Actor Head (action logits)
            -> Critic Head (value estimate)

    This deeper architecture handles high-dimensional inputs
    (30 stocks × 22 features = 660 dimensions) more effectively
    than shallow networks.
    """

    def __init__(
        self,
        num_inputs: int,
        num_actions: int,
        hidden_sizes: List[int] = [256, 128, 64],
        dropout: float = 0.1,
        init_std: float = 0.5
    ):
        """
        Initialize the Actor-Critic network.

        Args:
            num_inputs: Dimension of input features (flattened)
            num_actions: Dimension of action space (n_assets + 1 for cash)
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate for regularization
            init_std: Initial standard deviation for action distribution
        """
        super(ActorCritic, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions

        # ===== Shared Feature Extractor =====
        layers = []
        prev_size = num_inputs

        for i, hidden_size in enumerate(hidden_sizes):
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.ReLU())

            # Add dropout only to first layer
            if i == 0 and dropout > 0:
                layers.append(nn.Dropout(dropout))

            prev_size = hidden_size

        self.shared = nn.Sequential(*layers)

        # ===== Actor Head (Policy) =====
        # Outputs mean of Gaussian distribution for each action
        self.actor_mean = nn.Linear(hidden_sizes[-1], num_actions)

        # Learnable log std (shared across actions for stability)
        self.actor_log_std = nn.Parameter(torch.ones(num_actions) * np.log(init_std))

        # ===== Critic Head (Value Function) =====
        self.critic = nn.Sequential(
            nn.Linear(hidden_sizes[-1], 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights using orthogonal initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)

        # Smaller initialization for actor output (more conservative initial policy)
        nn.init.orthogonal_(self.actor_mean.weight, gain=0.01)
        nn.init.constant_(self.actor_mean.bias, 0.0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the network.

        Args:
            x: Input tensor of shape (batch_size, num_inputs)

        Returns:
            Tuple of (action_mean, value_estimate)
        """
        features = self.shared(x)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action_distribution(self, x: torch.Tensor) -> Normal:
        """
        Get the action distribution for given state.

        Args:
            x: Input tensor

        Returns:
            Normal distribution over actions
        """
        features = self.shared(x)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)
        return Normal(action_mean, action_std)

    def evaluate_actions(
        self,
        x: torch.Tensor,
        actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Evaluate log probabilities and entropy for given actions.

        Used during PPO training to compute policy gradient.

        Args:
            x: Input states
            actions: Actions taken

        Returns:
            Tuple of (log_probs, values, entropy)
        """
        features = self.shared(x)
        action_mean = self.actor_mean(features)
        action_std = torch.exp(self.actor_log_std).expand_as(action_mean)

        dist = Normal(action_mean, action_std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        value = self.critic(features)

        return log_probs, value.squeeze(-1), entropy


class BaseAgent:
    """
    Base class for all specialized agents.

    Provides common functionality:
      - Action selection (deterministic/stochastic)
      - Network parameter access
      - Device management
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 128, 64],
        lr: float = 3e-4,
        device: str = "cpu"
    ):
        """
        Initialize base agent.

        Args:
            obs_dim: Observation dimension (flattened features)
            action_dim: Action dimension (n_assets + 1)
            hidden_sizes: Hidden layer sizes
            lr: Learning rate for optimizer
            device: Device to run on ('cpu' or 'cuda')
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        self.model = ActorCritic(
            obs_dim,
            action_dim,
            hidden_sizes=hidden_sizes
        ).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # For tracking training progress
        self.training_step = 0

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action given state.

        Args:
            state: Observation array (flattened features)
            deterministic: If True, use mean action; else sample

        Returns:
            Action array (logits for softmax in environment)
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

            if deterministic:
                action_mean, _ = self.model(state_tensor)
                return action_mean.cpu().numpy()[0]
            else:
                dist = self.model.get_action_distribution(state_tensor)
                action = dist.sample()
                return action.cpu().numpy()[0]

    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.model(state_tensor)
            return value.cpu().numpy()[0, 0]

    def parameters(self):
        """Return model parameters."""
        return self.model.parameters()

    def state_dict(self):
        """Return model state dict."""
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        """Load model state dict."""
        self.model.load_state_dict(state_dict)


class ReturnAgent(BaseAgent):
    """
    Agent specialized for return maximization.

    Objective: Maximize portfolio returns (Sharpe ratio)
    Reward: Daily portfolio return or rolling Sharpe

    This agent focuses on:
      - Identifying high-momentum stocks
      - Timing market entry/exit
      - Capturing positive returns
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 128, 64],
        lr: float = 3e-4,
        device: str = "cpu"
    ):
        super().__init__(obs_dim, action_dim, hidden_sizes, lr, device)
        self.agent_type = "return"

    def compute_reward(
        self,
        portfolio_return: float,
        returns_history: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute return-focused reward.

        Args:
            portfolio_return: Current period return
            returns_history: Historical returns for Sharpe calculation

        Returns:
            Reward value (higher = better returns)
        """
        if returns_history is not None and len(returns_history) >= 20:
            # Rolling Sharpe reward
            mean_ret = np.mean(returns_history[-20:])
            std_ret = np.std(returns_history[-20:]) + 1e-8
            sharpe = (mean_ret / std_ret) * np.sqrt(252)
            return sharpe
        else:
            # Simple return reward (scaled for stability)
            return portfolio_return * 100  # Scale daily returns


class RiskAgent(BaseAgent):
    """
    Agent specialized for risk minimization.

    Objective: Minimize tail risk (CVaR-95%)
    Reward: Negative CVaR or downside deviation

    This agent focuses on:
      - Avoiding extreme losses
      - Diversification across uncorrelated assets
      - Defensive positioning during volatility
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 128, 64],
        lr: float = 3e-4,
        device: str = "cpu"
    ):
        super().__init__(obs_dim, action_dim, hidden_sizes, lr, device)
        self.agent_type = "risk"

    def compute_reward(
        self,
        portfolio_return: float,
        returns_history: Optional[np.ndarray] = None,
        cvar_95: Optional[float] = None
    ) -> float:
        """
        Compute risk-focused reward.

        Args:
            portfolio_return: Current period return
            returns_history: Historical returns for CVaR calculation
            cvar_95: Pre-computed CVaR-95% (if available)

        Returns:
            Reward value (higher = lower risk)
        """
        if cvar_95 is not None:
            # Direct CVaR reward (negated since lower CVaR is better)
            return -cvar_95

        elif returns_history is not None and len(returns_history) >= 20:
            # Compute CVaR from history
            sorted_returns = np.sort(returns_history[-20:])
            var_idx = int(0.05 * len(sorted_returns))
            cvar = np.mean(sorted_returns[:max(1, var_idx)])
            return -cvar  # Negative because we want to minimize losses

        else:
            # Penalize large negative returns
            if portfolio_return < 0:
                return portfolio_return * 2  # Double penalty for losses
            return 0.0


class ExplainAgent(BaseAgent):
    """
    Agent specialized for explainability.

    Objective: Maximize decision interpretability (SHAP stability)
    Reward: Consistency of portfolio decisions over time

    This agent focuses on:
      - Stable portfolio allocations
      - Consistent decision patterns
      - Interpretable weight distributions
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 128, 64],
        lr: float = 3e-4,
        device: str = "cpu"
    ):
        super().__init__(obs_dim, action_dim, hidden_sizes, lr, device)
        self.agent_type = "explain"
        self.previous_action = None

    def compute_reward(
        self,
        current_action: np.ndarray,
        shap_score: Optional[float] = None,
        weight_history: Optional[List[np.ndarray]] = None
    ) -> float:
        """
        Compute explainability-focused reward.

        Args:
            current_action: Current portfolio weights
            shap_score: Pre-computed SHAP-based score (if available)
            weight_history: Historical weight decisions

        Returns:
            Reward value (higher = more explainable)
        """
        if shap_score is not None:
            return shap_score

        # Proxy: Weight consistency (less turnover = more explainable)
        if self.previous_action is not None:
            # Penalize large weight changes
            turnover = np.sum(np.abs(current_action - self.previous_action))
            consistency = 1.0 - min(turnover / 2.0, 1.0)  # Normalize to [0, 1]
            self.previous_action = current_action.copy()
            return consistency

        self.previous_action = current_action.copy()
        return 0.5  # Neutral initial reward


class MultiAgentEnsemble:
    """
    Ensemble of specialized agents that collaborate on portfolio decisions.

    Combines ReturnAgent, RiskAgent, and ExplainAgent into a unified
    decision-making system. Can use various aggregation strategies:
      - Weighted average of agent actions
      - Voting (rank-based selection)
      - Learned combination network
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: List[int] = [256, 128, 64],
        lr: float = 3e-4,
        device: str = "cpu",
        aggregation: str = "weighted_average"
    ):
        """
        Initialize multi-agent ensemble.

        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            hidden_sizes: Hidden layer sizes for each agent
            lr: Learning rate
            device: Computation device
            aggregation: How to combine agent actions
                - "weighted_average": Weight by agent performance
                - "equal": Simple average
                - "voting": Rank-based voting
        """
        self.return_agent = ReturnAgent(obs_dim, action_dim, hidden_sizes, lr, device)
        self.risk_agent = RiskAgent(obs_dim, action_dim, hidden_sizes, lr, device)
        self.explain_agent = ExplainAgent(obs_dim, action_dim, hidden_sizes, lr, device)

        self.agents = [self.return_agent, self.risk_agent, self.explain_agent]
        self.agent_weights = np.array([0.4, 0.4, 0.2])  # Return, Risk, Explain
        self.aggregation = aggregation

    def get_action(
        self,
        state: np.ndarray,
        deterministic: bool = False
    ) -> np.ndarray:
        """
        Get ensemble action by aggregating individual agent actions.

        Args:
            state: Observation array
            deterministic: Whether to use deterministic actions

        Returns:
            Aggregated action array
        """
        actions = []
        for agent in self.agents:
            action = agent.get_action(state, deterministic)
            actions.append(action)

        actions = np.array(actions)

        if self.aggregation == "weighted_average":
            # Weighted combination of actions
            return np.average(actions, axis=0, weights=self.agent_weights)

        elif self.aggregation == "equal":
            # Simple average
            return np.mean(actions, axis=0)

        elif self.aggregation == "voting":
            # Rank-based: for each asset, use median action
            return np.median(actions, axis=0)

        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")

    def update_weights(self, performance: np.ndarray):
        """
        Update agent weights based on recent performance.

        Args:
            performance: Array of [return_perf, risk_perf, explain_perf]
        """
        # Softmax weighting based on performance
        exp_perf = np.exp(performance - np.max(performance))
        self.agent_weights = exp_perf / np.sum(exp_perf)


# Backward compatibility aliases
Agent = BaseAgent


if __name__ == "__main__":
    # Test agent creation
    print("Testing agent architectures...")

    obs_dim = 660  # 30 stocks × 22 features
    action_dim = 31  # 30 stocks + cash

    # Test individual agents
    return_agent = ReturnAgent(obs_dim, action_dim)
    risk_agent = RiskAgent(obs_dim, action_dim)
    explain_agent = ExplainAgent(obs_dim, action_dim)

    # Test forward pass
    test_state = np.random.randn(obs_dim).astype(np.float32)

    print(f"\nReturnAgent action shape: {return_agent.get_action(test_state).shape}")
    print(f"RiskAgent action shape: {risk_agent.get_action(test_state).shape}")
    print(f"ExplainAgent action shape: {explain_agent.get_action(test_state).shape}")

    # Test ensemble
    ensemble = MultiAgentEnsemble(obs_dim, action_dim)
    ensemble_action = ensemble.get_action(test_state)
    print(f"Ensemble action shape: {ensemble_action.shape}")

    # Count parameters
    total_params = sum(p.numel() for p in return_agent.parameters())
    print(f"\nParameters per agent: {total_params:,}")
    print(f"Total ensemble parameters: {total_params * 3:,}")

    print("\nAll tests passed!")
