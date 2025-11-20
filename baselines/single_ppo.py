"""
Single-Agent PPO Baseline (Scalarized Multi-Objective)

Implements PPO with scalarized reward combining return, risk, and explainability.

This serves as a baseline for comparing multi-objective optimization approaches.
Uses a weighted linear combination of objectives:

Reward: r = w1*sharpe + w2*risk_reward + w3*explain_reward

where weights w1, w2, w3 are hyperparameters that determine the trade-off
between objectives.

Reference:
    Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class PPOActorCritic(nn.Module):
    """
    Actor-Critic network for PPO.

    For continuous actions, outputs mean and log_std for Gaussian policy.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(PPOActorCritic, self).__init__()

        self.action_dim = action_dim

        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

        # Actor head (policy)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Parameter(torch.zeros(action_dim))

        # Critic head (value function)
        self.critic = nn.Linear(hidden_size, 1)

    def forward(self, state):
        features = self.shared(state)
        action_mean = self.actor_mean(features)
        value = self.critic(features)
        return action_mean, value

    def get_action_and_value(self, state, action=None):
        """
        Get action, log probability, and value.

        Args:
            state: Current state
            action: If provided, calculate log prob for this action

        Returns:
            (action, log_prob, entropy, value)
        """
        action_mean, value = self.forward(state)
        action_std = torch.exp(self.actor_log_std)

        # Create Gaussian distribution
        dist = Normal(action_mean, action_std)

        if action is None:
            action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)

        return action, log_prob, entropy, value


class PPOAgent:
    """
    Single-agent PPO with scalarized multi-objective reward.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        learning_rate: float = 3e-4,
        gamma: float = 0.95,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        reward_weights: tuple = (0.5, 0.3, 0.2)
    ):
        """
        Initialize PPO agent.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_size: Hidden layer size
            learning_rate: Learning rate
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_epsilon: PPO clipping parameter
            value_coef: Value loss coefficient
            entropy_coef: Entropy bonus coefficient
            reward_weights: (w_sharpe, w_risk, w_explain) for scalarization
        """
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.reward_weights = reward_weights

        # Create network
        self.network = PPOActorCritic(state_dim, action_dim, hidden_size)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)

        # Storage for trajectory
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []

    def get_action(self, state, deterministic=False):
        """
        Get action from policy.

        Args:
            state: Current state
            deterministic: If True, return mean action

        Returns:
            Action
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        with torch.no_grad():
            if deterministic:
                action_mean, _ = self.network(state_tensor)
                action = action_mean.squeeze(0).numpy()
            else:
                action, log_prob, _, value = self.network.get_action_and_value(state_tensor)
                action = action.squeeze(0).numpy()

                # Store for training
                self.states.append(state)
                self.actions.append(action)
                self.log_probs.append(log_prob.item())
                self.values.append(value.item())

        return action

    def store_reward_and_done(self, multi_reward, done):
        """
        Store reward and done flag.

        Args:
            multi_reward: Multi-objective reward [sharpe, risk, explain]
            done: Whether episode is done
        """
        # Scalarize multi-objective reward
        w1, w2, w3 = self.reward_weights
        scalar_reward = w1 * multi_reward[0] + w2 * multi_reward[1] + w3 * multi_reward[2]

        self.rewards.append(scalar_reward)
        self.dones.append(done)

    def train(self, num_epochs: int = 4, batch_size: int = 64):
        """
        Update policy using collected trajectory.

        Args:
            num_epochs: Number of optimization epochs
            batch_size: Mini-batch size
        """
        if len(self.states) == 0:
            return

        # Convert to tensors
        states = torch.FloatTensor(np.array(self.states))
        actions = torch.FloatTensor(np.array(self.actions))
        old_log_probs = torch.FloatTensor(np.array(self.log_probs))
        rewards = np.array(self.rewards)
        values = np.array(self.values)
        dones = np.array(self.dones)

        # Calculate advantages using GAE
        advantages = self._calculate_gae(rewards, values, dones)
        returns = advantages + values

        advantages_tensor = torch.FloatTensor(advantages)
        returns_tensor = torch.FloatTensor(returns)

        # Normalize advantages
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # PPO update
        for _ in range(num_epochs):
            # Get current policy outputs
            _, new_log_probs, entropy, new_values = self.network.get_action_and_value(states, actions)

            # Ratio for PPO
            ratio = torch.exp(new_log_probs - old_log_probs)

            # Clipped surrogate objective
            surr1 = ratio * advantages_tensor
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages_tensor
            policy_loss = -torch.min(surr1, surr2).mean()

            # Value loss
            value_loss = F.mse_loss(new_values.squeeze(), returns_tensor)

            # Entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss

            # Optimization step
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), 0.5)
            self.optimizer.step()

        # Clear trajectory
        self.clear_trajectory()

    def _calculate_gae(self, rewards, values, dones):
        """Calculate Generalized Advantage Estimation."""
        advantages = np.zeros_like(rewards)
        last_advantage = 0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]

            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * last_advantage

        return advantages

    def clear_trajectory(self):
        """Clear stored trajectory."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        self.dones = []


def run_single_ppo_baseline(
    env_train,
    env_val,
    env_test,
    episodes: int = 100,
    max_steps: int = 500,
    reward_weights: tuple = (0.5, 0.3, 0.2)
):
    """
    Run single-agent PPO baseline.

    Args:
        env_train: Training environment
        env_val: Validation environment
        env_test: Test environment
        episodes: Number of training episodes
        max_steps: Max steps per episode
        reward_weights: Weights for (sharpe, risk, explain)

    Returns:
        Dictionary with test metrics
    """
    print("\n" + "="*80)
    print("Running Single-Agent PPO Baseline (Scalarized)")
    print(f"Reward weights: Sharpe={reward_weights[0]}, Risk={reward_weights[1]}, Explain={reward_weights[2]}")
    print("="*80)

    state_dim = env_train.observation_space['features'].shape[0] * env_train.observation_space['features'].shape[1]
    action_dim = env_train.n_assets + 1

    # Create PPO agent
    agent = PPOAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=256,
        learning_rate=3e-4,
        gamma=0.95,
        reward_weights=reward_weights
    )

    # Training loop
    print(f"\nTraining for {episodes} episodes...")
    for episode in range(episodes):
        obs, _ = env_train.reset()
        agent.clear_trajectory()
        episode_reward = 0

        for step in range(max_steps):
            state = obs['features'].flatten()
            action = agent.get_action(state, deterministic=False)

            obs, multi_rewards, terminated, truncated, _ = env_train.step(action)
            done = terminated or truncated

            # Store reward
            agent.store_reward_and_done(multi_rewards, done)
            episode_reward += multi_rewards[0]  # Track Sharpe

            if done:
                break

        # Train after each episode
        agent.train(num_epochs=4)

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{episodes}, Sharpe: {episode_reward:.4f}")

    print("Training complete.")

    # Validate
    print("\nValidating...")
    obs, _ = env_val.reset()
    terminated = False
    val_sharpe = 0
    steps = 0

    while not terminated:
        state = obs['features'].flatten()
        action = agent.get_action(state, deterministic=True)
        obs, rewards, terminated, truncated, _ = env_val.step(action)
        val_sharpe += rewards[0]
        steps += 1

    val_sharpe /= max(steps, 1)
    print(f"Validation Sharpe: {val_sharpe:.4f}")

    # Test
    print("\nTesting...")
    obs, _ = env_test.reset()
    terminated = False

    while not terminated:
        state = obs['features'].flatten()
        action = agent.get_action(state, deterministic=True)
        obs, _, terminated, _, _ = env_test.step(action)

    from utils.metrics import calculate_metrics
    test_metrics = calculate_metrics(env_test.portfolio_history)

    print("\nTest Results:")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.6f}")

    return {
        'model': agent,
        'test_metrics': test_metrics
    }


if __name__ == "__main__":
    print("Single-Agent PPO Baseline Module")
    print("PPO with scalarized multi-objective reward.")
