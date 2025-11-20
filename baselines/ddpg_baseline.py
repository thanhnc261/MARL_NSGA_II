"""
DDPG (Deep Deterministic Policy Gradient) Baseline

Implements DDPG for continuous control portfolio optimization.

DDPG is an off-policy actor-critic algorithm designed for continuous
action spaces, combining ideas from DPG and DQN.

Reference:
    Lillicrap, T. P., Hunt, J. J., Pritzel, A., Heess, N., Erez, T.,
    Tassa, Y., ... & Wierstra, D. (2015). Continuous control with deep
    reinforcement learning. arXiv preprint arXiv:1509.02971.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class Actor(nn.Module):
    """
    Actor network for DDPG.

    Maps states to continuous actions (portfolio weights).
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Actor, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()  # Output in [-1, 1]
        )

    def forward(self, state):
        return self.network(state)


class Critic(nn.Module):
    """
    Critic network for DDPG.

    Estimates Q-value Q(s, a) for state-action pairs.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super(Critic, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.network(x)


class ReplayBuffer:
    """Experience replay buffer for DDPG."""

    def __init__(self, max_size: int = 100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)


class DDPGAgent:
    """
    DDPG Agent for portfolio optimization.

    Uses deterministic policy gradient to learn continuous portfolio allocations.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_size: int = 256,
        actor_lr: float = 1e-4,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.005,
        buffer_size: int = 100000
    ):
        """
        Initialize DDPG agent.

        Args:
            state_dim: State dimension
            action_dim: Action dimension
            hidden_size: Hidden layer size
            actor_lr: Actor learning rate
            critic_lr: Critic learning rate
            gamma: Discount factor
            tau: Soft update parameter
            buffer_size: Replay buffer size
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau

        # Create networks
        self.actor = Actor(state_dim, action_dim, hidden_size)
        self.actor_target = Actor(state_dim, action_dim, hidden_size)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(state_dim, action_dim, hidden_size)
        self.critic_target = Critic(state_dim, action_dim, hidden_size)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

        # Replay buffer
        self.buffer = ReplayBuffer(buffer_size)

        # Noise for exploration
        self.noise_std = 0.1

    def get_action(self, state, deterministic=False, add_noise=True):
        """
        Get action from policy.

        Args:
            state: Current state
            deterministic: If True, no noise
            add_noise: Whether to add exploration noise

        Returns:
            Action (portfolio weights)
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state_tensor).squeeze(0).numpy()

        # Add exploration noise
        if add_noise and not deterministic:
            noise = np.random.normal(0, self.noise_std, size=action.shape)
            action = action + noise

        return action

    def train(self, batch_size: int = 64):
        """
        Update actor and critic networks.

        Args:
            batch_size: Batch size for training
        """
        if len(self.buffer) < batch_size:
            return

        # Sample batch
        states, actions, rewards, next_states, dones = self.buffer.sample(batch_size)

        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

    def _soft_update(self, source, target):
        """Soft update target network."""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * source_param.data + (1 - self.tau) * target_param.data)


def run_ddpg_baseline(
    env_train,
    env_val,
    env_test,
    episodes: int = 100,
    max_steps: int = 500,
    batch_size: int = 64
):
    """
    Run DDPG baseline end-to-end.

    Args:
        env_train: Training environment
        env_val: Validation environment
        env_test: Test environment
        episodes: Number of training episodes
        max_steps: Maximum steps per episode
        batch_size: Batch size for training

    Returns:
        Dictionary with test metrics
    """
    print("\n" + "="*80)
    print("Running DDPG Baseline")
    print("="*80)

    state_dim = env_train.observation_space['features'].shape[0] * env_train.observation_space['features'].shape[1]
    action_dim = env_train.n_assets + 1

    # Create DDPG agent
    agent = DDPGAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_size=256,
        actor_lr=1e-4,
        critic_lr=1e-3,
        gamma=0.95,
        tau=0.005
    )

    # Training loop
    print(f"\nTraining for {episodes} episodes...")
    for episode in range(episodes):
        obs, _ = env_train.reset()
        episode_reward = 0

        for step in range(max_steps):
            # Get action
            state = obs['features'].flatten()
            action = agent.get_action(state, deterministic=False, add_noise=True)

            # Step environment
            next_obs, rewards, terminated, truncated, _ = env_train.step(action)

            # Use Sharpe ratio as single reward
            reward = rewards[0]

            # Store in replay buffer
            next_state = next_obs['features'].flatten()
            done = float(terminated or truncated)
            agent.buffer.add(state, action, reward, next_state, done)

            # Train
            if len(agent.buffer) >= batch_size:
                agent.train(batch_size)

            episode_reward += reward
            obs = next_obs

            if terminated or truncated:
                break

        if (episode + 1) % 10 == 0:
            print(f"  Episode {episode+1}/{episodes}, Reward: {episode_reward:.4f}, Buffer: {len(agent.buffer)}")

    print("Training complete.")

    # Evaluate on validation
    print("\nValidating...")
    obs, _ = env_val.reset()
    terminated = False
    val_sharpe_sum = 0
    steps = 0

    while not terminated:
        state = obs['features'].flatten()
        action = agent.get_action(state, deterministic=True, add_noise=False)
        obs, rewards, terminated, truncated, _ = env_val.step(action)
        val_sharpe_sum += rewards[0]
        steps += 1

    val_sharpe = val_sharpe_sum / max(steps, 1)
    print(f"Validation Sharpe: {val_sharpe:.4f}")

    # Test on test set
    print("\nTesting...")
    obs, _ = env_test.reset()
    terminated = False

    while not terminated:
        state = obs['features'].flatten()
        action = agent.get_action(state, deterministic=True, add_noise=False)
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
    print("DDPG Baseline Module")
    print("Deep Deterministic Policy Gradient for continuous portfolio control.")
