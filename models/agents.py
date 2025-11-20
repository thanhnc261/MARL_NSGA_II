import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=64):
        super(ActorCritic, self).__init__()
        
        self.common = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )
        
        self.actor = nn.Linear(hidden_size, num_actions)
        self.critic = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        x = self.common(x)
        return self.actor(x), self.critic(x)

class Agent:
    def __init__(self, obs_dim, action_dim, lr=3e-4):
        self.model = ActorCritic(obs_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
    def get_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0)
        logits, _ = self.model(state)
        
        if deterministic:
            action = torch.argmax(logits, dim=1) # Wait, continuous action?
            # If continuous, we output mean/std or just mean.
            # Guide says: "continuous vector ... softmax-normalized"
            # So we output logits, and the Env does softmax.
            # So action is just the logits vector.
            return logits.detach().numpy()[0]
        else:
            # Sample from distribution?
            # For continuous actions, usually Gaussian.
            # But here we output logits for Softmax.
            # We can add noise to logits or sample from Categorical?
            # "Action: continuous vector ... softmax-normalized to new weights"
            # This implies we output the "scores" for each asset.
            # We can treat this as a deterministic output + exploration noise.
            return logits.detach().numpy()[0]

class ReturnAgent(Agent):
    pass

class RiskAgent(Agent):
    pass

class ExplainAgent(Agent):
    pass
