import numpy as np
import shap
import torch

class ExplainabilityEvaluator:
    def __init__(self, env, policy, num_rollouts=5):
        self.env = env
        self.policy = policy
        self.num_rollouts = num_rollouts
        self.explainer = None
        self.background_data = None

    def setup_explainer(self, background_size=100):
        """Initializes the SHAP explainer with background data."""
        # Sample background data from the environment
        # We need flattened observations
        states = []
        obs, _ = self.env.reset()
        for _ in range(background_size):
            # Random actions to explore state space
            action = self.env.action_space.sample()
            obs, _, terminated, truncated, _ = self.env.step(action)
            
            # Flatten observation: weights + features
            # obs is dict: {'weights': (N+1,), 'features': (N, F)}
            # We need to flatten it to match the agent's input
            flat_obs = self._flatten_obs(obs)
            states.append(flat_obs)
            
            if terminated or truncated:
                obs, _ = self.env.reset()
                
        self.background_data = torch.tensor(np.array(states), dtype=torch.float32)
        
        # Initialize GradientExplainer
        # We explain the 'common' part of the network or the output?
        # The agent outputs logits. We explain logits.
        self.explainer = shap.GradientExplainer(self.policy, self.background_data)

    def _flatten_obs(self, obs):
        """Flattens the observation dictionary into a single vector."""
        # This must match how the agent processes input
        # In agents.py, the agent takes 'state' which is expected to be flattened?
        # Wait, main.py passes 'flat_features' to agent.get_action.
        # Let's check main.py's evaluate function.
        # It does: flat_features = obs['features'].flatten()
        # It ignores weights?
        # Let's check agents.py.
        # ReturnAgent takes input_dim = n_assets * n_features.
        # So it ONLY looks at features, not current weights.
        # This simplifies things.
        return obs['features'].flatten()

    def compute_explainability_score(self, episode_start_date=None):
        """
        Computes E = 1 - normalized variance of SHAP values across rollouts.
        """
        if self.explainer is None:
            self.setup_explainer()
            
        all_shap_values = [] # List of (T, F) arrays
        
        # Run rollouts
        for _ in range(self.num_rollouts):
            obs, _ = self.env.reset()
            episode_shap = []
            
            # Run for a fixed number of steps or until done
            # To save compute, maybe just 50 steps?
            max_steps = 50 
            for _ in range(max_steps):
                flat_obs = self._flatten_obs(obs)
                input_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0)
                
                # Compute SHAP values for this state
                # shap_values returns a list of arrays (one for each output class)
                # Since we output a vector of weights (continuous), it might return a list of (1, F) arrays?
                # Or a (1, F, OutputDim) array?
                # GradientExplainer for regression/multi-output returns list of tensors.
                
                # We want feature importance for the *chosen* action or *all* actions?
                # Let's aggregate importance across all outputs (sum of absolute SHAP values)
                # to get a single "importance vector" for the input features.
                shap_vals = self.explainer.shap_values(input_tensor)
                
                # shap_vals is a list of length OutputDim (n_assets+1), each element is (1, InputDim)
                # We stack them to get (OutputDim, InputDim)
                # Then sum absolute values across OutputDim -> (InputDim,)
                
                if isinstance(shap_vals, list):
                    # Sum absolute attributions across all output dimensions (portfolio weights)
                    total_importance = np.sum([np.abs(s[0]) for s in shap_vals], axis=0)
                else:
                    # If it returns a single array
                    total_importance = np.abs(shap_vals[0])
                    
                episode_shap.append(total_importance)
                
                # Step environment
                # We need the agent's action
                with torch.no_grad():
                    logits, _ = self.policy(input_tensor)
                    action = logits.numpy().flatten()
                
                obs, _, terminated, truncated, _ = self.env.step(action)
                if terminated or truncated:
                    break
            
            all_shap_values.append(np.array(episode_shap))
            
        # Calculate variance across rollouts
        # Stack: (NumRollouts, T, F)
        # We need to ensure all rollouts have same length T
        min_len = min(len(r) for r in all_shap_values)
        shap_stack = np.stack([r[:min_len] for r in all_shap_values])
        
        # Variance across rollouts (axis 0)
        # shap_stack: (N, T, F)
        variances = np.var(shap_stack, axis=0) # (T, F)
        
        # Mean variance across time and features
        mean_variance = np.mean(variances)
        
        # Normalize?
        # We want E in [0, 1].
        # If variance is 0, E = 1.
        # If variance is high, E -> 0.
        # We need a normalization factor.
        # Let's use the mean magnitude of SHAP values as a baseline.
        mean_magnitude = np.mean(np.abs(shap_stack))
        
        if mean_magnitude == 0:
            return 0.0
            
        normalized_variance = mean_variance / (mean_magnitude ** 2 + 1e-6)
        
        # Sigmoid-like squashing or just 1 / (1 + var)
        score = 1.0 / (1.0 + normalized_variance)
        
        return float(score)
