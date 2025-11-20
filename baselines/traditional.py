import numpy as np
import pandas as pd
from scipy.optimize import minimize

class TraditionalBaselines:
    def __init__(self, env):
        self.env = env
        self.n_assets = env.n_assets
        
    def equal_weight(self):
        """
        Returns equal weights for all assets (excluding cash).
        """
        # 1/N for assets, 0 for cash
        w = np.ones(self.n_assets) / self.n_assets
        return np.concatenate([w, [0.0]])
        
    def min_variance(self, lookback=60):
        """
        Minimum Variance Portfolio.
        """
        # Get historical returns
        current_step = self.env.current_step
        current_date = self.env.dates[current_step]
        
        # We need past data
        # The env has self.df which is the full dataset.
        # We should only peek at past data.
        
        # Get data up to current_date (exclusive)
        # Or inclusive if we rebalance at close?
        # Let's assume we use data available at decision time.
        
        past_data = self.env.df.loc[:current_date].iloc[-lookback-1:-1] # Exclude current day?
        # If we are at Open, we know Close t-1.
        
        if len(past_data) < lookback:
            return self.equal_weight()
            
        # Compute covariance
        # We need returns of assets.
        # Extract 'ret_1' for all tickers
        returns = past_data['ret_1'].unstack()
        cov = returns.cov().values
        
        # Optimization
        def objective(w):
            return w.T @ cov @ w
            
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(self.n_assets))
        
        init_w = np.ones(self.n_assets) / self.n_assets
        
        res = minimize(objective, init_w, method='SLSQP', bounds=bounds, constraints=constraints)
        
        if res.success:
            return np.concatenate([res.x, [0.0]])
        else:
            return self.equal_weight()

    def risk_parity(self, lookback=60):
        """
        Risk Parity Portfolio (Equal Risk Contribution).
        """
        # Simplified: Inverse Volatility
        # w_i \propto 1/sigma_i
        
        current_step = self.env.current_step
        current_date = self.env.dates[current_step]
        past_data = self.env.df.loc[:current_date].iloc[-lookback-1:-1]
        
        if len(past_data) < lookback:
            return self.equal_weight()
            
        returns = past_data['ret_1'].unstack()
        stds = returns.std().values
        
        inv_stds = 1 / (stds + 1e-8)
        w = inv_stds / np.sum(inv_stds)
        
        return np.concatenate([w, [0.0]])
