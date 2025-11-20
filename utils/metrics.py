import numpy as np
import pandas as pd
from typing import List, Optional
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.risk_metrics import calculate_cvar, calculate_sortino_ratio

def calculate_metrics(portfolio_history, risk_free_rate=0.04):
    """
    Calculates performance metrics from portfolio history.
    portfolio_history: List of dicts with keys 'date', 'value', 'return', 'cost', 'turnover'
    """
    df = pd.DataFrame(portfolio_history)
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    
    returns = df['return']
    
    # 1. Annualized Return
    # Total return = (End Value / Start Value) - 1
    total_ret = (df['value'].iloc[-1] / df['value'].iloc[0]) - 1
    # Annualized: (1 + total_ret)^(252/N) - 1
    n_days = len(df)
    ann_ret = (1 + total_ret) ** (252 / n_days) - 1
    
    # 2. Annualized Volatility
    ann_vol = returns.std() * np.sqrt(252)
    
    # 3. Sharpe Ratio
    sharpe = (ann_ret - risk_free_rate) / (ann_vol + 1e-8)
    
    # 4. Max Drawdown
    cum_ret = (1 + returns).cumprod()
    roll_max = cum_ret.cummax()
    drawdown = (cum_ret - roll_max) / roll_max
    max_dd = drawdown.min()
    
    # 5. Calmar Ratio
    calmar = ann_ret / abs(max_dd + 1e-8)
    
    # 6. Annual Turnover
    # Mean daily turnover * 252
    ann_turnover = df['turnover'].mean() * 252
    
    # 7. Average Transaction Cost
    avg_cost = df['cost'].mean()

    # 8. CVaR (95%)
    cvar_95 = calculate_cvar(returns.values, confidence_level=0.95)

    # 9. Sortino Ratio
    sortino = calculate_sortino_ratio(returns.values, target_return=risk_free_rate/252, periods_per_year=252)

    return {
        "Annualized Return": ann_ret,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Calmar Ratio": calmar,
        "Annual Turnover": ann_turnover,
        "Avg Transaction Cost": avg_cost,
        "CVaR 95%": cvar_95,
        "Sortino Ratio": sortino
    }


def calculate_hypervolume(pareto_front: np.ndarray, reference_point: Optional[np.ndarray] = None) -> float:
    """
    Calculate hypervolume indicator for Pareto front quality assessment.

    Hypervolume is the volume of objective space dominated by the Pareto front.
    Higher values indicate better Pareto front quality.

    Args:
        pareto_front: Array of shape (n_solutions, n_objectives)
                     Each row is a solution with objective values
        reference_point: Worst point for each objective (for maximization problems)
                        If None, uses minimum of each objective - 1

    Returns:
        Hypervolume value (higher = better)

    Note: This implements a simple 3D hypervolume calculation.
          For production use, consider using pygmo or pymoo libraries.
    """
    if len(pareto_front) == 0:
        return 0.0

    n_objectives = pareto_front.shape[1]

    if reference_point is None:
        # For maximization: reference is below all points
        reference_point = np.min(pareto_front, axis=0) - 1.0

    # Simple hypervolume calculation for 2D and 3D
    if n_objectives == 2:
        # 2D case: sort by first objective
        sorted_front = pareto_front[np.argsort(pareto_front[:, 0])]

        hv = 0.0
        for i in range(len(sorted_front)):
            if i == 0:
                width = sorted_front[i, 0] - reference_point[0]
            else:
                width = sorted_front[i, 0] - sorted_front[i-1, 0]

            height = sorted_front[i, 1] - reference_point[1]
            hv += width * height

        return hv

    elif n_objectives == 3:
        # 3D case: Monte Carlo approximation
        # Sample points in the objective space
        n_samples = 10000

        # Determine bounds
        min_point = np.minimum(reference_point, np.min(pareto_front, axis=0))
        max_point = np.max(pareto_front, axis=0)

        # Generate random points
        random_points = np.random.uniform(
            min_point, max_point, size=(n_samples, n_objectives)
        )

        # Count points dominated by at least one solution in Pareto front
        dominated_count = 0
        for point in random_points:
            # Check if point is dominated by any solution
            for solution in pareto_front:
                # For maximization: solution dominates point if solution >= point in all objectives
                if np.all(solution >= point):
                    dominated_count += 1
                    break

        # Hypervolume = (fraction dominated) * (volume of bounding box)
        box_volume = np.prod(max_point - min_point)
        hv = (dominated_count / n_samples) * box_volume

        return hv

    else:
        # For higher dimensions, return approximation or use external library
        print(f"Warning: Hypervolume calculation for {n_objectives}D not fully implemented")
        return 0.0


def calculate_generational_distance(pareto_front: np.ndarray, true_pareto: np.ndarray) -> float:
    """
    Calculate Generational Distance (GD) metric.

    GD measures how far the obtained Pareto front is from the true Pareto front.
    Lower values are better.

    Args:
        pareto_front: Obtained Pareto front
        true_pareto: True/reference Pareto front

    Returns:
        GD value (lower = better, 0 = perfect)
    """
    if len(pareto_front) == 0 or len(true_pareto) == 0:
        return float('inf')

    distances = []
    for point in pareto_front:
        # Find minimum Euclidean distance to true Pareto front
        min_dist = np.min(np.linalg.norm(true_pareto - point, axis=1))
        distances.append(min_dist)

    gd = np.mean(distances)
    return gd


def calculate_spread(pareto_front: np.ndarray) -> float:
    """
    Calculate spread/diversity metric for Pareto front.

    Measures how well-distributed solutions are along the Pareto front.
    Lower values indicate better spread.

    Args:
        pareto_front: Pareto front solutions

    Returns:
        Spread value (lower = better distribution)
    """
    if len(pareto_front) < 2:
        return 0.0

    # Calculate pairwise distances
    distances = []
    for i in range(len(pareto_front)):
        for j in range(i + 1, len(pareto_front)):
            dist = np.linalg.norm(pareto_front[i] - pareto_front[j])
            distances.append(dist)

    if len(distances) == 0:
        return 0.0

    # Spread is coefficient of variation of distances
    mean_dist = np.mean(distances)
    std_dist = np.std(distances)

    if mean_dist < 1e-8:
        return 0.0

    spread = std_dist / mean_dist
    return spread
