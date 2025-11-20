"""
Risk Metrics Module

Implements various risk measures for portfolio optimization:
- CVaR (Conditional Value at Risk) - also known as Expected Shortfall
- VaR (Value at Risk)
- Sortino Ratio
- Maximum Drawdown

References:
    Rockafellar, R. T., & Uryasev, S. (2000). Optimization of conditional
    value-at-risk. Journal of Risk, 2, 21-42.
"""

import numpy as np
from typing import Optional, Union


def calculate_var(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) at given confidence level.

    VaR is the maximum loss expected (at a given confidence level) over a time period.

    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95%)

    Returns:
        VaR value (positive number representing loss threshold)
    """
    if len(returns) == 0:
        return 0.0

    # VaR is the (1-alpha) quantile of the loss distribution
    # Since we have returns, losses are negative returns
    losses = -returns
    var = np.quantile(losses, confidence_level)

    return float(var)


def calculate_cvar(returns: np.ndarray, confidence_level: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR) / Expected Shortfall.

    CVaR is the expected loss given that the loss exceeds VaR.
    It's a more conservative risk measure than VaR.

    Formula: CVaR_α = E[Loss | Loss ≥ VaR_α]

    Args:
        returns: Array of returns
        confidence_level: Confidence level (e.g., 0.95 for 95% CVaR)

    Returns:
        CVaR value (positive number representing expected tail loss)
    """
    if len(returns) == 0:
        return 0.0

    # Calculate VaR first
    var = calculate_var(returns, confidence_level)

    # CVaR is the mean of losses that exceed VaR
    losses = -returns
    tail_losses = losses[losses >= var]

    if len(tail_losses) == 0:
        # If no losses exceed VaR, use VaR itself
        cvar = var
    else:
        cvar = np.mean(tail_losses)

    return float(cvar)


def calculate_sortino_ratio(
    returns: np.ndarray,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino Ratio.

    Similar to Sharpe ratio but only penalizes downside volatility.

    Formula: Sortino = (Mean Return - Target) / Downside Deviation

    Args:
        returns: Array of returns
        target_return: Target/minimum acceptable return (default 0)
        periods_per_year: Periods per year for annualization (252 for daily)

    Returns:
        Sortino ratio (higher is better)
    """
    if len(returns) == 0:
        return 0.0

    mean_return = np.mean(returns)

    # Downside deviation: std of returns below target
    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        # No downside -> infinite Sortino (clip to large value)
        return 10.0

    downside_std = np.std(downside_returns)

    if downside_std < 1e-8:
        return 10.0

    # Annualize
    sortino = (mean_return - target_return) / downside_std * np.sqrt(periods_per_year)

    return float(sortino)


def calculate_maximum_drawdown(cumulative_returns: np.ndarray) -> float:
    """
    Calculate Maximum Drawdown.

    MaxDD is the largest peak-to-trough decline in cumulative returns.

    Args:
        cumulative_returns: Cumulative return series (e.g., portfolio values)

    Returns:
        Maximum drawdown as positive fraction (e.g., 0.20 for 20% drawdown)
    """
    if len(cumulative_returns) == 0:
        return 0.0

    # Calculate running maximum
    running_max = np.maximum.accumulate(cumulative_returns)

    # Calculate drawdown at each point
    drawdowns = (cumulative_returns - running_max) / (running_max + 1e-8)

    # Maximum drawdown is the worst (most negative) drawdown
    max_dd = abs(np.min(drawdowns))

    return float(max_dd)


def calculate_downside_deviation(
    returns: np.ndarray,
    target_return: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate annualized downside deviation.

    Args:
        returns: Array of returns
        target_return: Target return threshold
        periods_per_year: Periods for annualization

    Returns:
        Annualized downside deviation
    """
    if len(returns) == 0:
        return 0.0

    downside_returns = returns[returns < target_return]

    if len(downside_returns) == 0:
        return 0.0

    downside_std = np.std(downside_returns)
    annualized_dd = downside_std * np.sqrt(periods_per_year)

    return float(annualized_dd)


class RiskMetricsCalculator:
    """
    Helper class for calculating risk metrics from portfolio history.

    Designed to work with portfolio environment's history format.
    """

    def __init__(self, periods_per_year: int = 252):
        """
        Initialize risk calculator.

        Args:
            periods_per_year: Trading periods per year (252 for daily)
        """
        self.periods_per_year = periods_per_year

    def calculate_from_history(
        self,
        portfolio_history: list,
        confidence_level: float = 0.95
    ) -> dict:
        """
        Calculate all risk metrics from portfolio history.

        Args:
            portfolio_history: List of dicts with 'return', 'value' keys
            confidence_level: For VaR/CVaR calculation

        Returns:
            Dictionary of risk metrics
        """
        if len(portfolio_history) == 0:
            return {
                'var': 0.0,
                'cvar': 0.0,
                'sortino': 0.0,
                'max_drawdown': 0.0,
                'downside_deviation': 0.0
            }

        # Extract returns
        returns = np.array([h['return'] for h in portfolio_history])
        values = np.array([h['value'] for h in portfolio_history])

        # Calculate metrics
        var = calculate_var(returns, confidence_level)
        cvar = calculate_cvar(returns, confidence_level)
        sortino = calculate_sortino_ratio(returns, 0.0, self.periods_per_year)
        max_dd = calculate_maximum_drawdown(values)
        downside_dev = calculate_downside_deviation(returns, 0.0, self.periods_per_year)

        return {
            'var': var,
            'cvar': cvar,
            'sortino': sortino,
            'max_drawdown': max_dd,
            'downside_deviation': downside_dev
        }


if __name__ == "__main__":
    # Example usage
    print("Risk Metrics Module")
    print("=" * 50)

    # Generate sample returns (normal distribution)
    np.random.seed(42)
    sample_returns = np.random.normal(0.001, 0.02, 252)  # Daily returns for 1 year

    print(f"\nSample returns statistics:")
    print(f"  Mean: {np.mean(sample_returns):.6f}")
    print(f"  Std: {np.std(sample_returns):.6f}")

    # Calculate VaR and CVaR
    var_95 = calculate_var(sample_returns, 0.95)
    cvar_95 = calculate_cvar(sample_returns, 0.95)

    print(f"\nRisk Measures (95% confidence):")
    print(f"  VaR: {var_95:.6f}")
    print(f"  CVaR (Expected Shortfall): {cvar_95:.6f}")

    # Calculate Sortino
    sortino = calculate_sortino_ratio(sample_returns)
    print(f"\nSortino Ratio: {sortino:.4f}")

    # Calculate Max Drawdown
    cumulative = np.cumprod(1 + sample_returns)
    max_dd = calculate_maximum_drawdown(cumulative)
    print(f"Maximum Drawdown: {max_dd:.4f} ({max_dd*100:.2f}%)")

    print("\n" + "=" * 50)
    print("All metrics calculated successfully!")
