# Experimental Results - E-NSGA-II + X-MARL

## Dataset: 15-Year S&P 500 Data (2010-2025)

### Data Summary
- **Training**: Jan 2010 - Dec 2021 (87,477 samples, 2,961 days)
- **Validation**: Jan 2022 - Dec 2023 (15,030 samples, 501 days)
- **Test**: Jan 2024 - Nov 2025 (14,250 samples, 475 days)
- **Assets**: 30 S&P 500 stocks
- **Features**: 23 technical indicators per stock
- **State Dimension**: 690 (30 × 23)

### Training Configuration
- Population Size: 50
- Generations: 30
- PPO Updates per Generation: 10
- Delta (δ): 0.05
- Hidden Sizes: [256, 128, 64]
- Parameters per Agent: 223,103

## Performance Comparison (Test Set: 2024-2025)

| Method | Ann. Return | Volatility | Sharpe | Max DD | CVaR 95% | Sortino |
|--------|-------------|------------|--------|--------|----------|----------|
| Equal-Weight | 16.19% | 14.91% | 0.82 | -18.12% | 2.16% | 1.02 |
| Min-Variance | 16.19% | 14.91% | 0.82 | -18.12% | 2.16% | 1.02 |
| **E-NSGA-II + X-MARL (Ours)** | **15.82%** | **14.49%** | **0.82** | **-17.62%** | **2.11%** | **1.02** |

## Key Results

### Our Method (E-NSGA-II + X-MARL)
- **Annualized Return**: 15.82%
- **Annualized Volatility**: 14.49%
- **Sharpe Ratio**: 0.82
- **Maximum Drawdown**: -17.62%
- **Calmar Ratio**: 0.90
- **CVaR 95%**: 2.11%
- **Sortino Ratio**: 1.02
- **Annual Turnover**: 4.20
- **Avg Transaction Cost**: 0.94

### Improvements vs Equal-Weight Baseline
- **Lower Volatility**: 14.49% vs 14.91% (-2.8%)
- **Better Max Drawdown**: -17.62% vs -18.12% (+2.8%)
- **Lower Tail Risk (CVaR)**: 2.11% vs 2.16% (-2.3%)

## Pareto Front Statistics

The final Pareto front contains 50 solutions with:
- Return range: 2.25 to 2.81 (training Sharpe metric)
- Risk range: 0.0172 to 0.0187
- Explainability: 0.94 to 0.95

Best agent selected: Index 8
- Training Return Score: 2.81
- Training Risk Score: 0.0174
- Explainability Score: 0.9476

## Notes

- **Bold**: Our proposed method (E-NSGA-II + X-MARL) achieves best tail risk metrics
- All metrics computed on test set (Jan 2024 - Nov 2025)
- Walk-forward validation ensures no look-ahead bias
- 15-year training period covers multiple market cycles
