# Experimental Results - E-NSGA-II + X-MARL

## Performance Comparison Across All Methods

| Method | Ann. Return | Volatility | Sharpe | Max DD | CVaR 95% | Sortino |
|--------|-------------|------------|--------|--------|----------|----------|
| Equal-Weight | 11.70% | 18.4% | 0.42 | -18.42% | 2.61% | 0.68 |
| Min-Variance | 11.70% | 18.4% | 0.42 | -18.42% | 2.61% | 0.68 |
| Risk-Parity | -29.01% | 28.8% | -1.15 | -22.56% | 4.37% | -0.38 |
| LSTM | 3.92% | 3.2% | -0.02 | -1.91% | 0.28% | 0.54 |
| DDPG | 9.63% | 18.4% | 0.31 | -18.56% | 2.64% | 0.57 |
| SINGLE PPO | 11.91% | 17.5% | 0.45 | -17.33% | 2.47% | 0.74 |
| PURE NSGA II | 29.42% | 25.4% | 1.10 | -22.15% | 3.64% | 1.38 |
| Ablation-1 (δ=0) | 11.16% | 17.6% | 0.40 | -17.69% | 2.51% | 0.66 |
| Ablation-2 (Single-Agent) | 10.34% | 17.8% | 0.36 | -17.81% | 2.54% | 0.61 |
| **E-NSGA-II + X-MARL (Ours)** | 11.93% | 17.6% | 0.45 | -17.47% | 2.49% | 0.72 |


## Notes

- **Bold**: Our proposed method (E-NSGA-II + X-MARL)
- All metrics computed on test set (2025 data)
- Deep learning methods show mean values across random seeds
- Negative returns reflect challenging market period in test data
- Best performing method in each column would be highlighted in paper
