#!/bin/bash

################################################################################
# Quick Test Script for E-NSGA-II + X-MARL
#
# Tests the implementation with minimal data and parameters.
# Use this to verify everything works before running full experiments.
#
# Usage: ./quick_test.sh
################################################################################

set -e

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}"
echo "================================================================================"
echo "  E-NSGA-II + X-MARL - Quick Test"
echo "================================================================================"
echo -e "${NC}"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "✓ Virtual environment activated"
else
    echo "⚠ Virtual environment not found. Using system Python."
fi

# Test 1: Import all modules
echo ""
echo -e "${GREEN}Test 1: Importing all modules...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')

print('  Importing data modules...')
from data.downloader import get_sp500_tickers
from data.preprocessor import compute_features

print('  Importing environment...')
from env.portfolio_env import PortfolioEnv

print('  Importing agents...')
from models.agents import ReturnAgent, RiskAgent, ExplainAgent

print('  Importing algorithms...')
from algorithms.nsga_ii import NSGA2

print('  Importing utils...')
from utils.metrics import calculate_metrics, calculate_hypervolume
from utils.risk_metrics import calculate_cvar, calculate_sortino_ratio
from utils.explainability import ExplainabilityScorer

print('  Importing baselines...')
from baselines.traditional import TraditionalBaselines
from baselines.pure_nsga_ii import PureNSGAII
from baselines.lstm_baseline import LSTMBaseline
from baselines.ddpg_baseline import DDPGAgent
from baselines.single_ppo import PPOAgent
from baselines.ablations import NSGA2NoDominance

print('✓ All imports successful!')
"

# Test 2: Check data files
echo ""
echo -e "${GREEN}Test 2: Checking data files...${NC}"
if [ -f "data/train.csv" ] && [ -f "data/val.csv" ] && [ -f "data/test.csv" ]; then
    echo "✓ All data files present"
    for file in data/train.csv data/val.csv data/test.csv; do
        lines=$(wc -l < "$file")
        echo "  - $file: $lines rows"
    done
else
    echo "⚠ Data files missing. Run: python data/downloader.py && python data/preprocessor.py"
fi

# Test 3: Test environment creation
echo ""
echo -e "${GREEN}Test 3: Testing environment...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from env.portfolio_env import PortfolioEnv

# Load test data
test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)
print(f'  Loaded test data: {len(test_df)} rows')

# Create environment
env = PortfolioEnv(test_df)
print(f'  Environment created: {env.n_assets} assets, {env.n_features} features')

# Test reset and step
obs, _ = env.reset()
print(f'  Reset successful: obs shape = {obs[\"features\"].shape}')

# Test random action
action = np.random.randn(env.n_assets + 1)
obs, rewards, terminated, truncated, _ = env.step(action)
print(f'  Step successful: rewards = {rewards}')
print(f'  Reward components: Sharpe={rewards[0]:.4f}, Risk={rewards[1]:.4f}, Explain={rewards[2]:.4f}')

print('✓ Environment working correctly!')
"

# Test 4: Test metrics calculation
echo ""
echo -e "${GREEN}Test 4: Testing metrics...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
from utils.risk_metrics import calculate_cvar, calculate_sortino_ratio
from utils.metrics import calculate_hypervolume

# Test CVaR
returns = np.random.normal(0.001, 0.02, 100)
cvar = calculate_cvar(returns, 0.95)
print(f'  CVaR-95%: {cvar:.6f}')

# Test Sortino
sortino = calculate_sortino_ratio(returns)
print(f'  Sortino Ratio: {sortino:.6f}')

# Test Hypervolume (2D)
pareto_front = np.array([[1.0, 0.5], [0.8, 0.7], [0.6, 0.9]])
hv = calculate_hypervolume(pareto_front)
print(f'  Hypervolume (2D): {hv:.6f}')

# Test Hypervolume (3D)
pareto_front_3d = np.array([[1.0, 0.5, 0.6], [0.8, 0.7, 0.7], [0.6, 0.9, 0.5]])
hv_3d = calculate_hypervolume(pareto_front_3d)
print(f'  Hypervolume (3D): {hv_3d:.6f}')

print('✓ Metrics working correctly!')
"

# Test 5: Quick baseline test
echo ""
echo -e "${GREEN}Test 5: Testing traditional baseline...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
from env.portfolio_env import PortfolioEnv
from baselines.traditional import TraditionalBaselines
from utils.metrics import calculate_metrics

# Load small sample of test data
test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)
test_df_small = test_df.iloc[:500]  # Use only 500 rows for quick test

env = PortfolioEnv(test_df_small)
bl = TraditionalBaselines(env)

# Test equal-weight
env.reset()
steps = 0
while env.current_step < min(50, len(env.dates) - 1):  # Only 50 steps
    weights = bl.equal_weight()
    action = np.log(weights + 1e-8) * 100
    env.step(action)
    steps += 1

metrics = calculate_metrics(env.portfolio_history)
print(f'  Equal-weight (50 steps):')
print(f'    Sharpe Ratio: {metrics[\"Sharpe Ratio\"]:.4f}')
print(f'    Ann. Return: {metrics[\"Annualized Return\"]*100:.2f}%')
print(f'    Ann. Volatility: {metrics[\"Annualized Volatility\"]*100:.2f}%')

print('✓ Traditional baseline working!')
"

# Test 6: Test NSGA-II framework
echo ""
echo -e "${GREEN}Test 6: Testing NSGA-II framework...${NC}"
python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
from algorithms.nsga_ii import NSGA2

# Create dummy population
class DummyIndividual:
    def __init__(self, value):
        self.value = value

def dummy_eval(ind):
    # Return random multi-objective fitness
    return [np.random.rand(), np.random.rand(), np.random.rand()]

nsga = NSGA2(population_size=5, generations=2)
population = [DummyIndividual(i) for i in range(5)]

print('  Running 2 generations with 5 individuals...')
final_pop, final_fitness = nsga.evolve(population, dummy_eval)

print(f'  Final population size: {len(final_pop)}')
print(f'  Final fitness values: {len(final_fitness)}')
print(f'  Sample fitness: {final_fitness[0]}')

print('✓ NSGA-II framework working!')
"

# Summary
echo ""
echo -e "${GREEN}================================================================================"
echo "  All Quick Tests Passed!"
echo "================================================================================${NC}"
echo ""
echo "You can now run full experiments with:"
echo "  ./run_all_experiments.sh --quick        (Quick mode with reduced parameters)"
echo "  ./run_all_experiments.sh                (Full experiments)"
echo ""
echo "Or run specific components:"
echo "  python data/downloader.py               (Download data)"
echo "  python data/preprocessor.py             (Preprocess data)"
echo "  python main.py --test                   (Quick test of main method)"
echo ""

# Deactivate if we activated
if [ -d "venv" ]; then
    deactivate
fi

exit 0
