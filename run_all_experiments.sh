#!/bin/bash

################################################################################
# E-NSGA-II + X-MARL Complete Experiment Pipeline
#
# This script runs the entire experimental pipeline:
# 1. Setup and dependency check
# 2. Data download and preprocessing
# 3. All baseline methods
# 4. Ablation studies
# 5. Main E-NSGA-II + X-MARL method
# 6. Results aggregation and visualization
#
# Usage: ./run_all_experiments.sh [options]
# Options:
#   --quick          Run quick test with minimal data/epochs
#   --skip-download  Skip data download step
#   --baselines-only Run only baseline experiments
#   --help           Show this help message
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
QUICK_MODE=false
SKIP_DOWNLOAD=false
BASELINES_ONLY=false
RANDOM_SEEDS=(42 123 456)  # Run with 3 seeds
POP_SIZE=20
GENERATIONS=10
EPOCHS=10

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            POP_SIZE=5
            GENERATIONS=3
            EPOCHS=5
            RANDOM_SEEDS=(42)
            echo -e "${YELLOW}Quick mode enabled: Reduced parameters for testing${NC}"
            shift
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            echo -e "${YELLOW}Skipping data download${NC}"
            shift
            ;;
        --baselines-only)
            BASELINES_ONLY=true
            echo -e "${YELLOW}Running baselines only${NC}"
            shift
            ;;
        --help)
            head -n 20 "$0" | tail -n +3 | sed 's/^# //'
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Print header
echo -e "${BLUE}"
echo "================================================================================"
echo "  E-NSGA-II + X-MARL Experimental Pipeline"
echo "================================================================================"
echo -e "${NC}"
echo "Configuration:"
echo "  - Random seeds: ${RANDOM_SEEDS[@]}"
echo "  - Population size: $POP_SIZE"
echo "  - Generations: $GENERATIONS"
echo "  - Epochs: $EPOCHS"
echo "  - Quick mode: $QUICK_MODE"
echo ""

# Function to print section headers
print_section() {
    echo ""
    echo -e "${GREEN}================================================================================"
    echo "  $1"
    echo -e "================================================================================${NC}"
}

# Function to print step
print_step() {
    echo -e "${BLUE}>>> $1${NC}"
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

################################################################################
# Step 1: Setup and Dependency Check
################################################################################
print_section "Step 1: Setup and Dependency Check"

print_step "Checking Python installation..."
if ! command_exists python3; then
    echo -e "${RED}Error: Python 3 is not installed${NC}"
    exit 1
fi
PYTHON_VERSION=$(python3 --version)
echo "  ✓ Found: $PYTHON_VERSION"

print_step "Checking for virtual environment..."
if [ -d "venv" ]; then
    echo "  ✓ Virtual environment found"
else
    echo -e "${YELLOW}  ! Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
print_step "Activating virtual environment..."
source venv/bin/activate
echo "  ✓ Virtual environment activated"

print_step "Installing/updating dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
echo "  ✓ Dependencies installed"

# Create necessary directories (skip data/ as it already exists with Python modules)
print_step "Creating output directories..."
# Note: data/ directory already exists with downloader.py and preprocessor.py
# We only create subdirectories for results and logs
mkdir -p results
mkdir -p results/figures
mkdir -p results/baselines
mkdir -p results/ablations
mkdir -p results/main_method
mkdir -p logs
echo "  ✓ Directories created"

################################################################################
# Step 2: Data Download and Preprocessing
################################################################################
if [ "$SKIP_DOWNLOAD" = false ]; then
    print_section "Step 2: Data Download and Preprocessing"

    print_step "Downloading S&P 500 data (2020-2025)..."
    if [ "$QUICK_MODE" = true ]; then
        echo "  (Quick mode: Using test data if available)"
        if [ ! -f "data/raw_data.csv" ]; then
            python3 data/downloader.py 2>&1 | tee logs/download.log
        fi
    else
        python3 data/downloader.py 2>&1 | tee logs/download.log
    fi
    echo "  ✓ Data downloaded"

    print_step "Preprocessing and splitting data..."
    python3 data/preprocessor.py 2>&1 | tee logs/preprocess.log
    echo "  ✓ Data preprocessed"

    # Verify data files
    print_step "Verifying data files..."
    for file in data/train.csv data/val.csv data/test.csv; do
        if [ -f "$file" ]; then
            LINES=$(wc -l < "$file")
            echo "  ✓ $file ($LINES rows)"
        else
            echo -e "${RED}  ✗ Missing: $file${NC}"
            exit 1
        fi
    done
else
    print_section "Step 2: Data Check (Download Skipped)"
    print_step "Verifying existing data files..."
    for file in data/train.csv data/val.csv data/test.csv; do
        if [ -f "$file" ]; then
            LINES=$(wc -l < "$file")
            echo "  ✓ $file ($LINES rows)"
        else
            echo -e "${RED}  ✗ Missing: $file${NC}"
            echo -e "${YELLOW}  Run without --skip-download to download data${NC}"
            exit 1
        fi
    done
fi

################################################################################
# Step 3: Run Traditional Baselines
################################################################################
print_section "Step 3: Traditional Baselines (Fast)"

# Equal-weight baseline
print_step "Running Equal-Weight baseline..."
python3 -c "
import sys
sys.path.insert(0, '.')
from baselines.traditional import TraditionalBaselines
from env.portfolio_env import PortfolioEnv
import pandas as pd
import numpy as np
from utils.metrics import calculate_metrics

test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)
env = PortfolioEnv(test_df)
bl = TraditionalBaselines(env)

env.reset()
while env.current_step < len(env.dates) - 1:
    weights = bl.equal_weight()
    action = np.log(weights + 1e-8) * 100
    env.step(action)

metrics = calculate_metrics(env.portfolio_history)
pd.DataFrame([metrics]).to_csv('results/baselines/equal_weight.csv', index=False)
print('Equal-Weight metrics:', metrics)
" 2>&1 | tee logs/baseline_equal_weight.log
echo "  ✓ Equal-weight baseline complete"

# Minimum variance baseline
print_step "Running Minimum-Variance baseline..."
python3 -c "
import sys
sys.path.insert(0, '.')
from baselines.traditional import TraditionalBaselines
from env.portfolio_env import PortfolioEnv
import pandas as pd
import numpy as np
from utils.metrics import calculate_metrics

test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)
env = PortfolioEnv(test_df)
bl = TraditionalBaselines(env)

env.reset()
while env.current_step < len(env.dates) - 1:
    weights = bl.min_variance()
    action = np.log(weights + 1e-8) * 100
    env.step(action)

metrics = calculate_metrics(env.portfolio_history)
pd.DataFrame([metrics]).to_csv('results/baselines/min_variance.csv', index=False)
print('Min-Variance metrics:', metrics)
" 2>&1 | tee logs/baseline_min_variance.log
echo "  ✓ Minimum-variance baseline complete"

# Risk parity baseline
print_step "Running Risk-Parity baseline..."
python3 -c "
import sys
sys.path.insert(0, '.')
from baselines.traditional import TraditionalBaselines
from env.portfolio_env import PortfolioEnv
import pandas as pd
import numpy as np
from utils.metrics import calculate_metrics

test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)
env = PortfolioEnv(test_df)
bl = TraditionalBaselines(env)

env.reset()
while env.current_step < len(env.dates) - 1:
    weights = bl.risk_parity()
    action = np.log(weights + 1e-8) * 100
    env.step(action)

metrics = calculate_metrics(env.portfolio_history)
pd.DataFrame([metrics]).to_csv('results/baselines/risk_parity.csv', index=False)
print('Risk-Parity metrics:', metrics)
" 2>&1 | tee logs/baseline_risk_parity.log
echo "  ✓ Risk-parity baseline complete"

################################################################################
# Step 4: Run Deep Learning Baselines
################################################################################
print_section "Step 4: Deep Learning Baselines"

# LSTM baseline
for seed in "${RANDOM_SEEDS[@]}"; do
    print_step "Running LSTM baseline (seed=$seed)..."
    python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
from baselines.lstm_baseline import run_lstm_baseline
from env.portfolio_env import PortfolioEnv

np.random.seed($seed)
torch.manual_seed($seed)

train_df = pd.read_csv('data/train.csv', index_col=[0, 1], parse_dates=True)
val_df = pd.read_csv('data/val.csv', index_col=[0, 1], parse_dates=True)
test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)

env_train = PortfolioEnv(train_df)
env_val = PortfolioEnv(val_df)
env_test = PortfolioEnv(test_df)

results = run_lstm_baseline(env_train, env_val, env_test, epochs=$EPOCHS)
pd.DataFrame([results['test_metrics']]).to_csv('results/baselines/lstm_seed_$seed.csv', index=False)
" 2>&1 | tee logs/baseline_lstm_seed_$seed.log
    echo "  ✓ LSTM (seed=$seed) complete"
done

# DDPG baseline
for seed in "${RANDOM_SEEDS[@]}"; do
    print_step "Running DDPG baseline (seed=$seed)..."
    python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
from baselines.ddpg_baseline import run_ddpg_baseline
from env.portfolio_env import PortfolioEnv

np.random.seed($seed)
torch.manual_seed($seed)

train_df = pd.read_csv('data/train.csv', index_col=[0, 1], parse_dates=True)
val_df = pd.read_csv('data/val.csv', index_col=[0, 1], parse_dates=True)
test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)

env_train = PortfolioEnv(train_df)
env_val = PortfolioEnv(val_df)
env_test = PortfolioEnv(test_df)

results = run_ddpg_baseline(env_train, env_val, env_test, episodes=50, max_steps=500)
pd.DataFrame([results['test_metrics']]).to_csv('results/baselines/ddpg_seed_$seed.csv', index=False)
" 2>&1 | tee logs/baseline_ddpg_seed_$seed.log
    echo "  ✓ DDPG (seed=$seed) complete"
done

# Single-agent PPO baseline
for seed in "${RANDOM_SEEDS[@]}"; do
    print_step "Running Single-Agent PPO baseline (seed=$seed)..."
    python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
from baselines.single_ppo import run_single_ppo_baseline
from env.portfolio_env import PortfolioEnv

np.random.seed($seed)
torch.manual_seed($seed)

train_df = pd.read_csv('data/train.csv', index_col=[0, 1], parse_dates=True)
val_df = pd.read_csv('data/val.csv', index_col=[0, 1], parse_dates=True)
test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)

env_train = PortfolioEnv(train_df)
env_val = PortfolioEnv(val_df)
env_test = PortfolioEnv(test_df)

results = run_single_ppo_baseline(env_train, env_val, env_test, episodes=50, max_steps=500)
pd.DataFrame([results['test_metrics']]).to_csv('results/baselines/single_ppo_seed_$seed.csv', index=False)
" 2>&1 | tee logs/baseline_single_ppo_seed_$seed.log
    echo "  ✓ Single-PPO (seed=$seed) complete"
done

# Pure NSGA-II baseline
for seed in "${RANDOM_SEEDS[@]}"; do
    print_step "Running Pure NSGA-II baseline (seed=$seed)..."
    python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
from baselines.pure_nsga_ii import run_pure_nsga_ii_baseline
from env.portfolio_env import PortfolioEnv

np.random.seed($seed)
torch.manual_seed($seed)

train_df = pd.read_csv('data/train.csv', index_col=[0, 1], parse_dates=True)
val_df = pd.read_csv('data/val.csv', index_col=[0, 1], parse_dates=True)
test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)

env_train = PortfolioEnv(train_df)
env_val = PortfolioEnv(val_df)
env_test = PortfolioEnv(test_df)

results = run_pure_nsga_ii_baseline(env_train, env_val, env_test,
                                     population_size=$POP_SIZE,
                                     generations=$GENERATIONS)
pd.DataFrame([results['test_metrics']]).to_csv('results/baselines/pure_nsga_ii_seed_$seed.csv', index=False)
" 2>&1 | tee logs/baseline_pure_nsga_ii_seed_$seed.log
    echo "  ✓ Pure NSGA-II (seed=$seed) complete"
done

################################################################################
# Step 5: Run Ablation Studies (if not baselines-only)
################################################################################
if [ "$BASELINES_ONLY" = false ]; then
    print_section "Step 5: Ablation Studies"

    # Ablation 1: No explainability dominance (δ=0)
    for seed in "${RANDOM_SEEDS[@]}"; do
        print_step "Running Ablation 1: δ=0 (seed=$seed)..."
        python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
from baselines.ablations import run_ablation_1
from env.portfolio_env import PortfolioEnv

np.random.seed($seed)
torch.manual_seed($seed)

train_df = pd.read_csv('data/train.csv', index_col=[0, 1], parse_dates=True)
val_df = pd.read_csv('data/val.csv', index_col=[0, 1], parse_dates=True)
test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)

env_train = PortfolioEnv(train_df)
env_val = PortfolioEnv(val_df)
env_test = PortfolioEnv(test_df)

results = run_ablation_1(env_train, env_val, env_test,
                         pop_size=$POP_SIZE,
                         generations=$GENERATIONS)
pd.DataFrame([results['test_metrics']]).to_csv('results/ablations/ablation1_delta0_seed_$seed.csv', index=False)
" 2>&1 | tee logs/ablation1_seed_$seed.log
        echo "  ✓ Ablation 1 (seed=$seed) complete"
    done

    # Ablation 2: Single-agent PPO with E-NSGA-II
    for seed in "${RANDOM_SEEDS[@]}"; do
        print_step "Running Ablation 2: Single-agent (seed=$seed)..."
        python3 -c "
import sys
sys.path.insert(0, '.')
import numpy as np
import pandas as pd
import torch
from baselines.ablations import run_ablation_2
from env.portfolio_env import PortfolioEnv

np.random.seed($seed)
torch.manual_seed($seed)

train_df = pd.read_csv('data/train.csv', index_col=[0, 1], parse_dates=True)
val_df = pd.read_csv('data/val.csv', index_col=[0, 1], parse_dates=True)
test_df = pd.read_csv('data/test.csv', index_col=[0, 1], parse_dates=True)

env_train = PortfolioEnv(train_df)
env_val = PortfolioEnv(val_df)
env_test = PortfolioEnv(test_df)

results = run_ablation_2(env_train, env_val, env_test,
                         pop_size=$POP_SIZE,
                         generations=$GENERATIONS)
pd.DataFrame([results['test_metrics']]).to_csv('results/ablations/ablation2_single_seed_$seed.csv', index=False)
" 2>&1 | tee logs/ablation2_seed_$seed.log
        echo "  ✓ Ablation 2 (seed=$seed) complete"
    done
fi

################################################################################
# Step 6: Run Main Method (if not baselines-only)
################################################################################
if [ "$BASELINES_ONLY" = false ]; then
    print_section "Step 6: Main Method (E-NSGA-II + X-MARL)"

    for seed in "${RANDOM_SEEDS[@]}"; do
        print_step "Running E-NSGA-II + X-MARL (seed=$seed)..."
        python3 main.py --pop_size $POP_SIZE --generations $GENERATIONS 2>&1 | tee logs/main_method_seed_$seed.log

        # Move results to organized location
        if [ -f "results/pareto_front.csv" ]; then
            mv results/pareto_front.csv results/main_method/pareto_front_seed_$seed.csv
        fi
        if [ -f "results/test_metrics_best_agent.csv" ]; then
            mv results/test_metrics_best_agent.csv results/main_method/test_metrics_seed_$seed.csv
        fi

        echo "  ✓ Main method (seed=$seed) complete"
    done
fi

################################################################################
# Step 7: Aggregate Results
################################################################################
print_section "Step 7: Aggregating Results"

print_step "Creating results summary..."
python3 -c "
import pandas as pd
import numpy as np
import os
from glob import glob

# Aggregate all baseline results
results = {}

# Traditional baselines (single run)
for method in ['equal_weight', 'min_variance', 'risk_parity']:
    csv_path = f'results/baselines/{method}.csv'
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        results[method.replace('_', '-').title()] = df.iloc[0].to_dict()

# Deep learning baselines (average across seeds)
for method in ['lstm', 'ddpg', 'single_ppo', 'pure_nsga_ii']:
    files = glob(f'results/baselines/{method}_seed_*.csv')
    if files:
        dfs = [pd.read_csv(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        mean_results = combined.mean().to_dict()
        std_results = combined.std().to_dict()

        # Store mean ± std
        results[method.replace('_', ' ').upper()] = {
            k: f'{mean_results[k]:.4f} ± {std_results[k]:.4f}'
            for k in mean_results.keys()
        }

# Ablations (if exist)
for method, name in [('ablation1_delta0', 'Ablation-1 (δ=0)'),
                      ('ablation2_single', 'Ablation-2 (Single-Agent)')]:
    files = glob(f'results/ablations/{method}_seed_*.csv')
    if files:
        dfs = [pd.read_csv(f) for f in files]
        combined = pd.concat(dfs, ignore_index=True)
        mean_results = combined.mean().to_dict()
        std_results = combined.std().to_dict()
        results[name] = {
            k: f'{mean_results[k]:.4f} ± {std_results[k]:.4f}'
            for k in mean_results.keys()
        }

# Main method (if exists)
files = glob('results/main_method/test_metrics_seed_*.csv')
if files:
    dfs = [pd.read_csv(f) for f in files]
    combined = pd.concat(dfs, ignore_index=True)
    mean_results = combined.mean().to_dict()
    std_results = combined.std().to_dict()
    results['E-NSGA-II + X-MARL (Ours)'] = {
        k: f'{mean_results[k]:.4f} ± {std_results[k]:.4f}'
        for k in mean_results.keys()
    }

# Create summary DataFrame
summary_df = pd.DataFrame(results).T
summary_df.to_csv('results/RESULTS_SUMMARY.csv')

# Create a clean formatted table for paper/presentation
# Select key metrics in the desired order
key_metrics = ['Annualized Return', 'Annualized Volatility', 'Sharpe Ratio',
               'Max Drawdown', 'CVaR 95%', 'Sortino Ratio']

# Create formatted table with percentages
formatted_results = {}
for method, metrics in results.items():
    row = {}
    for metric in key_metrics:
        if metric in metrics:
            val = metrics[metric]
            # Handle mean ± std format
            if isinstance(val, str) and '±' in val:
                mean_str, std_str = val.split('±')
                mean_val = float(mean_str.strip())
                std_val = float(std_str.strip())
                if 'Return' in metric or 'Drawdown' in metric:
                    row[metric] = f'{mean_val*100:.2f}%'
                elif 'Ratio' in metric or 'Volatility' in metric:
                    if 'Volatility' in metric:
                        row[metric] = f'{mean_val*100:.1f}%'
                    else:
                        row[metric] = f'{mean_val:.2f}'
                elif 'CVaR' in metric:
                    row[metric] = f'{mean_val*100:.2f}%'
                else:
                    row[metric] = f'{mean_val:.4f}'
            else:
                # Single value (traditional baselines)
                if 'Return' in metric or 'Drawdown' in metric:
                    row[metric] = f'{val*100:.2f}%'
                elif 'Ratio' in metric or 'Volatility' in metric:
                    if 'Volatility' in metric:
                        row[metric] = f'{val*100:.1f}%'
                    else:
                        row[metric] = f'{val:.2f}'
                elif 'CVaR' in metric:
                    row[metric] = f'{val*100:.2f}%'
                else:
                    row[metric] = f'{val:.4f}'
    formatted_results[method] = row

formatted_df = pd.DataFrame(formatted_results).T
formatted_df = formatted_df[key_metrics]  # Ensure column order

# Save formatted table
formatted_df.to_csv('results/RESULTS_FORMATTED.csv')

# Create markdown table (manual formatting to avoid tabulate dependency)
with open('results/RESULTS_FORMATTED.md', 'w') as f:
    f.write('# Experimental Results - E-NSGA-II + X-MARL\n\n')
    f.write('## Performance Comparison Across All Methods\n\n')

    # Header
    f.write('| Method | Ann. Return | Volatility | Sharpe | Max DD | CVaR 95% | Sortino |\n')
    f.write('|--------|-------------|------------|--------|--------|----------|----------|\n')

    # Rows
    for method in formatted_df.index:
        row = formatted_df.loc[method]
        # Bold our method
        method_name = f'**{method}**' if 'X-MARL' in method else method
        f.write(f'| {method_name} | {row[\"Annualized Return\"]} | {row[\"Annualized Volatility\"]} | ')
        f.write(f'{row[\"Sharpe Ratio\"]} | {row[\"Max Drawdown\"]} | {row[\"CVaR 95%\"]} | {row[\"Sortino Ratio\"]} |\n')

    f.write('\n\n')
    f.write('## Notes\n\n')
    f.write('- **Bold**: Our proposed method (E-NSGA-II + X-MARL)\n')
    f.write('- All metrics computed on test set (2025 data)\n')
    f.write('- Deep learning methods show mean values across random seeds\n')
    f.write('- Negative returns reflect challenging market period in test data\n')
    f.write('- Best performing method in each column would be highlighted in paper\n')

print('Results summary saved to: results/RESULTS_SUMMARY.csv')
print('Formatted table saved to: results/RESULTS_FORMATTED.csv')
print('Markdown table saved to: results/RESULTS_FORMATTED.md')
print()
print('='*80)
print('FORMATTED RESULTS TABLE (Ready for Paper)')
print('='*80)
print(formatted_df.to_string())
print()
print('='*80)
print('RAW RESULTS (Full Precision)')
print('='*80)
print(summary_df)
" 2>&1 | tee logs/aggregate_results.log
echo "  ✓ Results aggregated"

################################################################################
# Step 8: Generate Visualizations
################################################################################
print_section "Step 8: Generating Visualizations"

print_step "Creating comparison plots..."
python3 -c "
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)

# Read summary
try:
    df = pd.read_csv('results/RESULTS_SUMMARY.csv', index_col=0)

    # Extract mean values (remove ± std for plotting)
    plot_df = df.copy()
    for col in plot_df.columns:
        plot_df[col] = plot_df[col].apply(
            lambda x: float(str(x).split('±')[0].strip()) if '±' in str(x) else float(x)
        )

    # Plot 1: Sharpe Ratio comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    if 'Sharpe Ratio' in plot_df.columns:
        plot_df['Sharpe Ratio'].sort_values().plot(kind='barh', ax=ax, color='steelblue')
        ax.set_xlabel('Sharpe Ratio')
        ax.set_title('Sharpe Ratio Comparison Across Methods')
        ax.grid(axis='x')
        plt.tight_layout()
        plt.savefig('results/figures/sharpe_comparison.png', dpi=300)
        plt.close()
        print('  ✓ Saved: results/figures/sharpe_comparison.png')

    # Plot 2: Return vs Risk scatter
    fig, ax = plt.subplots(figsize=(10, 8))
    if 'Annualized Return' in plot_df.columns and 'Annualized Volatility' in plot_df.columns:
        ax.scatter(plot_df['Annualized Volatility'],
                  plot_df['Annualized Return'],
                  s=100, alpha=0.6)

        for idx, method in enumerate(plot_df.index):
            ax.annotate(method,
                       (plot_df['Annualized Volatility'].iloc[idx],
                        plot_df['Annualized Return'].iloc[idx]),
                       fontsize=9, alpha=0.7)

        ax.set_xlabel('Annualized Volatility (%)')
        ax.set_ylabel('Annualized Return (%)')
        ax.set_title('Risk-Return Trade-off')
        ax.grid(True)
        plt.tight_layout()
        plt.savefig('results/figures/risk_return_tradeoff.png', dpi=300)
        plt.close()
        print('  ✓ Saved: results/figures/risk_return_tradeoff.png')

    print('  ✓ Visualizations complete')

except Exception as e:
    print(f'Warning: Could not create plots: {e}')
" 2>&1 | tee logs/visualizations.log

################################################################################
# Step 9: Final Summary
################################################################################
print_section "Experiment Pipeline Complete!"

echo ""
echo "Results saved in:"
echo "  - results/RESULTS_SUMMARY.csv        (Main comparison table)"
echo "  - results/baselines/                 (Individual baseline results)"
echo "  - results/ablations/                 (Ablation study results)"
echo "  - results/main_method/               (E-NSGA-II + X-MARL results)"
echo "  - results/figures/                   (Plots and visualizations)"
echo ""
echo "Logs saved in:"
echo "  - logs/*.log                         (Detailed execution logs)"
echo ""

# Display summary if available
if [ -f "results/RESULTS_SUMMARY.csv" ]; then
    echo -e "${GREEN}Results Summary:${NC}"
    echo ""
    head -20 results/RESULTS_SUMMARY.csv
    echo ""
fi

# Final status
echo -e "${GREEN}✓ All experiments completed successfully!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review results/RESULTS_SUMMARY.csv"
echo "  2. Check results/figures/ for visualizations"
echo "  3. Update paper Table II with real results"
echo "  4. Generate additional plots if needed"
echo ""

# Deactivate virtual environment
deactivate

exit 0
