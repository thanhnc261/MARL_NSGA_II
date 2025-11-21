"""
Feature Engineering and Data Preprocessing for E-NSGA-II + X-MARL

This module computes technical indicators and prepares data for RL training.

Features (22 total per stock):
  - Price Returns: ret_1, ret_5, ret_10, ret_20, ret_60
  - Volatility: roll_std_20, roll_std_60
  - Technical: RSI_14, MACD, MACD_signal, MACD_hist
  - Bollinger: bb_upper, bb_lower, bb_width
  - Volume: vol_ratio, volume_vol
  - Position: dist_high_252, dist_low_252
  - Rolling Metrics: roll_sharpe_20, roll_sortino_20
  - Cross-Sectional: rank_ret_1, rank_vol_20

Data Split Strategy (Walk-Forward):
  Train: 2010-01-01 → 2021-12-31 (12 years, ~3,000 days)
  Val:   2022-01-01 → 2023-12-31 (2 years, ~500 days)
  Test:  2024-01-01 → 2025-11-30 (2 years, ~500 days)

Author: E-NSGA-II + X-MARL Research Team
"""

import pandas as pd
import numpy as np
import os
from typing import Tuple, List


def load_data(path: str = "data/raw_data.csv") -> pd.DataFrame:
    """
    Load raw OHLCV data from CSV file.

    Args:
        path: Path to the raw data CSV file

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price Type)
    """
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Compute Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over period

    Args:
        prices: Series of prices (typically Close)
        period: Lookback period (default 14)

    Returns:
        RSI values in range [0, 100]
    """
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    rs = avg_gain / (avg_loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))

    return rsi


def compute_macd(
    prices: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute MACD (Moving Average Convergence Divergence).

    MACD Line = EMA(fast) - EMA(slow)
    Signal Line = EMA(MACD Line, signal)
    Histogram = MACD Line - Signal Line

    Args:
        prices: Series of prices
        fast: Fast EMA period (default 12)
        slow: Slow EMA period (default 26)
        signal: Signal line period (default 9)

    Returns:
        Tuple of (MACD line, Signal line, Histogram)
    """
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def compute_bollinger_bands(
    prices: pd.Series,
    period: int = 20,
    std_dev: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute Bollinger Bands.

    Middle Band = SMA(period)
    Upper Band = Middle + std_dev * STD(period)
    Lower Band = Middle - std_dev * STD(period)

    Args:
        prices: Series of prices
        period: Lookback period (default 20)
        std_dev: Number of standard deviations (default 2)

    Returns:
        Tuple of (Upper band, Lower band, Band width)
    """
    sma = prices.rolling(window=period, min_periods=period).mean()
    std = prices.rolling(window=period, min_periods=period).std()

    upper = sma + std_dev * std
    lower = sma - std_dev * std

    # Band width as percentage of middle
    width = (upper - lower) / (sma + 1e-10)

    return upper, lower, width


def compute_sortino_rolling(
    returns: pd.Series,
    period: int = 20,
    target: float = 0.0
) -> pd.Series:
    """
    Compute rolling Sortino ratio.

    Sortino = (Mean Return - Target) / Downside Deviation

    Args:
        returns: Series of returns
        period: Lookback period
        target: Target return (default 0)

    Returns:
        Rolling Sortino ratio (annualized)
    """
    def sortino_func(r):
        excess = r - target
        downside = r[r < target]
        if len(downside) < 2:
            return 0.0
        downside_std = downside.std()
        if downside_std < 1e-10:
            return 0.0
        return (excess.mean() / downside_std) * np.sqrt(252)

    return returns.rolling(window=period, min_periods=period).apply(sortino_func, raw=False)


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comprehensive features for each stock.

    Features computed (22 per stock):
      1. ret_1: 1-day return
      2. ret_5: 5-day return
      3. ret_10: 10-day return
      4. ret_20: 20-day (monthly) return
      5. ret_60: 60-day (quarterly) return
      6. roll_std_20: 20-day rolling volatility
      7. roll_std_60: 60-day rolling volatility
      8. RSI_14: 14-day RSI
      9. MACD: MACD line
      10. MACD_signal: MACD signal line
      11. MACD_hist: MACD histogram
      12. bb_upper: Bollinger upper band (normalized)
      13. bb_lower: Bollinger lower band (normalized)
      14. bb_width: Bollinger band width
      15. vol_ratio: Volume vs 20-day average
      16. volume_vol: Volume volatility (20-day)
      17. dist_high_252: Distance from 52-week high
      18. dist_low_252: Distance from 52-week low
      19. roll_sharpe_20: 20-day rolling Sharpe
      20. roll_sortino_20: 20-day rolling Sortino
      21. rank_ret_1: Cross-sectional return rank
      22. rank_vol_20: Cross-sectional volatility rank

    Args:
        df: MultiIndex DataFrame (Ticker, OHLCV)

    Returns:
        DataFrame with MultiIndex (Date, ticker) -> Features
    """
    tickers = df.columns.get_level_values(0).unique()
    feature_dfs = []

    print(f"Computing features for {len(tickers)} stocks...")

    for i, ticker in enumerate(tickers):
        try:
            stock = df[ticker].copy()

            # Get price and volume
            if 'Adj Close' in stock.columns:
                close = stock['Adj Close']
            elif 'Close' in stock.columns:
                close = stock['Close']
            else:
                print(f"  Skipping {ticker}: No close price")
                continue

            if 'Volume' not in stock.columns:
                print(f"  Skipping {ticker}: No volume data")
                continue

            # Forward fill small gaps
            close = close.ffill(limit=5)
            volume = stock['Volume'].ffill(limit=5)

            # Get High/Low for technical indicators
            high = stock['High'].ffill(limit=5) if 'High' in stock.columns else close
            low = stock['Low'].ffill(limit=5) if 'Low' in stock.columns else close

            # ===== Price Returns =====
            daily_ret = close.pct_change()
            ret_5 = close.pct_change(5)
            ret_10 = close.pct_change(10)
            ret_20 = close.pct_change(20)
            ret_60 = close.pct_change(60)

            # ===== Volatility =====
            roll_std_20 = daily_ret.rolling(window=20, min_periods=10).std()
            roll_std_60 = daily_ret.rolling(window=60, min_periods=30).std()

            # ===== Technical Indicators =====
            rsi_14 = compute_rsi(close, period=14)
            macd, macd_signal, macd_hist = compute_macd(close)
            bb_upper, bb_lower, bb_width = compute_bollinger_bands(close)

            # Normalize Bollinger bands relative to price
            bb_upper_norm = (bb_upper - close) / (close + 1e-10)
            bb_lower_norm = (close - bb_lower) / (close + 1e-10)

            # ===== Volume Features =====
            vol_avg_20 = volume.rolling(window=20, min_periods=10).mean()
            vol_ratio = volume / (vol_avg_20 + 1e-10)
            volume_vol = volume.rolling(window=20, min_periods=10).std() / (vol_avg_20 + 1e-10)

            # ===== Position Features =====
            roll_high_252 = close.rolling(window=252, min_periods=60).max()
            roll_low_252 = close.rolling(window=252, min_periods=60).min()
            dist_high = (close - roll_high_252) / (roll_high_252 + 1e-10)
            dist_low = (close - roll_low_252) / (roll_low_252 + 1e-10)

            # ===== Rolling Risk-Adjusted Metrics =====
            roll_mean = daily_ret.rolling(window=20, min_periods=10).mean()
            roll_sharpe = (roll_mean / (roll_std_20 + 1e-10)) * np.sqrt(252)
            roll_sortino = compute_sortino_rolling(daily_ret, period=20)

            # ===== Combine Features =====
            features = pd.DataFrame({
                # Returns (5)
                'ret_1': daily_ret,
                'ret_5': ret_5,
                'ret_10': ret_10,
                'ret_20': ret_20,
                'ret_60': ret_60,

                # Volatility (2)
                'roll_std_20': roll_std_20,
                'roll_std_60': roll_std_60,

                # Technical (4)
                'RSI_14': rsi_14 / 100.0,  # Normalize to [0, 1]
                'MACD': macd / (close + 1e-10),  # Normalize
                'MACD_signal': macd_signal / (close + 1e-10),
                'MACD_hist': macd_hist / (close + 1e-10),

                # Bollinger (3)
                'bb_upper': bb_upper_norm,
                'bb_lower': bb_lower_norm,
                'bb_width': bb_width,

                # Volume (2)
                'vol_ratio': vol_ratio,
                'volume_vol': volume_vol,

                # Position (2)
                'dist_high': dist_high,
                'dist_low': dist_low,

                # Rolling Metrics (2)
                'roll_sharpe': roll_sharpe,
                'roll_sortino': roll_sortino,

                # Keep close for portfolio calculations
                'close': close
            })

            features['ticker'] = ticker
            features = features.reset_index().set_index(['Date', 'ticker'])
            feature_dfs.append(features)

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(tickers)} stocks")

        except Exception as e:
            print(f"  Error processing {ticker}: {e}")
            continue

    # Concatenate all stocks
    full_features = pd.concat(feature_dfs)

    # ===== Cross-Sectional Features =====
    print("Computing cross-sectional ranks...")

    # Rank returns within each date
    ret_1_unstacked = full_features['ret_1'].unstack()
    rank_ret = ret_1_unstacked.rank(axis=1, pct=True)
    rank_ret = rank_ret.stack().to_frame('rank_ret_1')

    # Rank volatility within each date
    vol_unstacked = full_features['roll_std_20'].unstack()
    rank_vol = vol_unstacked.rank(axis=1, pct=True)
    rank_vol = rank_vol.stack().to_frame('rank_vol_20')

    # Join cross-sectional features
    full_features = full_features.join(rank_ret).join(rank_vol)

    print(f"Total features per stock: {len(full_features.columns)}")

    return full_features


def split_and_normalize(
    df: pd.DataFrame,
    train_end: str = "2021-12-31",
    val_end: str = "2023-12-31"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data using walk-forward methodology and normalizes features.

    Walk-Forward Split Strategy:
      - Train: Historical data (2010-2021) - 12 years
      - Val: Out-of-sample validation (2022-2023) - 2 years
      - Test: Final evaluation (2024-2025) - ~2 years

    This ensures:
      - No look-ahead bias
      - Multiple market regimes in training
      - Recent data for final testing

    Normalization:
      - Z-score normalization using ONLY training statistics
      - ret_1 and close are NOT normalized (needed for portfolio calculations)

    Args:
        df: Feature DataFrame with MultiIndex (Date, ticker)
        train_end: Last date of training period
        val_end: Last date of validation period

    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    df = df.sort_index()
    dates = df.index.get_level_values(0)

    # Create masks for each split
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end

    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()

    # Validate splits
    if len(val_df) == 0 or len(test_df) == 0:
        print("WARNING: Date-based split failed. Using ratio-based fallback.")

        unique_dates = dates.unique()
        n_dates = len(unique_dates)

        # 80% train, 10% val, 10% test
        train_idx = int(n_dates * 0.8)
        val_idx = int(n_dates * 0.9)

        train_date_end = unique_dates[train_idx]
        val_date_end = unique_dates[val_idx]

        train_mask = dates <= train_date_end
        val_mask = (dates > train_date_end) & (dates <= val_date_end)
        test_mask = dates > val_date_end

        train_df = df[train_mask].copy()
        val_df = df[val_mask].copy()
        test_df = df[test_mask].copy()

    # Print split information
    print(f"\n" + "=" * 60)
    print("Data Split Summary")
    print("=" * 60)
    print(f"Train: {train_df.index.get_level_values(0).min()} to {train_df.index.get_level_values(0).max()}")
    print(f"       {len(train_df)} samples ({len(train_df.index.get_level_values(0).unique())} days)")
    print(f"Val:   {val_df.index.get_level_values(0).min()} to {val_df.index.get_level_values(0).max()}")
    print(f"       {len(val_df)} samples ({len(val_df.index.get_level_values(0).unique())} days)")
    print(f"Test:  {test_df.index.get_level_values(0).min()} to {test_df.index.get_level_values(0).max()}")
    print(f"       {len(test_df)} samples ({len(test_df.index.get_level_values(0).unique())} days)")
    print("=" * 60)

    # ===== Normalization =====
    # Features to normalize (exclude ret_1, close, and ticker if present)
    cols_to_exclude = ['close', 'ret_1']
    cols_to_norm = [c for c in train_df.columns if c not in cols_to_exclude]

    # Compute training statistics
    train_mean = train_df[cols_to_norm].mean()
    train_std = train_df[cols_to_norm].std()
    train_std = train_std.replace(0, 1)  # Avoid division by zero

    def normalize(data: pd.DataFrame) -> pd.DataFrame:
        """Apply z-score normalization using training statistics."""
        data = data.copy()
        data[cols_to_norm] = (data[cols_to_norm] - train_mean) / train_std
        return data

    train_df = normalize(train_df)
    val_df = normalize(val_df)
    test_df = normalize(test_df)

    # Handle NaN values
    # Drop NaN rows in training (from rolling windows at start)
    # Fill NaN in val/test with 0 (should be minimal if data is contiguous)
    initial_train_len = len(train_df)
    train_df = train_df.dropna()
    dropped = initial_train_len - len(train_df)
    if dropped > 0:
        print(f"Dropped {dropped} training samples with NaN (from rolling window warmup)")

    val_df = val_df.fillna(0)
    test_df = test_df.fillna(0)

    # ===== Ensure Consistent Stock Universe =====
    # Only keep stocks that appear in ALL three splits
    train_tickers = set(train_df.index.get_level_values(1).unique())
    val_tickers = set(val_df.index.get_level_values(1).unique())
    test_tickers = set(test_df.index.get_level_values(1).unique())

    common_tickers = train_tickers & val_tickers & test_tickers

    if len(common_tickers) < len(train_tickers):
        removed = len(train_tickers) - len(common_tickers)
        print(f"Filtering to {len(common_tickers)} stocks common across all splits (removed {removed})")

        train_df = train_df[train_df.index.get_level_values(1).isin(common_tickers)]
        val_df = val_df[val_df.index.get_level_values(1).isin(common_tickers)]
        test_df = test_df[test_df.index.get_level_values(1).isin(common_tickers)]

    return train_df, val_df, test_df


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "data"
):
    """
    Save processed data splits to CSV files.

    Args:
        train_df: Training data
        val_df: Validation data
        test_df: Test data
        output_dir: Output directory
    """
    os.makedirs(output_dir, exist_ok=True)

    train_path = os.path.join(output_dir, "train.csv")
    val_path = os.path.join(output_dir, "val.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path)
    val_df.to_csv(val_path)
    test_df.to_csv(test_path)

    print(f"\nSaved processed data:")
    print(f"  Train: {train_path} ({len(train_df)} rows)")
    print(f"  Val:   {val_path} ({len(val_df)} rows)")
    print(f"  Test:  {test_path} ({len(test_df)} rows)")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("E-NSGA-II + X-MARL Feature Engineering")
    print("=" * 60)

    # Check for raw data
    raw_data_path = "data/raw_data.csv"
    if not os.path.exists(raw_data_path):
        print(f"ERROR: Raw data not found at {raw_data_path}")
        print("Please run 'python data/downloader.py' first.")
        exit(1)

    # Load raw data
    print(f"\nLoading raw data from {raw_data_path}...")
    df = load_data(raw_data_path)
    print(f"Raw data shape: {df.shape}")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")

    # Compute features
    print("\nComputing features...")
    features = compute_features(df)
    print(f"Feature matrix shape: {features.shape}")

    # Split and normalize
    print("\nSplitting and normalizing data...")
    train_df, val_df, test_df = split_and_normalize(
        features,
        train_end="2021-12-31",
        val_end="2023-12-31"
    )

    # Save processed data
    save_splits(train_df, val_df, test_df)

    print("\n" + "=" * 60)
    print("Feature engineering complete!")
    print("Next step: Run experiments with 'python main.py'")
    print("=" * 60)
