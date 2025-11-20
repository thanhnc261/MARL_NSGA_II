import pandas as pd
import numpy as np
import os

def load_data(path="data/raw_data.csv"):
    # Read CSV with MultiIndex header (Ticker, Price Type)
    # Note: pandas read_csv might need header=[0,1] if it was saved that way.
    # If yfinance saved it, it's likely header=[0,1] or header=[0,1,2] depending on options.
    # Let's assume standard 2-level header from yfinance group_by='ticker'.
    df = pd.read_csv(path, header=[0, 1], index_col=0, parse_dates=True)
    return df

def compute_features(df):
    """
    Computes features for each stock.
    df: MultiIndex DataFrame (Ticker, OHLCV)
    """
    # Extract unique tickers
    tickers = df.columns.get_level_values(0).unique()
    
    feature_dfs = []
    
    print("Computing features...")
    
    for ticker in tickers:
        try:
            # Extract single stock data
            stock = df[ticker].copy()
            
            # Ensure we have necessary columns
            if 'Adj Close' in stock.columns:
                close = stock['Adj Close']
            elif 'Close' in stock.columns:
                close = stock['Close']
            else:
                continue
                
            if 'Volume' not in stock.columns:
                continue
                
            # Forward fill missing data (up to a limit) to handle small gaps
            close = close.ffill(limit=5)
            volume = stock['Volume'].ffill(limit=5)
            
            # 1. Returns
            # r_t = (P_t / P_{t-1}) - 1
            daily_ret = close.pct_change()
            
            # 2. Past returns: 1, 5, 10, 20, 60 days
            # We already have 1-day return.
            # For n-day return, we can use pct_change(n)
            ret_5 = close.pct_change(5)
            ret_10 = close.pct_change(10)
            ret_20 = close.pct_change(20)
            ret_60 = close.pct_change(60)
            
            # 3. Volume ratio (vs 20-day average)
            vol_avg_20 = volume.rolling(window=20, min_periods=5).mean()
            vol_ratio = volume / (vol_avg_20 + 1e-8) # Avoid div by zero
            
            # 4. Rolling Sharpe (20-day) & Volatility (20-day)
            # Sharpe = Mean Return / Std Dev
            # We compute on daily returns window
            roll_mean = daily_ret.rolling(window=20, min_periods=5).mean()
            roll_std = daily_ret.rolling(window=20, min_periods=5).std()
            roll_sharpe = roll_mean / (roll_std + 1e-8) * np.sqrt(252) # Annualized? Guide doesn't specify, but usually yes.
            # Guide says: "Rolling Sharpe (20-day), volatility (20-day)"
            # Let's keep it simple.
            
            # 5. Distance from 52-week high/low (252 trading days)
            roll_high_252 = close.rolling(window=252, min_periods=60).max()
            roll_low_252 = close.rolling(window=252, min_periods=60).min()
            dist_high = (close - roll_high_252) / (roll_high_252 + 1e-8)
            dist_low = (close - roll_low_252) / (roll_low_252 + 1e-8)
            
            # Combine into a DataFrame
            features = pd.DataFrame({
                'ret_1': daily_ret,
                'ret_5': ret_5,
                'ret_10': ret_10,
                'ret_20': ret_20,
                'ret_60': ret_60,
                'vol_ratio': vol_ratio,
                'roll_std': roll_std,
                'roll_sharpe': roll_sharpe,
                'dist_high': dist_high,
                'dist_low': dist_low,
                'close': close # Keep close price for portfolio value calc
            })
            
            # Add ticker column for later indexing if needed, or use MultiIndex
            # Let's make it a MultiIndex (Date, Ticker) -> Features
            features['ticker'] = ticker
            features = features.reset_index().set_index(['Date', 'ticker'])
            
            feature_dfs.append(features)
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            continue
            
    # Concatenate all
    full_features = pd.concat(feature_dfs)
    
    # 6. Normalized rank of returns within the universe
    # This needs to be done cross-sectionally per date
    # We unstack to get (Date, Ticker) structure again or use groupby
    
    print("Computing cross-sectional ranks...")
    # Group by Date and compute rank for 'ret_1'
    # We want normalized rank: (rank - 1) / (count - 1) -> [0, 1]
    # Or just rank / count
    
    # It's faster to unstack, rank, stack
    ret_1_unstacked = full_features['ret_1'].unstack()
    ranks = ret_1_unstacked.rank(axis=1, pct=True) # pct=True gives percentile rank [0, 1]
    ranks = ranks.stack().to_frame('rank_ret_1')
    
    full_features = full_features.join(ranks)
    
    return full_features

def split_and_normalize(df, train_end="2023-12-31", val_end="2024-12-31"):
    """
    Splits data and normalizes based on training set statistics.
    """
    # df index is (Date, ticker)
    
    # Sort index just in case
    df = df.sort_index()
    
    # Create masks
    dates = df.index.get_level_values(0)
    
    train_mask = dates <= train_end
    val_mask = (dates > train_end) & (dates <= val_end)
    test_mask = dates > val_end
    
    train_df = df[train_mask].copy()
    val_df = df[val_mask].copy()
    test_df = df[test_mask].copy()
    
    # Fallback to ratio split if date split fails (e.g. small dataset or wrong dates)
    if len(val_df) == 0 or len(test_df) == 0:
        print("Warning: Date-based split resulted in empty validation or test set.")
        print("Falling back to ratio-based split (80% Train, 10% Val, 10% Test)...")
        
        unique_dates = dates.unique()
        n_dates = len(unique_dates)
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

    print(f"Train samples: {len(train_df)}")
    print(f"Val samples: {len(val_df)}")
    print(f"Test samples: {len(test_df)}")
    
    # Normalize features
    # We exclude 'close' and 'ticker' (if it was a column) from normalization
    # IMPORTANT: We also exclude 'ret_1' from normalization because the environment
    # needs actual returns (not z-scores) to calculate portfolio values correctly.
    # We keep ret_1 as raw returns for portfolio calculation.
    # All other features can be normalized for the neural network inputs.
    # Guide says: "Normalize all features using running mean/std computed only on training period"
    # "Running mean/std" usually implies online normalization or using the training set stats.
    # Given the context "computed only on training period", it likely means z-score using train stats.

    cols_to_norm = [c for c in train_df.columns if c not in ['close', 'ret_1']]
    
    train_mean = train_df[cols_to_norm].mean()
    train_std = train_df[cols_to_norm].std()
    
    # Avoid div by zero
    train_std = train_std.replace(0, 1)
    
    def normalize(d):
        d[cols_to_norm] = (d[cols_to_norm] - train_mean) / train_std
        return d
    
    train_df = normalize(train_df)
    val_df = normalize(val_df)
    test_df = normalize(test_df)
    
    # Fill NaNs (e.g. from rolling windows at start)
    # For training, we might drop the first 252 days or fill 0.
    # Dropping is safer for training stability.
    train_df = train_df.dropna()
    val_df = val_df.fillna(0) # Should not happen much if contiguous
    test_df = test_df.fillna(0)
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Load raw data
    raw_data_path = "data/raw_data.csv"
    if not os.path.exists(raw_data_path):
        print(f"Raw data not found at {raw_data_path}. Please run downloader.py first.")
        exit() # Exit if raw data is not available
    
    print(f"Loading raw data from {raw_data_path}...")
    df = load_data(raw_data_path)
    print(f"Raw data loaded. Shape: {df.shape}")

    print("Computing features...")
    features = compute_features(df)
    print(f"Features computed. Shape: {features.shape}")

    print("Splitting and normalizing data...")
    train_df, val_df, test_df = split_and_normalize(features)
    print("Data split and normalized.")

    # Save processed data
    print(f"Saving processed data to {os.getcwd()}/data/...")
    train_df.to_csv("data/train.csv")
    val_df.to_csv("data/val.csv")
    test_df.to_csv("data/test.csv")
    
    print(f"Train data saved to {os.path.abspath('data/train.csv')} ({len(train_df)} rows)")
    print(f"Val data saved to {os.path.abspath('data/val.csv')} ({len(val_df)} rows)")
    print(f"Test data saved to {os.path.abspath('data/test.csv')} ({len(test_df)} rows)")
    print("Preprocessing complete.")
