import yfinance as yf
import pandas as pd
import os
import requests
from io import StringIO
from datetime import datetime, timedelta
import requests
from io import StringIO

def get_sp500_tickers():
    """
    Scrapes the list of S&P 500 tickers from Wikipedia.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        # Use requests with headers to avoid 403 Forbidden
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Read HTML from response content
        table = pd.read_html(StringIO(response.text))
        df = table[0]
        tickers = df['Symbol'].tolist()
        # Replace dots with dashes for yfinance (e.g. BRK.B -> BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        # Fallback to 30 diversified stocks across all sectors
        print("Using fallback ticker list (30 stocks across sectors).")
        return [
            # Technology (8 stocks)
            'AAPL', 'MSFT', 'NVDA', 'GOOGL', 'META', 'TSLA', 'ADBE', 'CRM',
            # Consumer Discretionary (4 stocks)
            'AMZN', 'HD', 'NKE', 'MCD',
            # Financials (5 stocks)
            'JPM', 'BAC', 'V', 'MA', 'GS',
            # Healthcare (5 stocks)
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO',
            # Consumer Staples (2 stocks)
            'PG', 'KO',
            # Energy (2 stocks)
            'XOM', 'CVX',
            # Industrials (2 stocks)
            'BA', 'CAT',
            # Communication Services (2 stocks)
            'DIS', 'NFLX'
        ]

def download_data(start_date="2020-01-01", end_date="2025-10-31", save_path="data/raw_data.csv", threads=True):
    """
    Downloads daily data for S&P 500 stocks.
    """
    print(f"Fetching S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} tickers.")

    print(f"Downloading data from {start_date} to {end_date}...")
    
    # Download in chunks or all at once? yfinance handles bulk download well.
    # auto_adjust=True gives us Adj Close as Close, which is usually what we want for returns.
    # But the guide asks for "Open, High, Low, Close, Adj Close, Volume".
    # So we set auto_adjust=False.
    
    data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date, 
        group_by='ticker', 
        auto_adjust=False,
        actions=False,
        threads=threads
    )
    
    # The data will be a MultiIndex DataFrame.
    # Level 0: Ticker, Level 1: OHLCV
    # Or Level 0: OHLCV, Level 1: Ticker (depending on yfinance version, usually Ticker if group_by='ticker')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
        print(f"Data saved to {save_path}")
        
    return data

if __name__ == "__main__":
    # Download S&P 500 data for the study period
    # Data split:
    #   Train: 2020-01-01 → 2023-12-31 (~1,007 trading days)
    #   Val:   2024-01-01 → 2024-12-31 (~252 trading days)
    #   Test:  2025-01-01 → 2025-11-30 (~230 trading days)
    # Download from start of 2020 to end of November 2025 (or latest available)
    download_data(start_date="2020-01-01", end_date="2025-11-30")
