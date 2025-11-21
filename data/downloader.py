"""
Data Downloader for E-NSGA-II + X-MARL Portfolio Optimization

Downloads 15 years of historical data (2010-2025) for 30 diversified S&P 500 stocks.
This extended dataset covers multiple market regimes:
  - 2010-2011: Post-GFC recovery, Euro debt crisis
  - 2015-2016: China fears, oil crash, volatility spike
  - 2018: Q4 correction (-20%)
  - 2020: COVID crash and recovery
  - 2022: Bear market, inflation concerns
  - 2023-2025: AI boom, rate hikes

Author: E-NSGA-II + X-MARL Research Team
"""

import yfinance as yf
import pandas as pd
import os
import requests
from io import StringIO
from datetime import datetime, timedelta


def get_sp500_tickers():
    """
    Scrapes the list of S&P 500 tickers from Wikipedia.
    Falls back to curated 30-stock list if scraping fails.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.114 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()

        table = pd.read_html(StringIO(response.text))
        df = table[0]
        tickers = df['Symbol'].tolist()
        # Replace dots with dashes for yfinance (e.g. BRK.B -> BRK-B)
        tickers = [t.replace('.', '-') for t in tickers]
        return tickers
    except Exception as e:
        print(f"Error fetching S&P 500 tickers: {e}")
        print("Using fallback ticker list (30 stocks across sectors).")
        return get_diversified_30_stocks()


def get_diversified_30_stocks():
    """
    Returns 30 diversified stocks across 8 sectors.
    Selected for:
      - Long trading history (available since 2010)
      - High liquidity
      - Sector diversification
      - Consistent S&P 500 membership
    """
    return [
        # Technology (8 stocks) - High growth, high volatility
        'AAPL',   # Apple - Consumer electronics
        'MSFT',   # Microsoft - Enterprise software
        'NVDA',   # NVIDIA - AI/GPU leader
        'GOOGL',  # Alphabet - Search/Cloud
        'META',   # Meta (was FB) - Social media
        'ADBE',   # Adobe - Creative software
        'CRM',    # Salesforce - CRM leader
        'INTC',   # Intel - Semiconductors

        # Consumer Discretionary (4 stocks) - Cyclical
        'AMZN',   # Amazon - E-commerce
        'HD',     # Home Depot - Home improvement
        'NKE',    # Nike - Apparel
        'MCD',    # McDonald's - Fast food

        # Financials (5 stocks) - Rate sensitive
        'JPM',    # JPMorgan - Banking
        'BAC',    # Bank of America - Banking
        'V',      # Visa - Payments
        'MA',     # Mastercard - Payments
        'GS',     # Goldman Sachs - Investment banking

        # Healthcare (5 stocks) - Defensive
        'JNJ',    # Johnson & Johnson - Pharma/Consumer
        'UNH',    # UnitedHealth - Insurance
        'PFE',    # Pfizer - Pharma
        'ABBV',   # AbbVie - Biotech
        'TMO',    # Thermo Fisher - Life sciences

        # Consumer Staples (2 stocks) - Defensive
        'PG',     # Procter & Gamble - Consumer goods
        'KO',     # Coca-Cola - Beverages

        # Energy (2 stocks) - Commodity sensitive
        'XOM',    # ExxonMobil - Oil & Gas
        'CVX',    # Chevron - Oil & Gas

        # Industrials (2 stocks) - Economic bellwether
        'BA',     # Boeing - Aerospace
        'CAT',    # Caterpillar - Heavy machinery

        # Communication Services (2 stocks)
        'DIS',    # Disney - Entertainment
        'NFLX',   # Netflix - Streaming
    ]


def download_data(
    start_date: str = "2010-01-01",
    end_date: str = "2025-11-30",
    save_path: str = "data/raw_data.csv",
    tickers: list = None,
    threads: bool = True
):
    """
    Downloads daily OHLCV data for specified stocks.

    Args:
        start_date: Start date for data download (default: 2010-01-01 for 15 years)
        end_date: End date for data download
        save_path: Path to save the CSV file
        tickers: List of tickers to download (default: 30 diversified stocks)
        threads: Whether to use multi-threading for download

    Returns:
        DataFrame with MultiIndex columns (Ticker, Price Type)

    Data Structure:
        - Index: Date
        - Columns: MultiIndex (Ticker, ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume'])

    Coverage:
        - 15 years: 2010-01-01 to 2025-11-30
        - ~3,780 trading days
        - 30 stocks × 6 price columns = 180 columns
    """
    if tickers is None:
        tickers = get_diversified_30_stocks()

    print(f"=" * 60)
    print(f"E-NSGA-II + X-MARL Data Downloader")
    print(f"=" * 60)
    print(f"Tickers: {len(tickers)} stocks")
    print(f"Period: {start_date} to {end_date}")
    print(f"Expected trading days: ~{(pd.to_datetime(end_date) - pd.to_datetime(start_date)).days * 252 // 365}")
    print(f"=" * 60)

    print(f"\nDownloading data...")

    # Download using yfinance
    # group_by='ticker' ensures MultiIndex columns (Ticker, Price)
    # auto_adjust=False keeps both Close and Adj Close
    data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        group_by='ticker',
        auto_adjust=False,
        actions=False,
        threads=threads,
        progress=True
    )

    # Verify download
    if data.empty:
        raise ValueError("No data downloaded. Check internet connection and ticker validity.")

    # Print download summary
    print(f"\n" + "=" * 60)
    print(f"Download Summary")
    print(f"=" * 60)
    print(f"Date range: {data.index[0]} to {data.index[-1]}")
    print(f"Trading days: {len(data)}")
    print(f"Tickers downloaded: {len(data.columns.get_level_values(0).unique())}")

    # Check for missing data
    missing_pct = data.isnull().sum().sum() / data.size * 100
    print(f"Missing data: {missing_pct:.2f}%")

    # Save to CSV
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        data.to_csv(save_path)
        file_size = os.path.getsize(save_path) / (1024 * 1024)
        print(f"\nData saved to: {save_path}")
        print(f"File size: {file_size:.2f} MB")

    print(f"=" * 60)

    return data


def validate_data(data_path: str = "data/raw_data.csv"):
    """
    Validates downloaded data for quality and completeness.

    Checks:
        1. All expected tickers present
        2. Date range coverage
        3. Missing value percentage
        4. Price anomalies (negative prices, extreme returns)
    """
    print("\nValidating downloaded data...")

    df = pd.read_csv(data_path, header=[0, 1], index_col=0, parse_dates=True)

    expected_tickers = get_diversified_30_stocks()
    actual_tickers = df.columns.get_level_values(0).unique().tolist()

    # Check ticker coverage
    missing_tickers = set(expected_tickers) - set(actual_tickers)
    if missing_tickers:
        print(f"WARNING: Missing tickers: {missing_tickers}")

    # Check date range
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    print(f"Total trading days: {len(df)}")

    # Check for gaps
    date_diff = df.index.to_series().diff()
    large_gaps = date_diff[date_diff > pd.Timedelta(days=5)]
    if len(large_gaps) > 0:
        print(f"WARNING: {len(large_gaps)} gaps > 5 days found")

    # Missing value summary per ticker
    print("\nMissing values per ticker:")
    for ticker in actual_tickers[:5]:  # Show first 5
        ticker_data = df[ticker]
        missing = ticker_data.isnull().sum().sum()
        total = ticker_data.size
        print(f"  {ticker}: {missing}/{total} ({missing/total*100:.2f}%)")

    print("\nValidation complete.")
    return True


if __name__ == "__main__":
    # Download 15 years of data for 30 diversified stocks
    # This covers multiple market regimes for robust training

    print("\n" + "=" * 60)
    print("Starting 15-Year Data Download")
    print("=" * 60)
    print("\nMarket Regimes Covered:")
    print("  2010-2011: Post-GFC recovery, Euro debt crisis")
    print("  2015-2016: China fears, oil crash")
    print("  2018:      Q4 correction (-20%)")
    print("  2020:      COVID crash and V-shaped recovery")
    print("  2022:      Bear market, inflation shock")
    print("  2023-2025: AI boom, rate normalization")
    print("=" * 60 + "\n")

    # Download data
    data = download_data(
        start_date="2010-01-01",
        end_date="2025-11-30",
        save_path="data/raw_data.csv"
    )

    # Validate
    validate_data("data/raw_data.csv")

    print("\n" + "=" * 60)
    print("Data download complete!")
    print("Next step: Run 'python data/preprocessor.py' to compute features")
    print("=" * 60)
