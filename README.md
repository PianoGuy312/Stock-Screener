# Stock Screener

A Python stock screener that analyzes stocks, assigns ratings, and provides backtesting capabilities.

## 1. Download the Repository

Make sure you have downloaded all files from the repository.

## 2. Set Up a Virtual Environment

Create your virtual environment:

```bash
python3 -m venv ~/venv
source ~/venv/bin/activate
```

Whenever you open a new terminal, activate it again:

```bash
source ~/venv/bin/activate
```

## 3. Install Requirements

```bash
pip install -r requirements.txt
```

## 4. Running the Screener

**Run the default watchlist:**

```bash
python stock_screener.py
```

**Analyze specific stocks:**

```bash
python stock_screener.py --tickers TSLA AMD SPY
```

Replace these with your own Yahoo Finance ticker symbols.

**Run an S&P 500 scan:**

```bash
python stock_screener.py --sp500
```

**Backtest the default stocks:**

```bash
python stock_screener.py --backtest
```

**Backtest the S&P 500:**

```bash
python stock_screener.py --backtest --universe sp500
```

## 5. Custom Indexes Folder

You can create your own index lists using CSV files instead of relying on online index scraping.

Your project structure should look like this:

```text
stock_screener/
├── stock_screener.py
├── requirements.txt
├── README.md
├── watchlist.csv
└── indexes/
    ├── sp400.csv
    ├── sp600.csv
    ├── nasdaq100.csv
    ├── dow30.csv
    └── russell2000.csv
```

### Creating the indexes folder

```bash
mkdir -p ~/indexes
```

### Creating a CSV file

For example, create a Nasdaq-100 CSV:

```bash
nano ~/indexes/nasdaq100.csv
```

Use this format:

```csv
Symbol
AAPL
MSFT
NVDA
AMZN
GOOGL
META
AVGO
COST
NFLX
TSLA
```

Put one ticker per line. The first line must be `Symbol`.

Save your file with:
- CTRL + O
- ENTER
- CTRL + X

## 6. Running Your Custom Indexes

Use the `--file` argument to run a CSV file.

**S&P 400:**

```bash
python stock_screener.py --file indexes/sp400.csv
```

**S&P 600:**

```bash
python stock_screener.py --file indexes/sp600.csv
```

**Nasdaq-100:**

```bash
python stock_screener.py --file indexes/nasdaq100.csv
```

