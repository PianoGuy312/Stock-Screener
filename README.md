# Stock Analysis Toolkit

A collection of Python tools for stock screening, quantitative analysis, backtesting, and company research.

The project includes three main programs:

- `stock_screener.py` — Stock ratings, technical indicators, sentiment analysis, and backtesting.
- `stock_model.py` — Fundamental DCF valuation, technical analysis, and oversold stock detection.
- `stock_report.py` — Detailed company research reports using Yahoo Finance and SEC EDGAR.

---

# 1. Download the Repository

Download or clone the repository and make sure you have all the project files.

Your project should look something like this:

```text
stock_screener/
├── stock_screener.py
├── stock_model.py
├── stock_report.py
├── requirements.txt
├── README.md
├── watchlist.csv
├── indexes/
│   ├── sp400.csv
│   ├── sp600.csv
│   ├── nasdaq100.csv
│   ├── dow30.csv
│   └── russell2000.csv
└── reports/
```

The `indexes` and `reports` folders can be created as needed.

# 2. Set Up a Virtual Environment

It is recommended to run the programs inside a Python virtual environment.

Create the environment:

```bash
python3 -m venv ~/venv
```

Activate it:

```bash
source ~/venv/bin/activate
```

Whenever you open a new terminal, activate the environment again:

```bash
source ~/venv/bin/activate
```

# 3. Install Requirements

Install the required Python libraries:

```bash
pip install -r requirements.txt
```

If you need to install the libraries individually:

```bash
pip install yfinance pandas numpy nltk lxml requests scikit-learn
```

# 4. Stock Screener (`stock_screener.py`)

The stock screener analyzes stocks and assigns ratings based on technical indicators, relative strength, sentiment, and other factors.

It also includes historical backtesting capabilities.

### Run the default watchlist

```bash
python stock_screener.py
```

### Analyze specific stocks

```bash
python stock_screener.py --tickers TSLA AMD SPY
```

Replace these with your own ticker symbols as recognized by Yahoo Finance.

### Run the S&P 500

```bash
python stock_screener.py --sp500
```

You can also use:

```bash
python stock_screener.py --universe sp500
```

### Backtest the S&P 500

```bash
python stock_screener.py --backtest --universe sp500
```

### Backtest a custom CSV

```bash
python stock_screener.py --file indexes/nasdaq100.csv --backtest
```

### Generate an HTML report

```bash
python stock_screener.py --universe sp500 --html
```

# 5. Custom Indexes Folder

Instead of relying on online index scraping, you can maintain your own index lists using CSV files.

The S&P 500 is available through the built-in option. Other indexes can be maintained as your own CSV files.

### Create the indexes folder

From your project directory, run:

```bash
mkdir -p indexes
```

### Create a CSV file

For example, to create a Nasdaq-100 list:

```bash
nano indexes/nasdaq100.csv
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

The example above is only a sample, not the complete index.

Put one ticker per line, with `Symbol` as the header.

Save the file using:

- CTRL + O
- ENTER
- CTRL + X

You can create as many index CSV files as you want.

### Example index files

```text
indexes/
├── sp400.csv
├── sp600.csv
├── nasdaq100.csv
├── dow30.csv
└── russell2000.csv
```

### Run your custom indexes

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

**Dow Jones 30:**

```bash
python stock_screener.py --file indexes/dow30.csv
```

**Russell 2000:**

```bash
python stock_screener.py --file indexes/russell2000.csv
```

You can backtest any of these lists by adding `--backtest`:

```bash
python stock_screener.py --file indexes/nasdaq100.csv --backtest
```

# 6. Stock Model (`stock_model.py`)

The stock model performs fundamental DCF valuation, technical analysis, and identifies potentially oversold stocks showing bullish reversal patterns.

### Analyze one stock

```bash
python stock_model.py AAPL
```

Replace `AAPL` with any ticker symbol.

### Analyze an entire index using a CSV

```bash
python stock_model.py --csv indexes/nasdaq100.csv --out-dir reports
```

### Show the full dashboard for every stock

```bash
python stock_model.py --csv indexes/nasdaq100.csv --out-dir reports --detail
```

### Change the sorting method

The model supports four sorting options:

```bash
python stock_model.py --csv indexes/nasdaq100.csv --sort upside
```

Other options:

- `--sort rsi`
- `--sort signals`
- `--sort ticker`

You can replace `nasdaq100.csv` with any of your other index CSV files.

# 7. Company Research Reports (`stock_report.py`)

The research report generator creates detailed company reports using Yahoo Finance and SEC EDGAR data.

Reports can include:

- Company overview and business description
- SEC filings and financial statements
- Revenue, earnings, and financial metrics
- Technical indicators
- Five-year discounted cash flow (DCF) valuation
- Risk factors from company filings
- Valuation scenarios and sensitivity analysis

### Generate a report for one company

```bash
python stock_report.py AAPL
```

### Generate reports for multiple companies

```bash
python stock_report.py AAPL MSFT NVDA
```

### Generate reports using an index CSV

```bash
python stock_report.py --csv indexes/nasdaq100.csv --out-dir reports
```

### Skip downloading 10-K excerpts

```bash
python stock_report.py AAPL --no-excerpt
```

### SEC User-Agent Setup

The SEC requires a descriptive User-Agent containing your name and email address.

Set it in your terminal:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

Replace the example with your own information.

You can also pass it directly using `--user-agent`.

Reports are saved as Markdown files inside the `reports` folder.

# 8. Personal Watchlist

You can edit `watchlist.csv` to create your own personal stock watchlist.

This is useful for tracking companies you are interested in without having to type their ticker symbols every time.

# 9. Output Files

The programs generate different types of output:

**Stock Screener:**
- CSV file containing stock ratings and analysis.
- HTML report when using `--html`.

**Stock Model:**
- Batch summary and historical signal CSV files when running in batch mode.

**Company Research Reports:**
- Individual Markdown research reports.

You can organize your generated files inside the `reports` folder.

# 10. Important Notes

- Make sure your virtual environment is activated before running any program.
- All ticker symbols must be recognized by Yahoo Finance.
- Keep your index CSV files updated when index membership changes.
- The SEC may limit requests if you do not use a descriptive User-Agent.
- Historical backtesting does not guarantee future performance.
- These tools are for educational and research purposes only. They are not financial advice.

---

**Thanks for using the Stock Analysis Toolkit!**