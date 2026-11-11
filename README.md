## 9. Stock Model (`stock_model.py`)

The Stock Model performs fundamental valuation, technical analysis, and oversold/hammer pattern screening.

**Analyze one stock:**

```bash
python stock_model.py AAPL
```

Replace AAPL with any ticker symbol.

**Analyze an entire CSV index:**

```bash
python stock_model.py --csv indexes/nasdaq100.csv --out-dir model_results
```

**Show detailed dashboards for every stock:**

```bash
python stock_model.py --csv indexes/nasdaq100.csv --detail
```

**Sort results:**

```bash
python stock_model.py --csv indexes/nasdaq100.csv --sort upside
```

Other sorting options include `rsi`, `signals`, and `ticker`.

The model can also be run with a CSV filename directly:

```bash
python stock_model.py indexes/sp400.csv
```

## 10. Company Research Reports (`stock_report.py`)

The Company Research Report Generator creates detailed research reports using Yahoo Finance and SEC filings.

Reports include:
- Company business overview
- SEC filings (10-K, 10-Q, and 8-K)
- Financial statements and historical financial data
- Five-year discounted cash flow (DCF) valuation
- WACC and valuation sensitivity analysis
- Risk factors and business information from SEC filings

**Generate a report for one company:**

```bash
python stock_report.py AAPL
```

**Generate reports for multiple companies:**

```bash
python stock_report.py AAPL MSFT NVDA
```

**Generate reports for an entire CSV index:**

```bash
python stock_report.py --csv indexes/nasdaq100.csv --out-dir reports
```

**Skip downloading 10-K excerpts:**

```bash
python stock_report.py AAPL --no-excerpt
```

Reports are saved as Markdown files in the `reports` folder by default.

### Important SEC Setup

The SEC requires a descriptive User-Agent containing your name and email address.

Set yours using:

```bash
export SEC_USER_AGENT="Your Name your.email@example.com"
```

Replace the example with your own name and email. You only need to set this again in a new terminal session.

## 11. Additional Dependencies

The Stock Model requires scikit-learn. Install it with:

```bash
pip install scikit-learn
```

The Stock Screener's colorful dashboard uses Rich, which is optional:

```bash
pip install rich
```

All three programs should be run inside your activated virtual environment.

```bash
source ~/venv/bin/activate
```