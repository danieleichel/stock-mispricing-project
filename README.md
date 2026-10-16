# When Do Stock Mispricings Correct? A Statistical Arbitrage Study

**Status: in progress**

Some stocks usually move together. When one moves away from its group, does it come back, how fast, and under what market conditions? This project studies those questions on daily data for the 500 most-traded US stocks from October 2021 to October 2026, a period that includes the 2022 rate hikes, the 2023 regional bank failures, and recent markets.

## Research questions

1. How quickly do stock-specific mispricings correct (half-life in days)?
2. When do they correct faster or slower: in calm vs. stressed markets, for larger vs. smaller stocks?
3. Which signal predicts corrections best: a classical mean-reversion model, a machine learning model, or a neural network?
4. Would trading on these signals be profitable after transaction costs?

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/01_download_massive.py` | Downloads 5 years of daily prices for all US stocks (including later-delisted stocks, to avoid survivorship bias), dividends, and splits from the Massive API |
| 2 | `src/02_build_universe.py` | Keeps common stocks, computes daily total returns with dividends, and selects the 500 most-traded stocks each month using only the previous month's data |
| 3 | `src/03_check_universe.py` | Checks extreme moves and missing data inside the universe; known spin-off errors are corrected in `config/return_corrections.csv` |
| 4 | `src/04_build_residuals.py` | Removes common factors (Fama-French and PCA, estimated out of sample on rolling windows) to get each stock's residual return |

## Findings so far

- **Factor models work as intended.** Average pairwise correlation between stocks falls from 0.098 (raw returns) to 0.015 (Fama-French 5) and −0.001 (PCA 5).
- **Robust estimation matters.** Plain least-squares betas were distorted by a few extreme return days, making pooled out-of-sample variance explained fall as factors were added (PCA 10: −11%). Winsorizing returns for beta estimation only fixed this (PCA 10: 31%), and the median stock showed the same rising pattern throughout.
- **About five factors are enough.** For the median stock, PCA 5 explains 35% of daily variance out of sample and PCA 10 explains 36%.

## Data

Price data comes from the [Massive](https://massive.com) API (formerly Polygon.io) and is licensed for personal use, so it is **not included** in this repository. Fama-French factors come from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). To reproduce the results, a Massive Stocks plan with 5 years of history is required.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/01_download_massive.py      # asks for your Massive API key
python src/02_build_universe.py
python src/03_check_universe.py
python src/04_build_residuals.py
```

## Reference

Methodology inspired by "Deep Learning Statistical Arbitrage" (2024), Stanford University, which decomposes statistical arbitrage into residual construction, signal extraction, and trading allocation.

## Author

Daniel Eichel
