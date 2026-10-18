# When Do Stock Mispricings Correct? A Statistical Arbitrage Study

**[Read the full report →](REPORT.md)** · Python · R · PyTorch · scikit-learn · time series · machine learning

When a stock moves away from similar stocks, does it come back? This project tests that question, the foundation of statistical arbitrage, on the **500 most-traded US stocks each month from October 2021 to October 2026** (13.9 million raw daily records, including later-delisted stocks), and compares five ways of predicting the correction, from the classical Ornstein-Uhlenbeck model to a neural network.

## Key findings

- **Fitted mean-reversion half-lives are an illusion.** The median 30-day OU half-life (4.2 days) is identical to that of simulated random walks, and only 6% of stock-days pass a 5% significance test (vs. 5% for random walks).
- **No reliable situation produces corrections.** No group defined by news, company size, sector, market stress or year shows a correction that survives a multiple-testing correction in both factor models. Panel regressions in R agree with the Python results on 92–93% of tests.
- **Machine learning finds nothing either.** In walk-forward tests over 713 out-of-sample days, ridge regression, gradient boosting and a PyTorch CNN all score a rank IC indistinguishable from zero; confidence intervals rule out signals above IC ≈ 0.018.
- **Not tradable.** Long-short portfolios earn about zero before costs and lose up to 5.9% a year after 5 bps trading costs.

![Model comparison](outputs/figures/model_comparison_pca5.png)

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 1 | `src/01_download_massive.py` | Downloads 5 years of daily prices for all US stocks (including delisted ones), dividends and splits from the Massive API |
| 2 | `src/02_build_universe.py` | Total returns with dividends; each month's top 500 stocks by dollar volume, chosen with last month's data only |
| 2b | `src/02b_download_company_details.py` | Market value and industry for every stock-month |
| 3 | `src/03_check_universe.py` | Data-quality checks; spin-off errors corrected in `config/return_corrections.csv` |
| 4 | `src/04_build_residuals.py` | Out-of-sample residuals from Fama-French and PCA factor models, with robust (winsorized) betas |
| 5 | `src/05_mispricing_correction.py` | OU half-lives and s-scores, random-walk benchmark, forward-return tests (Newey-West) |
| 6 | `src/06_when_do_mispricings_correct.py` | Conditional tests by news, size, sector, market stress and year, with Benjamini-Hochberg control |
| 6b | `src/06b_panel_robustness.R` | Same tests as panel regressions with two-way clustered standard errors (R, fixest) |
| 7 | `src/07_machine_learning.py` | Walk-forward ridge regression and gradient boosting on 27 features |
| 8 | `src/08_neural_network.py` | PyTorch CNN on the raw 30-day residual path; five-model comparison |
| 9 | `src/09_backtest.py` | Long-short backtest with trading costs, drawdowns and a Fama-French alpha regression |

Every step was validated on simulated data (pure noise and planted patterns) before being run on real data. The R script and the neural network run their own self-tests at start-up.

## Methods used

Out-of-sample factor models (Fama-French, PCA eigenportfolios) · Ornstein-Uhlenbeck / AR(1) estimation · Monte Carlo critical values · Fama-MacBeth regressions · Newey-West standard errors · panel regressions with two-way clustering · Benjamini-Hochberg false discovery rate · walk-forward validation with embargo · gradient boosting · convolutional neural networks · backtesting with transaction costs

## Data

Price, dividend, split and company data come from the [Massive](https://massive.com) API (formerly Polygon.io) and are licensed for personal use, so they are **not included** in this repository. Fama-French factors come from the [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html). Rerunning step 1 requires a Massive Stocks plan with five years of history.

## How to run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/01_download_massive.py             # asks for your Massive API key
python src/02_build_universe.py
python src/02b_download_company_details.py    # asks for your Massive API key
python src/03_check_universe.py
python src/04_build_residuals.py
python src/05_mispricing_correction.py
python src/06_when_do_mispricings_correct.py
Rscript src/06b_panel_robustness.R            # needs R packages: arrow, fixest, data.table
python src/07_machine_learning.py
python src/08_neural_network.py
python src/09_backtest.py
```

Results are written to `outputs/tables/` and `outputs/figures/`.

## Reference

The study's structure (residuals, signal, trading) follows *Deep Learning Statistical Arbitrage* (Stanford University, 2024); full citation in the [report](REPORT.md#references).

## Author

Daniel Eichel
