"""
Step 9: could you make money? A backtest with trading costs.

Turns every model's out-of-sample predictions (Steps 7-8) into a portfolio:

  each day t, after the close:
    - rank the stocks by the model's prediction
    - buy the top 10% and short the bottom 10%, equal weights,
      half the money on each side (dollar-neutral, gross exposure = 1)
  holding:
    - the models predict 5 days ahead, so each day's picks are held for
      5 days: the live portfolio is the average of the last 5 days' picks
      (overlapping portfolios, which keeps trading down)
  profit:
    - earned on day t+1 from the portfolio formed at the close of day t:
      no look-ahead
    - two versions: trading the actual STOCKS (raw total returns: realistic)
      and trading the RESIDUALS (as if every factor exposure were hedged
      for free: an optimistic upper bound)
  costs:
    - cost = cost rate x turnover, at 0, 5 and 10 basis points per dollar
      traded (5 bps is the assumption in the paper)

Statistics: annualized return, volatility and Sharpe ratio, Newey-West
t-statistic of the mean daily return, maximum drawdown, average daily
turnover, break-even cost, and a Fama-French 5-factor regression to check
whether any profit is just exposure to known factors.

Run from the project folder:
  python src/09_backtest.py           PCA-5 (main)
  python src/09_backtest.py ff5

Needs: Steps 2, 4, 7, 8 (data/processed/all_predictions_<model>.parquet).

Creates:
  outputs/tables/backtest_summary_<model>.csv
  outputs/tables/backtest_factor_alpha_<model>.csv
  outputs/tables/backtest_daily_returns_<model>.csv
  outputs/figures/backtest_cumulative_<model>.png
"""

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

HOLDING_DAYS = 5
TOP_FRACTION = 0.10
MIN_STOCKS = 50
COSTS_BPS = [0, 5, 10]
MAIN_COST_BPS = 5
ANNUAL_DAYS = 252

MODEL_LABELS = {
    "pred_sscore": "s-score rule (Step 5)",
    "pred_reversal": "5-day reversal",
    "pred_ridge": "ridge regression",
    "pred_boosting": "gradient boosting",
    "pred_cnn": "neural network (CNN)",
}

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RETURNS_FILE = PROCESSED_DIR / "returns.parquet"
RESIDUALS_FILE = PROCESSED_DIR / "residuals_winsorized_betas.parquet"
FF_ZIP = PROJECT_ROOT / "data" / "raw" / "french" / "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"


# ============================================================
# STATISTICS HELPERS
# ============================================================

def newey_west_t(x, lags=HOLDING_DAYS):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan
    e = x - x.mean()
    variance = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    se = math.sqrt(variance / n)
    return x.mean() / se if se > 0 else np.nan


def max_drawdown(daily):
    wealth = (1 + daily.fillna(0)).cumprod()
    return float((wealth / wealth.cummax() - 1).min())


def summarize(daily, turnover):
    mean, vol = daily.mean(), daily.std()
    return {
        "annual_return_pct": 100 * mean * ANNUAL_DAYS,
        "annual_volatility_pct": 100 * vol * math.sqrt(ANNUAL_DAYS),
        "sharpe_ratio": mean / vol * math.sqrt(ANNUAL_DAYS) if vol > 0 else np.nan,
        "nw_t_stat": newey_west_t(daily),
        "max_drawdown_pct": 100 * max_drawdown(daily),
        "share_days_positive": float((daily > 0).mean()),
        "avg_daily_turnover": float(turnover.mean()),
    }


def ols_newey_west(y, X, lags=HOLDING_DAYS):
    """OLS with Newey-West (HAC) standard errors. X includes a constant column first."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    n, k = X.shape
    xtx_inv = np.linalg.inv(X.T @ X)
    beta = xtx_inv @ X.T @ y
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu / n
    for lag in range(1, lags + 1):
        g = xu[lag:].T @ xu[:-lag] / n
        S += (1 - lag / (lags + 1)) * (g + g.T)
    cov = n * xtx_inv @ S @ xtx_inv
    se = np.sqrt(np.diag(cov))
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return beta, se, r2


# ============================================================
# 1. PORTFOLIOS
# ============================================================

def daily_picks(predictions, column, dates, tickers):
    """Top 10% long, bottom 10% short; +0.5 / -0.5 of the money on each side."""
    data = predictions[["date", "ticker", column]].dropna()
    counts = data.groupby("date")[column].transform("size")
    data = data[counts >= MIN_STOCKS].copy()
    pct = data.groupby("date")[column].rank(pct=True)
    long = data[pct > 1 - TOP_FRACTION]
    short = data[pct <= TOP_FRACTION]

    weights = pd.concat([
        long.assign(w=0.5 / long.groupby("date")["ticker"].transform("size")),
        short.assign(w=-0.5 / short.groupby("date")["ticker"].transform("size")),
    ])
    picks = weights.pivot_table(index="date", columns="ticker", values="w", aggfunc="sum")
    return picks.reindex(index=dates, columns=tickers).fillna(0.0)


def held_portfolio(picks):
    """Average of the last HOLDING_DAYS days' picks (overlapping portfolios)."""
    return picks.rolling(HOLDING_DAYS, min_periods=1).sum() / HOLDING_DAYS


def portfolio_returns(weights, next_returns):
    """Day t+1 return of the portfolio held at the close of day t.

    A held stock with no return the next day (rare: a trading gap or a
    delisting) contributes zero.
    """
    gross = (weights * next_returns.fillna(0.0)).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1)
    turnover.iloc[0] = weights.iloc[0].abs().sum()
    missing = ((weights != 0) & next_returns.isna()).sum().sum()
    return gross, turnover, int(missing)


# ============================================================
# 2. FACTOR REGRESSION
# ============================================================

def load_fama_french():
    path = Path(__file__).resolve().parent / "04_build_residuals.py"
    spec = importlib.util.spec_from_file_location("step4", path)
    step4 = importlib.util.module_from_spec(spec)
    saved_argv, sys.argv = sys.argv, [sys.argv[0]]   # Step 4 reads its own options
    spec.loader.exec_module(step4)
    sys.argv = saved_argv
    import zipfile
    with zipfile.ZipFile(FF_ZIP) as archive:
        text = archive.read(archive.namelist()[0]).decode("utf-8", errors="replace")
    return step4.parse_french_csv(text)


def factor_alpha(daily_by_model, ff):
    rows = []
    factors = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    for label, daily in daily_by_model.items():
        joined = pd.concat([daily.rename("y"), ff[factors]], axis=1, join="inner").dropna()
        if len(joined) < 100:
            continue
        X = np.column_stack([np.ones(len(joined)), joined[factors].to_numpy()])
        beta, se, r2 = ols_newey_west(joined["y"].to_numpy(), X)
        row = {"model": label, "days": len(joined),
               "alpha_annual_pct": 100 * beta[0] * ANNUAL_DAYS,
               "alpha_nw_t_stat": beta[0] / se[0], "r2": r2}
        for name, b, s in zip(factors, beta[1:], se[1:]):
            row[f"beta_{name}"] = b
            row[f"t_{name}"] = b / s
        rows.append(row)
    return pd.DataFrame(rows)


# ============================================================
# 3. FIGURE
# ============================================================

def plot_cumulative(curves, model_name, path, start, end):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, muted, grid, axis = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]   # categorical slots 1-5

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    ends = []
    for (label, daily), color in zip(curves.items(), colors):
        growth = 100 * ((1 + daily.fillna(0)).cumprod() - 1)
        ax.plot(growth.index, growth.values, color=color, linewidth=2, label=label)
        ends.append([growth.values[-1], label, growth.index[-1]])

    # End labels: keep them at least ~4% of the axis height apart so they never overlap.
    low, high = ax.get_ylim()
    gap = 0.045 * (high - low)
    ends.sort(key=lambda e: e[0])
    for i in range(1, len(ends)):
        ends[i][0] = max(ends[i][0], ends[i - 1][0] + gap)
    for y, label, x in ends:
        ax.annotate(label, xy=(x, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=8, color=ink_2, annotation_clip=False)

    ax.axhline(0, color=axis, linewidth=1)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(axis)
    ax.tick_params(colors=muted, labelsize=9)
    ax.margins(x=0.2)
    ax.set_ylabel("Cumulative return after costs (%)", color=ink_2, fontsize=10)
    ax.set_title(f"Long top 10% / short bottom 10%, held 5 days, {MAIN_COST_BPS} bps trading cost "
                 f"({model_name.upper()})\nTrading the actual stocks, {start} to {end}",
                 color=ink, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink_2, loc="best")

    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    model_name = sys.argv[1] if len(sys.argv) > 1 else "pca5"

    print("=" * 70)
    print(f"STEP 9 - BACKTEST WITH TRADING COSTS ({model_name})")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictions = pd.read_parquet(PROCESSED_DIR / f"all_predictions_{model_name}.parquet")
    models = {c: l for c, l in MODEL_LABELS.items() if c in predictions.columns}

    returns = pd.read_parquet(RETURNS_FILE, columns=["date", "ticker", "ret"])
    raw = returns.pivot(index="date", columns="ticker", values="ret").sort_index()
    residuals = pd.read_parquet(RESIDUALS_FILE, columns=["date", "ticker", model_name])
    resid = residuals.pivot(index="date", columns="ticker", values=model_name).reindex(
        index=raw.index, columns=raw.columns)

    # Every trading day from the first to the last prediction (a day with no
    # predictions keeps holding the earlier picks).
    calendar = raw.index
    dates = calendar[(calendar >= predictions["date"].min()) & (calendar <= predictions["date"].max())]
    tickers = sorted(predictions["ticker"].unique())

    # Returns earned on the NEXT trading day, lined up with the formation day.
    next_raw = raw.shift(-1).reindex(index=dates, columns=tickers)
    next_resid = resid.shift(-1).reindex(index=dates, columns=tickers)
    has_next = next_raw.notna().any(axis=1)
    dates = dates[has_next.to_numpy()]
    next_raw, next_resid = next_raw.loc[dates], next_resid.loc[dates]

    print(f"  {len(dates):,} trading days, {dates[0].date()} to {dates[-1].date()}, "
          f"{len(tickers):,} stocks")

    rows, daily_main, daily_gross_stock = [], {}, {}
    daily_table = pd.DataFrame(index=dates)

    for column, label in models.items():
        weights = held_portfolio(daily_picks(predictions, column, dates, tickers))
        n_long = (weights > 0).sum(axis=1).mean()
        n_short = (weights < 0).sum(axis=1).mean()

        for basis, nxt in [("stocks", next_raw), ("residuals", next_resid)]:
            gross, turnover, missing = portfolio_returns(weights, nxt)
            for cost in COSTS_BPS:
                net = gross - cost / 1e4 * turnover
                stats = summarize(net, turnover)
                rows.append({"model": label, "traded": basis, "cost_bps": cost, **stats,
                             "avg_stocks_long": n_long, "avg_stocks_short": n_short,
                             "missing_next_day_returns": missing})
                if basis == "stocks" and cost == MAIN_COST_BPS:
                    daily_main[label] = net
                    daily_table[f"{label} (net {cost} bps)"] = net
            if basis == "stocks":
                daily_gross_stock[label] = gross
                daily_table[f"{label} (gross)"] = gross

        # Break-even cost: the cost rate at which the average daily profit is zero.
        stock_gross = daily_gross_stock[label]
        turnover = portfolio_returns(weights, next_raw)[1]
        breakeven = 1e4 * stock_gross.mean() / turnover.mean() if turnover.mean() > 0 else np.nan
        for row in rows:
            if row["model"] == label and row["traded"] == "stocks":
                row["breakeven_cost_bps"] = breakeven

    summary = pd.DataFrame(rows)
    summary.to_csv(TABLES_DIR / f"backtest_summary_{model_name}.csv", index=False)
    daily_table.to_csv(TABLES_DIR / f"backtest_daily_returns_{model_name}.csv")

    # ----- Factor regression (gross, trading the actual stocks) -----
    alpha = pd.DataFrame()
    if FF_ZIP.exists():
        alpha = factor_alpha(daily_gross_stock, load_fama_french())
        alpha.to_csv(TABLES_DIR / f"backtest_factor_alpha_{model_name}.csv", index=False)

    plot_cumulative(daily_main, model_name, FIGURES_DIR / f"backtest_cumulative_{model_name}.png",
                    dates[0].date(), dates[-1].date())

    # ----- Print -----
    pd.set_option("display.width", 170)
    columns = ["model", "cost_bps", "annual_return_pct", "annual_volatility_pct",
               "sharpe_ratio", "nw_t_stat", "max_drawdown_pct", "avg_daily_turnover"]

    for basis, title in [("stocks", "TRADING THE ACTUAL STOCKS (realistic)"),
                         ("residuals", "TRADING THE RESIDUALS (perfect free hedging: optimistic upper bound)")]:
        print("\n" + "=" * 70)
        print(title)
        print("=" * 70)
        shown = summary[summary["traded"] == basis][columns].copy()
        print(shown.round(2).to_string(index=False))

    print("\n  BREAK-EVEN TRADING COST (bps per dollar traded; above this, the strategy loses money)")
    breakevens = summary[(summary["traded"] == "stocks") & (summary["cost_bps"] == 0)]
    for _, row in breakevens.iterrows():
        print(f"    {row['model']:24s} {row['breakeven_cost_bps']:7.2f} bps   "
              f"(average daily turnover {row['avg_daily_turnover']:.0%} of the portfolio)")

    if not alpha.empty:
        print("\n  FAMA-FRENCH 5-FACTOR REGRESSION (gross daily returns, trading the stocks)")
        shown = alpha[["model", "days", "alpha_annual_pct", "alpha_nw_t_stat",
                       "beta_Mkt-RF", "t_Mkt-RF", "r2"]]
        print(shown.round(3).to_string(index=False))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
