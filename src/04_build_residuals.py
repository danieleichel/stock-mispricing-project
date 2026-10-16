"""
Step 4: remove common factors to get each stock's residual return.

A stock's daily return has two parts:
  1. the part explained by common factors (the market, size, value, ...)
  2. the part that belongs to the stock alone: the residual

Mean-reversion trading works on part 2. This script estimates residuals out
of sample with two families of factor models:

  Fama-French  K = 1, 3, 5  observed factors from the Kenneth French Data Library
  PCA          K = 1, 3, 5, 10  statistical factors estimated from past returns

For every trading day t and every stock in that month's universe:
  - betas are estimated on the previous 60 trading days only (t-60 ... t-1)
  - residual on day t = return on day t - betas x factor returns on day t
  - PCA factors are rebuilt every day from the previous 252 trading days
So nothing from day t or later is used to estimate the betas: no look-ahead.

Betas are estimated on winsorized returns by default, so one extreme day in
the 60-day window (a meme-stock spike, a bank failure) cannot produce absurd
betas. Residuals always use the actual returns.

Run from the project folder:
  python src/04_build_residuals.py                  winsorized betas (main version)
  python src/04_build_residuals.py --no-winsorize   plain betas (for comparison)

Creates:
  data/raw/french/                                    the downloaded Fama-French file
  data/processed/residuals_<variant>.parquet          one row per stock-day, one column per model
  outputs/tables/residual_model_summary_<variant>.csv diagnostics for each model
"""

import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

BETA_WINDOW = 60      # days used to estimate each stock's betas
PCA_WINDOW = 252      # days used to estimate the PCA factors
FF_LEVELS = {"ff1": ["Mkt-RF"],
             "ff3": ["Mkt-RF", "SMB", "HML"],
             "ff5": ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]}
PCA_LEVELS = [1, 3, 5, 10]

# Robust beta estimation (see winsorize_columns). Run with --no-winsorize
# to reproduce plain least-squares betas for comparison.
WINSORIZE_BETAS = "--no-winsorize" not in sys.argv
WINSOR_MADS = 5

FF_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
          "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FRENCH_DIR = PROJECT_ROOT / "data" / "raw" / "french"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

RETURNS_FILE = PROCESSED_DIR / "returns.parquet"
UNIVERSE_FILE = PROCESSED_DIR / "universe.csv"
FF_ZIP = FRENCH_DIR / "F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

VARIANT = "winsorized_betas" if WINSORIZE_BETAS else "plain_betas"
RESIDUALS_OUTPUT = PROCESSED_DIR / f"residuals_{VARIANT}.parquet"
SUMMARY_OUTPUT = TABLES_DIR / f"residual_model_summary_{VARIANT}.csv"

MODELS = list(FF_LEVELS) + [f"pca{k}" for k in PCA_LEVELS]


# ============================================================
# 1. FAMA-FRENCH FACTORS
# ============================================================

def parse_french_csv(text):
    """Read the daily table from a Kenneth French CSV file.

    The file has a few description lines, then a header row with Mkt-RF,
    then rows like 20240102, 0.12, ... in percent.
    """
    lines = text.splitlines()
    header_index = next(i for i, line in enumerate(lines) if "Mkt-RF" in line)
    columns = [c.strip() for c in lines[header_index].split(",")]
    columns[0] = "date"

    rows = []
    for line in lines[header_index + 1:]:
        parts = [p.strip() for p in line.split(",")]
        if len(parts[0]) == 8 and parts[0].isdigit():
            rows.append(parts)
        elif rows:
            break  # the daily table has ended

    factors = pd.DataFrame(rows, columns=columns)
    factors["date"] = pd.to_datetime(factors["date"], format="%Y%m%d")
    factors = factors.set_index("date").astype(float) / 100  # percent -> decimal
    return factors


def load_fama_french():

    print("\n1. Fama-French factors")
    FRENCH_DIR.mkdir(parents=True, exist_ok=True)

    if not FF_ZIP.exists():
        print("  Downloading from the Kenneth French Data Library...")
        response = requests.get(FF_URL, timeout=60,
                                headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        FF_ZIP.write_bytes(response.content)

    with zipfile.ZipFile(FF_ZIP) as archive:
        name = archive.namelist()[0]
        text = archive.read(name).decode("utf-8", errors="replace")

    factors = parse_french_csv(text)
    print(f"  Factor data from {factors.index.min().date()} "
          f"to {factors.index.max().date()}")
    return factors


# ============================================================
# 2. RETURNS AND UNIVERSE
# ============================================================

def load_returns():

    print("\n2. Returns and universe")

    returns = pd.read_parquet(RETURNS_FILE, columns=["date", "ticker", "ret"])
    wide = returns.pivot(index="date", columns="ticker", values="ret").sort_index()

    universe = pd.read_csv(UNIVERSE_FILE)
    members = universe.groupby("month")["ticker"].apply(set).to_dict()

    print(f"  {wide.shape[0]:,} trading days x {wide.shape[1]:,} stocks")
    return wide, members


# ============================================================
# 3. RESIDUALS
# ============================================================

def winsorize_columns(y):
    """Cap each stock's returns at its median +/- WINSOR_MADS robust standard deviations.

    Used only to ESTIMATE betas, so one extreme day (a meme spike, a crash)
    cannot produce absurd betas. Residuals are always computed from the
    actual, unchanged returns.
    """
    median = np.median(y, axis=0)
    mad = np.median(np.abs(y - median), axis=0) * 1.4826
    mad = np.where(mad > 0, mad, np.inf)
    return np.clip(y, median - WINSOR_MADS * mad, median + WINSOR_MADS * mad)


def regress_window(y, x):
    """Least squares of every column of y on x (with intercept).

    Returns the slope coefficients (without intercept) and each stock's R^2.
    """
    if WINSORIZE_BETAS:
        y = winsorize_columns(y)
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    fitted = design @ coef
    ss_res = ((y - fitted) ** 2).sum(axis=0)
    ss_tot = ((y - y.mean(axis=0)) ** 2).sum(axis=0)
    r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, np.nan)
    return coef[1:], r2


def build_residuals(wide, members, ff):

    print("\n3. Residuals (this takes a few minutes)")

    dates = wide.index
    tickers = np.array(wide.columns)
    position = {t: i for i, t in enumerate(tickers)}
    values = wide.to_numpy()

    ff = ff.reindex(dates)              # NaN on days with no factor data
    rf = ff["RF"].to_numpy()
    ff_values = {name: ff[cols].to_numpy() for name, cols in FF_LEVELS.items()}

    output = []
    r2_records = {m: [] for m in MODELS}

    start = BETA_WINDOW
    for t in range(start, len(dates)):

        month = str(dates[t].to_period("M"))
        if month not in members:
            continue

        cols = np.array([position[s] for s in members[month] if s in position])
        today = values[t, cols]
        window = values[t - BETA_WINDOW:t, cols]

        # A stock gets a residual only with a full 60-day history and a return today.
        ok = ~np.isnan(window).any(axis=0) & ~np.isnan(today)
        if ok.sum() < 50:
            continue

        y = window[:, ok]
        r_today = today[ok]
        day = {"date": np.repeat(dates[t], ok.sum()),
               "ticker": tickers[cols[ok]],
               "ret": r_today}

        # ----- Fama-French -----
        rf_window = rf[t - BETA_WINDOW:t]
        ff_ready = not np.isnan(rf_window).any() and not np.isnan(rf[t])
        for name in FF_LEVELS:
            if ff_ready:
                x = ff_values[name][t - BETA_WINDOW:t]
                beta, r2 = regress_window(y - rf_window[:, None], x)
                day[name] = (r_today - rf[t]) - ff_values[name][t] @ beta
                r2_records[name].append(np.nanmean(r2))
            else:
                day[name] = np.full(ok.sum(), np.nan)

        # ----- PCA -----
        if t >= PCA_WINDOW:
            long_window = values[t - PCA_WINDOW:t, cols]
            pca_ok = ~np.isnan(long_window).any(axis=0)
            history = long_window[:, pca_ok]

            mean = history.mean(axis=0)
            std = history.std(axis=0, ddof=1)
            keep = std > 0
            history, mean, std = history[:, keep], mean[keep], std[keep]

            standardized = (history - mean) / std
            corr = standardized.T @ standardized / (len(history) - 1)
            eigenvalues, eigenvectors = np.linalg.eigh(corr)
            order = np.argsort(eigenvalues)[::-1]
            eigenvectors = eigenvectors[:, order[:max(PCA_LEVELS)]]

            # Eigenportfolio weights: eigenvector divided by each stock's volatility.
            weights = eigenvectors / std[:, None]

            pca_cols = cols[pca_ok][keep]
            factor_window = np.nan_to_num(values[t - BETA_WINDOW:t, pca_cols]) @ weights
            factor_today = np.nan_to_num(values[t, pca_cols]) @ weights

            for k in PCA_LEVELS:
                beta, r2 = regress_window(y, factor_window[:, :k])
                day[f"pca{k}"] = r_today - factor_today[:k] @ beta
                r2_records[f"pca{k}"].append(np.nanmean(r2))
        else:
            for k in PCA_LEVELS:
                day[f"pca{k}"] = np.full(ok.sum(), np.nan)

        output.append(pd.DataFrame(day))

        if t % 100 == 0:
            print(f"    {dates[t].date()} done")

    residuals = pd.concat(output, ignore_index=True)
    return residuals, r2_records


# ============================================================
# 4. DIAGNOSTICS
# ============================================================

def average_pairwise_correlation(frame, column, dates):
    """Mean correlation between every pair of stocks over the given dates."""
    wide = (frame[frame["date"].isin(dates)]
            .pivot(index="date", columns="ticker", values=column)
            .dropna(axis=1))
    corr = np.corrcoef(wide.to_numpy().T)
    upper = corr[np.triu_indices_from(corr, k=1)]
    return float(np.mean(upper)), wide.shape[1]


def average_lag1_autocorrelation(frame, column):
    """Average over stocks of corr(x_t, x_t-1): negative means reversal."""
    data = frame[["ticker", "date", column]].dropna().sort_values(["ticker", "date"])
    lagged = data.groupby("ticker")[column].shift(1)
    data = data.assign(lag=lagged).dropna()
    per_stock = data.groupby("ticker").apply(
        lambda g: g[column].corr(g["lag"]) if len(g) > 30 else np.nan,
        include_groups=False,
    )
    return float(per_stock.mean())


def summarize(residuals, r2_records):

    print("\n4. Diagnostics")

    # Common period where every model has a residual, for a fair comparison.
    complete = residuals.dropna(subset=MODELS)
    common_dates = np.sort(complete["date"].unique())
    last_year = common_dates[-252:]

    rows = []

    raw_corr, n_raw = average_pairwise_correlation(residuals, "ret", last_year)
    rows.append({
        "model": "raw returns",
        "stock_days": int(residuals["ret"].notna().sum()),
        "avg_r2_of_beta_regressions": np.nan,
        "share_of_variance_explained": 0.0,
        "median_stock_variance_explained": 0.0,
        "avg_pairwise_correlation_last_year": raw_corr,
        "avg_lag1_autocorrelation": average_lag1_autocorrelation(residuals, "ret"),
    })

    for model in MODELS:
        both = residuals[["ticker", "ret", model]].dropna()
        explained = 1 - both[model].var() / both["ret"].var()

        # Same measure stock by stock, then the median stock. Unlike the
        # pooled number, a few extremely volatile stocks cannot dominate it.
        per_stock = both.groupby("ticker").agg(
            n=("ret", "size"), v_ret=("ret", "var"), v_res=(model, "var")
        )
        per_stock = per_stock[per_stock["n"] >= BETA_WINDOW]
        median_explained = float((1 - per_stock["v_res"] / per_stock["v_ret"]).median())

        corr, _ = average_pairwise_correlation(residuals, model, last_year)
        rows.append({
            "model": model,
            "stock_days": int(residuals[model].notna().sum()),
            "avg_r2_of_beta_regressions": float(np.mean(r2_records[model])),
            "share_of_variance_explained": float(explained),
            "median_stock_variance_explained": median_explained,
            "avg_pairwise_correlation_last_year": corr,
            "avg_lag1_autocorrelation": average_lag1_autocorrelation(residuals, model),
        })

    summary = pd.DataFrame(rows)

    shown = summary.copy()
    for col in shown.columns[2:]:
        shown[col] = shown[col].round(3)
    print(shown.to_string(index=False))
    print(f"\n  Pairwise correlations use the last 252 common days "
          f"({pd.Timestamp(last_year[0]).date()} to {pd.Timestamp(last_year[-1]).date()}), "
          f"{n_raw} stocks with complete data.")

    return summary


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 4 - BUILD RESIDUALS")
    print(f"Beta estimation: {'winsorized' if WINSORIZE_BETAS else 'plain least squares'}")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    ff = load_fama_french()
    wide, members = load_returns()
    residuals, r2_records = build_residuals(wide, members, ff)

    print(f"\n  Residual stock-days: {len(residuals):,}")
    for model in MODELS:
        have = residuals.loc[residuals[model].notna(), "date"]
        print(f"  {model:5s}: {len(have):,} stock-days, "
              f"{have.min().date()} to {have.max().date()}")

    summary = summarize(residuals, r2_records)

    residuals.to_parquet(RESIDUALS_OUTPUT, index=False)
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Residuals: {RESIDUALS_OUTPUT}")
    print(f"Summary:   {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
