"""
Step 7: can machine learning find what the OU model missed?

Steps 5-6 used one summary of a stock's recent path: the OU s-score. This
step lets flexible models use the whole recent path plus context, and asks
whether they predict the next 5 days' residual return out of sample.

Models (all predict the next 5 days' residual return, ranked across stocks):
  s-score rule        -s_score (the classic signal from Step 5, no fitting)
  5-day reversal      -(residual return over the last 5 days), no fitting
  ridge regression    linear model on all features
  gradient boosting   histogram gradient boosting trees (scikit-learn's
                      implementation of the LightGBM algorithm) on all features

Features (all known at the close of day t):
  residual path       residual return over the last 1, 2, 5, 10, 20, 30 days,
                      also scaled by the stock's own residual volatility
  OU fit              s-score, b, R^2 of the AR(1) fit
  risk                30-day residual volatility, largest move in last 5 days
  raw returns         total return over the last 1, 5, 20 days
  trading activity    volume vs. its 30-day median (today and 5-day average),
                      news-like move in the last 5 days
  company             log market value (previous month end), sector
  market              market volatility (21 days), market return (5 days)

Walk-forward validation (no look-ahead):
  - Hyperparameters are chosen ONCE, on the first 12 months only
    (first 9 months to train, last 3 to validate).
  - Then, for each calendar quarter after the first 12 months: train on all
    earlier data, predict that quarter, roll forward.
  - Training rows end 5 trading days before each test period, so no training
    target overlaps the test period (an "embargo").

Evaluation, on out-of-sample predictions only:
  rank IC            daily rank correlation between prediction and outcome
                     (the standard measure in quantitative investing),
                     averaged with a Newey-West t-statistic
  decile spread      outcome of the top 10% of predictions minus the bottom 10%
  out-of-sample R^2  versus predicting zero (on daily-demeaned outcomes)

Run from the project folder:
  python src/07_machine_learning.py           PCA-5 residuals (main)
  python src/07_machine_learning.py ff5       FF-5 residuals

Needs: Steps 2, 2b, 4, 5, 6.  Install once:  pip install scikit-learn

Creates:
  data/processed/ml_predictions_<model>.parquet   out-of-sample predictions (for the backtest)
  outputs/tables/ml_performance_<model>.csv
  outputs/tables/ml_folds_<model>.csv
  outputs/tables/ml_tuning_<model>.csv
  outputs/tables/ml_feature_importance_<model>.csv
  outputs/figures/ml_cumulative_ic_<model>.png
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge


# ============================================================
# SETTINGS
# ============================================================

HORIZON = 5
FIRST_TRAIN_MONTHS = 12
TUNING_VALIDATION_MONTHS = 3
MIN_STOCKS_PER_DAY = 50
RANDOM_SEED = 0

BOOSTING_GRID = [
    {"max_iter": it, "max_leaf_nodes": leaves}
    for it in (50, 100, 200, 400) for leaves in (7, 15)
]
BOOSTING_FIXED = {"learning_rate": 0.05, "min_samples_leaf": 500,
                  "l2_regularization": 10.0, "max_bins": 255}
RIDGE_GRID = [1.0, 100.0, 10_000.0, 1_000_000.0]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "massive"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

RESIDUALS_FILE = PROCESSED_DIR / "residuals_winsorized_betas.parquet"
RETURNS_FILE = PROCESSED_DIR / "returns.parquet"
UNIVERSE_FILE = PROCESSED_DIR / "universe.csv"
DETAILS_FILE = RAW_DIR / "company_details.csv"

NUMERIC_FEATURES = [
    "res_1d", "res_2d", "res_5d", "res_10d", "res_20d", "res_30d",
    "res_1d_z", "res_2d_z", "res_5d_z", "res_10d_z", "res_20d_z", "res_30d_z",
    "s_score", "ou_b", "ou_r2",
    "res_vol_30d", "max_move_5d_z",
    "raw_1d", "raw_5d", "raw_20d",
    "volume_ratio", "volume_ratio_5d", "news_5d",
    "log_market_cap",
    "market_vol_21d", "market_ret_5d",
]
CATEGORICAL_FEATURES = ["sector"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ============================================================
# STATISTICS HELPERS
# ============================================================

def newey_west_mean(series, lags):
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan, np.nan, n
    mean = x.mean()
    e = x - mean
    variance = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    se = math.sqrt(variance / n)
    return mean, (mean / se if se > 0 else np.nan), n


def daily_rank_ic(frame, prediction, target="target"):
    """Spearman correlation between prediction and outcome, computed each day."""
    data = frame[["date", prediction, target]].dropna()
    counts = data.groupby("date")[target].transform("size")
    data = data[counts >= MIN_STOCKS_PER_DAY]
    ranks = data.groupby("date")[[prediction, target]].rank()
    ranks["date"] = data["date"]
    return ranks.groupby("date").apply(
        lambda g: g[prediction].corr(g[target]), include_groups=False)


def daily_decile_spread(frame, prediction, target="target"):
    """Each day: mean outcome of the top 10% of predictions minus the bottom 10%."""
    data = frame[["date", prediction, target]].dropna()
    counts = data.groupby("date")[target].transform("size")
    data = data[counts >= MIN_STOCKS_PER_DAY].copy()
    pct = data.groupby("date")[prediction].rank(pct=True)
    top = data[pct > 0.9].groupby("date")[target].mean()
    bottom = data[pct <= 0.1].groupby("date")[target].mean()
    return (top - bottom).dropna()


def out_of_sample_r2(frame, prediction, target="target"):
    """R^2 versus predicting zero, after removing each day's average."""
    data = frame[["date", prediction, target]].dropna()
    y = data[target] - data.groupby("date")[target].transform("mean")
    p = data[prediction] - data.groupby("date")[prediction].transform("mean")
    return 1 - ((y - p) ** 2).sum() / (y ** 2).sum()


# ============================================================
# 1. FEATURES
# ============================================================

SECTOR_RANGES = [
    ("Technology", [(3570, 3579), (3660, 3679), (7370, 7379)]),
    ("Health care", [(2830, 2836), (3840, 3851), (8000, 8099), (8731, 8731)]),
    ("Financials", [(6000, 6799)]),
    ("Energy", [(1200, 1399), (2900, 2999)]),
    ("Utilities", [(4900, 4999)]),
    ("Communications", [(4800, 4899), (2710, 2799), (7810, 7849)]),
    ("Consumer & retail", [(5000, 5999), (7000, 7299)]),
    ("Industrials & materials", [(1000, 1799), (2000, 3999), (4000, 4799)]),
]


def sector_from_sic(code):
    """Same sector definitions as Step 6."""
    if pd.isna(code):
        return "Unknown"
    c = int(code)
    for name, ranges in SECTOR_RANGES:
        if any(low <= c <= high for low, high in ranges):
            return name
    return "Other"


def stack(wide, name):
    return wide.stack(future_stack=True).rename(name)


def build_features(model):

    print("  Loading data...")
    residuals = pd.read_parquet(RESIDUALS_FILE, columns=["date", "ticker", model])
    R = (residuals.pivot(index="date", columns="ticker", values=model)
                  .sort_index().dropna(how="all"))
    index, columns = R.index, R.columns

    returns = pd.read_parquet(RETURNS_FILE, columns=["date", "ticker", "ret", "volume"])
    raw = returns.pivot(index="date", columns="ticker", values="ret").reindex(index=index, columns=columns)
    volume = returns.pivot(index="date", columns="ticker", values="volume").reindex(index=index, columns=columns)

    signals = pd.read_parquet(PROCESSED_DIR / f"ou_signals_{model}.parquet",
                              columns=["date", "ticker", "b", "r2"])
    panel = pd.read_parquet(PROCESSED_DIR / f"conditional_panel_{model}.parquet",
                            columns=["date", "ticker", "s_score", "news"])

    print("  Building features...")
    features = {}

    vol30 = R.rolling(30, min_periods=20).std()
    features["res_vol_30d"] = vol30
    for k in (1, 2, 5, 10, 20, 30):
        cumulative = R.rolling(k, min_periods=k).sum()
        features[f"res_{k}d"] = cumulative
        features[f"res_{k}d_z"] = cumulative / (vol30 * math.sqrt(k))
    features["max_move_5d_z"] = R.abs().rolling(5, min_periods=5).max() / vol30

    for k in (1, 5, 20):
        features[f"raw_{k}d"] = raw.rolling(k, min_periods=k).sum()

    median_volume = volume.shift(1).rolling(30, min_periods=20).median()
    ratio = volume / median_volume
    features["volume_ratio"] = np.log(ratio)
    features["volume_ratio_5d"] = np.log(ratio.rolling(5, min_periods=5).mean())

    # Company size and sector, as of the previous month end.
    details = pd.read_csv(DETAILS_FILE, usecols=["month", "ticker", "market_cap", "sic_code"],
                          dtype={"sic_code": "string"})
    months = pd.Index(index.to_period("M").astype(str))
    cap = (details.pivot(index="month", columns="ticker", values="market_cap")
                  .reindex(index=months, columns=columns).set_axis(index))
    features["log_market_cap"] = np.log(cap)
    details["sector"] = pd.to_numeric(details["sic_code"], errors="coerce").map(sector_from_sic)
    sector = (details.pivot(index="month", columns="ticker", values="sector")
                     .reindex(index=months, columns=columns).set_axis(index))

    # Market conditions: equal-weighted universe return.
    universe = pd.read_csv(UNIVERSE_FILE, usecols=["month", "ticker"])
    members = returns.assign(month=returns["date"].dt.to_period("M").astype(str))
    members = members.merge(universe, on=["month", "ticker"])
    market = members.groupby("date")["ret"].mean().reindex(index)
    market_frame = pd.DataFrame({
        "market_vol_21d": market.rolling(21, min_periods=15).std(),
        "market_ret_5d": market.rolling(5, min_periods=5).sum(),
    })

    # Target: residual return over the NEXT 5 days (t+1 ... t+5), complete windows only.
    future = R.shift(-HORIZON).rolling(HORIZON, min_periods=HORIZON).sum()

    # ----- Assemble one row per stock-day that has an s-score -----
    data = panel.rename(columns={"news": "news_label"})
    data["news_5d"] = (data["news_label"] == "news-like move in last 5 days").astype(float)
    data.loc[data["news_label"].isna(), "news_5d"] = np.nan
    data = data.drop(columns="news_label")

    data = data.join(signals.set_index(["date", "ticker"])
                            .rename(columns={"b": "ou_b", "r2": "ou_r2"}),
                     on=["date", "ticker"])
    for name, wide in features.items():
        data = data.join(stack(wide, name), on=["date", "ticker"])
    data = data.join(stack(sector, "sector"), on=["date", "ticker"])
    data = data.join(market_frame, on="date")
    data = data.join(stack(future, "target_raw"), on=["date", "ticker"])

    data["sector"] = data["sector"].fillna("Unknown")
    data = data.replace([np.inf, -np.inf], np.nan)
    data["target"] = data["target_raw"] - data.groupby("date")["target_raw"].transform("mean")
    data = data.sort_values(["date", "ticker"]).reset_index(drop=True)

    print(f"  {len(data):,} stock-days, {data['ticker'].nunique():,} stocks, "
          f"{data['date'].min().date()} to {data['date'].max().date()}")
    return data, list(index)


# ============================================================
# 2. MODELS
# ============================================================

class RidgeModel:
    """Ridge regression with standardization and one-hot sectors (fit on training data only)."""

    def __init__(self, alpha):
        self.alpha = alpha

    def _matrix(self, frame):
        x = frame[NUMERIC_FEATURES].to_numpy(dtype=float)
        x = (x - self.mean) / self.std
        x = np.nan_to_num(x)                      # missing -> training average
        dummies = np.column_stack([(frame["sector"] == s).to_numpy(float) for s in self.sectors])
        return np.column_stack([x, dummies])

    def fit(self, frame):
        x = frame[NUMERIC_FEATURES].to_numpy(dtype=float)
        self.mean = np.nanmean(x, axis=0)
        self.std = np.nanstd(x, axis=0)
        self.std[~(self.std > 0)] = 1.0
        self.sectors = sorted(frame["sector"].unique())
        self.model = Ridge(alpha=self.alpha).fit(self._matrix(frame), frame["target"])
        return self

    def predict(self, frame):
        return self.model.predict(self._matrix(frame))


class BoostingModel:
    """Histogram gradient boosting; sectors as a categorical feature."""

    def __init__(self, **params):
        self.params = params

    def _matrix(self, frame):
        x = frame[NUMERIC_FEATURES].copy()
        x["sector"] = frame["sector"].map(self.sector_codes).fillna(-1).astype(float)
        x.loc[x["sector"] < 0, "sector"] = np.nan
        return x

    def fit(self, frame):
        self.sector_codes = {s: i for i, s in enumerate(sorted(frame["sector"].unique()))}
        self.model = HistGradientBoostingRegressor(
            **BOOSTING_FIXED, **self.params,
            categorical_features=[len(NUMERIC_FEATURES)],
            early_stopping=False, random_state=RANDOM_SEED,
        ).fit(self._matrix(frame), frame["target"])
        return self

    def predict(self, frame):
        return self.model.predict(self._matrix(frame))


def training_rows(data, dates, test_start):
    """Rows whose 5-day target window ends before test_start (the embargo)."""
    position = dates.index(test_start) if test_start in dates else int(np.searchsorted(dates, test_start))
    last_allowed = dates[max(position - HORIZON - 1, 0)]
    rows = data[(data["date"] <= last_allowed) & data["target"].notna()]
    return rows


def mean_ic(frame, prediction):
    ic = daily_rank_ic(frame, prediction)
    return float(ic.mean()) if len(ic) else np.nan


# ============================================================
# 3. TUNING (first 12 months only) AND WALK-FORWARD
# ============================================================

def tune(data, dates, start):

    train_end = start + pd.DateOffset(months=FIRST_TRAIN_MONTHS - TUNING_VALIDATION_MONTHS)
    valid_end = start + pd.DateOffset(months=FIRST_TRAIN_MONTHS)
    validation_start = next(d for d in dates if d >= train_end)

    train = training_rows(data[data["date"] < train_end], dates, validation_start)
    valid = data[(data["date"] >= validation_start) & (data["date"] < valid_end)].copy()

    print(f"  Tuning: train {train['date'].min().date()} to {train['date'].max().date()}, "
          f"validate {valid['date'].min().date()} to {valid['date'].max().date()}")

    rows = []
    for alpha in RIDGE_GRID:
        valid["p"] = RidgeModel(alpha).fit(train).predict(valid)
        rows.append({"model": "ridge regression", "setting": f"alpha={alpha:g}",
                     "validation_ic": mean_ic(valid, "p")})
    for params in BOOSTING_GRID:
        valid["p"] = BoostingModel(**params).fit(train).predict(valid)
        rows.append({"model": "gradient boosting",
                     "setting": f"max_iter={params['max_iter']}, max_leaf_nodes={params['max_leaf_nodes']}",
                     "validation_ic": mean_ic(valid, "p"), **{"_params": params}})

    tuning = pd.DataFrame(rows)
    best_ridge = tuning[tuning["model"] == "ridge regression"].sort_values("validation_ic").iloc[-1]
    best_boost = tuning[tuning["model"] == "gradient boosting"].sort_values("validation_ic").iloc[-1]
    alpha = float(best_ridge["setting"].split("=")[1])
    params = best_boost["_params"]

    print(f"  Chosen: ridge {best_ridge['setting']} (validation IC {best_ridge['validation_ic']:.4f}), "
          f"boosting {best_boost['setting']} (validation IC {best_boost['validation_ic']:.4f})")
    return alpha, params, tuning.drop(columns="_params")


def walk_forward(data, dates, start, alpha, params):

    first_test = start + pd.DateOffset(months=FIRST_TRAIN_MONTHS)
    quarters = pd.period_range(first_test, data["date"].max(), freq="Q")

    predictions, folds, importances = [], [], []

    for quarter in quarters:
        q_start, q_end = quarter.start_time, quarter.end_time
        test = data[(data["date"] >= max(q_start, first_test)) & (data["date"] <= q_end)].copy()
        if test.empty:
            continue
        test_start = test["date"].min()
        train = training_rows(data, dates, test_start)

        t0 = time.time()
        ridge = RidgeModel(alpha).fit(train)
        boost = BoostingModel(**params).fit(train)
        test["pred_ridge"] = ridge.predict(test)
        test["pred_boosting"] = boost.predict(test)
        test["pred_sscore"] = -test["s_score"]
        test["pred_reversal"] = -test["res_5d"]

        # Which features does the boosting model rely on? Permutation importance
        # on this quarter's out-of-sample data (a sample, for speed).
        scored = test.dropna(subset=["target"])
        if len(scored) > 2000:
            sample = scored.sample(min(len(scored), 30_000), random_state=RANDOM_SEED)
            result = permutation_importance(
                boost.model, boost._matrix(sample), sample["target"],
                n_repeats=3, random_state=RANDOM_SEED, scoring="r2")
            importances.append(pd.Series(result.importances_mean, index=FEATURES, name=str(quarter)))

        fold = {"quarter": str(quarter), "train_rows": len(train),
                "train_end": train["date"].max().date(), "test_rows": len(test),
                "test_start": test_start.date(), "test_end": test["date"].max().date()}
        for name in ["sscore", "reversal", "ridge", "boosting"]:
            fold[f"ic_{name}"] = mean_ic(test, f"pred_{name}")
        folds.append(fold)
        predictions.append(test)

        print(f"    {quarter}: trained on {len(train):,} rows, tested on {len(test):,} "
              f"(IC boosting {fold['ic_boosting']:+.4f}, ridge {fold['ic_ridge']:+.4f}, "
              f"s-score {fold['ic_sscore']:+.4f})  [{time.time() - t0:.0f}s]")

    importance = pd.concat(importances, axis=1) if importances else pd.DataFrame()
    return pd.concat(predictions, ignore_index=True), pd.DataFrame(folds), importance


# ============================================================
# 4. EVALUATION AND FIGURE
# ============================================================

MODEL_NAMES = {
    "sscore": "s-score rule (Step 5)",
    "reversal": "5-day reversal",
    "ridge": "ridge regression",
    "boosting": "gradient boosting",
}


def evaluate(predictions):
    rows, ic_series = [], {}
    for key, label in MODEL_NAMES.items():
        column = f"pred_{key}"
        ic = daily_rank_ic(predictions, column)
        spread = daily_decile_spread(predictions, column)
        ic_mean, ic_t, days = newey_west_mean(ic, lags=HORIZON)
        spread_mean, spread_t, _ = newey_west_mean(spread, lags=HORIZON)
        rows.append({
            "model": label, "days": days,
            "mean_rank_ic": ic_mean, "ic_nw_t_stat": ic_t,
            "share_days_ic_positive": float((ic > 0).mean()),
            "decile_spread_bps_5d": spread_mean * 1e4, "spread_nw_t_stat": spread_t,
            # R^2 needs a forecast in return units; the two rules only rank stocks.
            "oos_r2_percent": (100 * out_of_sample_r2(predictions, column)
                               if key in ("ridge", "boosting") else np.nan),
        })
        ic_series[label] = ic
    return pd.DataFrame(rows), ic_series


def plot_cumulative_ic(ic_series, model, path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, muted, grid, axis = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]   # categorical slots 1-4, fixed order

    fig, ax = plt.subplots(figsize=(9, 5), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for (label, ic), color in zip(ic_series.items(), colors):
        cumulative = ic.sort_index().cumsum()
        ax.plot(cumulative.index, cumulative.values, color=color, linewidth=2, label=label)
        ax.annotate(label, xy=(cumulative.index[-1], cumulative.values[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=ink_2)

    ax.axhline(0, color=axis, linewidth=1)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(axis)
    ax.tick_params(colors=muted, labelsize=9)
    ax.margins(x=0.18)
    ax.set_ylabel("Cumulative daily rank IC", color=ink_2, fontsize=10)
    ax.set_title(f"Out-of-sample prediction of 5-day residual returns ({model.upper()})\n"
                 "A line that keeps rising = the model keeps ranking stocks correctly",
                 color=ink, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink_2, loc="best")

    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    model = sys.argv[1] if len(sys.argv) > 1 else "pca5"

    print("=" * 70)
    print(f"STEP 7 - MACHINE LEARNING ({model})")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data, dates = build_features(model)
    start = data["date"].min()

    print("\n  Choosing hyperparameters on the first 12 months only...")
    alpha, params, tuning = tune(data, dates, start)
    tuning.to_csv(TABLES_DIR / f"ml_tuning_{model}.csv", index=False)

    print("\n  Walk-forward: train on the past, predict the next quarter...")
    predictions, folds, importance = walk_forward(data, dates, start, alpha, params)

    performance, ic_series = evaluate(predictions)

    keep = ["date", "ticker", "target", "target_raw",
            "pred_sscore", "pred_reversal", "pred_ridge", "pred_boosting"]
    predictions[keep].to_parquet(PROCESSED_DIR / f"ml_predictions_{model}.parquet", index=False)
    performance.to_csv(TABLES_DIR / f"ml_performance_{model}.csv", index=False)
    folds.to_csv(TABLES_DIR / f"ml_folds_{model}.csv", index=False)

    if not importance.empty:
        summary = pd.DataFrame({
            "mean_importance": importance.mean(axis=1),
            "share_of_quarters_positive": (importance > 0).mean(axis=1),
        }).sort_values("mean_importance", ascending=False)
        summary.to_csv(TABLES_DIR / f"ml_feature_importance_{model}.csv")

    plot_cumulative_ic(ic_series, model, FIGURES_DIR / f"ml_cumulative_ic_{model}.png")

    pd.set_option("display.width", 160)
    print("\n" + "=" * 70)
    print("OUT-OF-SAMPLE PERFORMANCE "
          f"({predictions['date'].min().date()} to {predictions['date'].max().date()})")
    print("=" * 70)
    shown = performance.copy()
    for col in ["mean_rank_ic", "share_days_ic_positive"]:
        shown[col] = shown[col].round(4)
    for col in ["ic_nw_t_stat", "decile_spread_bps_5d", "spread_nw_t_stat", "oos_r2_percent"]:
        shown[col] = shown[col].round(2)
    print(shown.to_string(index=False))
    print("\n  Rank IC: |t| > 2 is significant. A useful stock-ranking signal usually has IC above ~0.02.")

    if not importance.empty:
        print("\n  TOP 10 FEATURES FOR GRADIENT BOOSTING (permutation importance, out of sample)")
        print(summary.head(10).round(6).to_string())

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
