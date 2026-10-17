"""
Step 6: WHEN do stock mispricings correct?

Step 5 found that, on average, the 30-day s-score does not predict residual
returns. An average can hide groups that behave in opposite ways. This step
repeats the test inside groups of stock-days:

  news          a big residual move (|z| >= 3) on unusually high volume
                (>= 2x the 30-day median) in the last 5 days, vs. none.
                A stand-in for news such as earnings: news moves may keep
                going while moves without news may come back.
  size          small / mid / large market value within that month's 500
                stocks (market value as of the previous month end)
  sector        broad industry from the company's SIC code
  stress        calm / normal / stressed market, from the volatility of the
                equal-weighted universe return over the previous 21 days
                (thresholds are full-sample terciles: a description of when
                corrections happen, not a trading rule)
  ou_fit        stock-days whose OU fit beat the random-walk 5% critical
                value in Step 5 vs. those that did not
  year          calendar year

The test inside every group is the same as Step 5: each day, regress the
next h days' residual return on the s-score across stocks (Fama-MacBeth),
then average the daily slopes with Newey-West t-statistics.
Negative slope = mispricings correct. Positive slope = moves keep going.

Many tests are run, so every p-value also gets a Benjamini-Hochberg
q-value (the expected share of false discoveries). Trust results with
q < 0.10; treat the rest as noise.

Run from the project folder:
  python src/06_when_do_mispricings_correct.py            PCA-5 and FF-5
  python src/06_when_do_mispricings_correct.py pca5       one model

Needs: Steps 2, 2b, 4 and 5.

Creates (for each model):
  outputs/tables/conditional_slopes_<model>.csv
  outputs/tables/conditional_differences_<model>.csv
  outputs/figures/news_vs_no_news_<model>.png
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

HORIZONS = [1, 5, 10, 20]
CURVE_HORIZONS = list(range(1, 21))
S_CAP = 4.0
MIN_STOCKS = 10                 # stocks a group needs on a day for that day's regression

ZSCORE_WINDOW = 30              # days used to judge whether a move is unusual
BIG_MOVE_Z = 3.0
HIGH_VOLUME_MULTIPLE = 2.0
NEWS_LOOKBACK = 5               # a news-like move within the last 5 days counts
STRESS_WINDOW = 21
OU_WINDOW = 30                  # must match Step 5

DEFAULT_MODELS = ["pca5", "ff5"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "massive"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

RESIDUALS_FILE = PROCESSED_DIR / "residuals_winsorized_betas.parquet"
RETURNS_FILE = PROCESSED_DIR / "returns.parquet"
UNIVERSE_FILE = PROCESSED_DIR / "universe.csv"
DETAILS_FILE = RAW_DIR / "company_details.csv"


# ============================================================
# STATISTICS HELPERS
# ============================================================

def newey_west_mean(series, lags):
    """Mean of a daily series, its Newey-West standard error, t-stat and day count."""
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 30:
        return np.nan, np.nan, np.nan, n
    mean = x.mean()
    e = x - mean
    variance = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        variance += 2 * (1 - k / (lags + 1)) * (e[k:] @ e[:-k]) / n
    se = math.sqrt(variance / n)
    return mean, se, (mean / se if se > 0 else np.nan), n


def two_sided_p(t):
    """p-value from a t-statistic (normal approximation; days are in the hundreds)."""
    return math.erfc(abs(t) / math.sqrt(2)) if not np.isnan(t) else np.nan


def benjamini_hochberg(p_values):
    """q-values: the false discovery rate at which each test would be rejected."""
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan)
    ok = ~np.isnan(p)
    if not ok.any():
        return q
    order = np.argsort(p[ok])
    ranked = p[ok][order] * ok.sum() / np.arange(1, ok.sum() + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    result = np.empty(ok.sum())
    result[order] = np.minimum(ranked, 1.0)
    q[ok] = result
    return q


def daily_slopes(s, forward, mask):
    """Slope of forward return on s-score, one regression per day, restricted to mask."""
    m = mask & ~np.isnan(s) & ~np.isnan(forward)
    n = m.sum(axis=1)
    x = np.where(m, s, 0.0)
    y = np.where(m, forward, 0.0)
    sx, sy = x.sum(axis=1), y.sum(axis=1)
    sxx, sxy = (x * x).sum(axis=1), (x * y).sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        denominator = sxx - sx * sx / n
        slope = (sxy - sx * sy / n) / denominator
    slope[(n < MIN_STOCKS) | ~(denominator > 0)] = np.nan
    return slope, n


def forward_sums(values, horizon):
    """Sum of residual returns over days t+1 ... t+h, only if all h days exist."""
    filled = np.nan_to_num(values)
    present = (~np.isnan(values)).astype(int)
    zero = np.zeros((1, values.shape[1]))
    c_sum = np.vstack([zero, np.cumsum(filled, axis=0)])
    c_cnt = np.vstack([zero, np.cumsum(present, axis=0)])
    T = values.shape[0]
    out = np.full(values.shape, np.nan)
    if horizon < T:
        i = np.arange(T - horizon)
        total = c_sum[i + 1 + horizon] - c_sum[i + 1]
        count = c_cnt[i + 1 + horizon] - c_cnt[i + 1]
        out[i] = np.where(count == horizon, total, np.nan)
    return out


def random_walk_b_critical(simulations=200_000, seed=0):
    """5% critical value of the AR(1) b for a pure random walk (same as Step 5)."""
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.standard_normal((OU_WINDOW, simulations)), axis=0)
    x, y = X[:-1], X[1:]
    xc = x - x.mean(axis=0)
    b = (xc * (y - y.mean(axis=0))).sum(axis=0) / (xc ** 2).sum(axis=0)
    return float(np.quantile(b, 0.05))


# ============================================================
# SECTORS FROM SIC CODES
# ============================================================

def sector_from_sic(code):
    if pd.isna(code):
        return "Unknown"
    c = int(code)
    if 3570 <= c <= 3579 or 3660 <= c <= 3679 or 7370 <= c <= 7379:
        return "Technology"
    if 2830 <= c <= 2836 or 3840 <= c <= 3851 or 8000 <= c <= 8099 or c == 8731:
        return "Health care"
    if 6000 <= c <= 6799:
        return "Financials"
    if 1200 <= c <= 1399 or 2900 <= c <= 2999:
        return "Energy"
    if 4900 <= c <= 4999:
        return "Utilities"
    if 4800 <= c <= 4899 or 2710 <= c <= 2799 or 7810 <= c <= 7849:
        return "Communications"
    if 5000 <= c <= 5999 or 7000 <= c <= 7299:
        return "Consumer & retail"
    if 1000 <= c <= 1799 or 2000 <= c <= 3999 or 4000 <= c <= 4799:
        return "Industrials & materials"
    return "Other"


# ============================================================
# 1. BUILD THE GROUPS
# ============================================================

def to_wide(long, column, index, columns):
    return (long.pivot(index="date", columns="ticker", values=column)
                .reindex(index=index, columns=columns))


def month_frame_to_wide(monthly, column, index, columns):
    """Spread a (month, ticker) value to every trading day of that month."""
    by_month = monthly.pivot(index="month", columns="ticker", values=column)
    months = pd.Index(index.to_period("M").astype(str))
    return by_month.reindex(index=months, columns=columns).set_axis(index).to_numpy()


def build_groups(residual_wide, signals_wide_b, returns, universe, details, b_critical):

    index, columns = residual_wide.index, residual_wide.columns
    R = residual_wide.to_numpy()
    groups = {}

    # ----- News-like moves: big residual move on unusually high volume -----
    prior_std = residual_wide.shift(1).rolling(ZSCORE_WINDOW, min_periods=20).std()
    z = (residual_wide / prior_std).to_numpy()

    volume = to_wide(returns, "volume", index, columns)
    prior_median_volume = volume.shift(1).rolling(ZSCORE_WINDOW, min_periods=20).median()
    volume_ratio = (volume / prior_median_volume).to_numpy()

    news_day = (np.abs(z) >= BIG_MOVE_Z) & (volume_ratio >= HIGH_VOLUME_MULTIPLE)
    news_recent = (pd.DataFrame(news_day.astype(float), index=index)
                     .rolling(NEWS_LOOKBACK, min_periods=1).max().to_numpy() > 0)
    has_history = ~np.isnan(z) & ~np.isnan(volume_ratio)
    groups["news"] = {
        "news-like move in last 5 days": news_recent & has_history,
        "no news-like move": ~news_recent & has_history,
    }

    # ----- Size: terciles of market value within each month's universe -----
    sized = details.merge(universe[["month", "ticker"]], on=["month", "ticker"])
    sized = sized.dropna(subset=["market_cap"]).copy()
    sized["size_tercile"] = sized.groupby("month")["market_cap"].transform(
        lambda x: pd.qcut(x.rank(method="first"), 3, labels=False))
    size = month_frame_to_wide(sized, "size_tercile", index, columns)
    groups["size"] = {
        "small (bottom third)": size == 0,
        "mid (middle third)": size == 1,
        "large (top third)": size == 2,
    }

    # ----- Sector -----
    details = details.copy()
    details["sector"] = pd.to_numeric(details["sic_code"], errors="coerce").map(sector_from_sic)
    sector_codes = {name: i for i, name in enumerate(sorted(details["sector"].unique()))}
    details["sector_code"] = details["sector"].map(sector_codes)
    sector = month_frame_to_wide(details, "sector_code", index, columns)
    groups["sector"] = {name: sector == code for name, code in sector_codes.items()}

    # ----- Market stress: volatility of the equal-weighted universe return -----
    members = returns.assign(month=returns["date"].dt.to_period("M").astype(str))
    members = members.merge(universe[["month", "ticker"]], on=["month", "ticker"])
    market = members.groupby("date")["ret"].mean().reindex(index)
    market_vol = market.rolling(STRESS_WINDOW, min_periods=15).std()
    terciles = market_vol.quantile([1 / 3, 2 / 3]).to_numpy()
    level = np.select(
        [market_vol <= terciles[0], market_vol <= terciles[1], market_vol > terciles[1]],
        [0, 1, 2], default=-1)
    day_level = np.repeat(level[:, None], len(columns), axis=1)
    groups["stress"] = {
        "calm market": day_level == 0,
        "normal market": day_level == 1,
        "stressed market": day_level == 2,
    }

    # ----- OU fit beat the random-walk benchmark -----
    b = signals_wide_b.to_numpy()
    significant = (b > 0) & (b < b_critical)
    groups["ou_fit"] = {
        "OU fit significant": significant & ~np.isnan(b),
        "OU fit not significant": ~significant & ~np.isnan(b),
    }

    # ----- Year -----
    years = index.year.to_numpy()
    groups["year"] = {
        str(y): np.repeat((years == y)[:, None], len(columns), axis=1)
        for y in np.unique(years)
    }

    groups["all"] = {"all stock-days": ~np.isnan(R)}

    return groups


# ============================================================
# 2. TESTS
# ============================================================

DIFFERENCES = [
    ("news", "news-like move in last 5 days", "no news-like move"),
    ("size", "small (bottom third)", "large (top third)"),
    ("stress", "stressed market", "calm market"),
    ("ou_fit", "OU fit significant", "OU fit not significant"),
]


def run_tests(s, forwards, groups):

    rows, diff_rows = [], []
    daily = {}

    for condition, members in groups.items():
        for group, mask in members.items():
            for h in HORIZONS:
                slopes, n = daily_slopes(s, forwards[h], mask)
                daily[(condition, group, h)] = slopes
                mean, se, t, days = newey_west_mean(slopes, lags=h)
                rows.append({
                    "condition": condition, "group": group, "horizon_days": h,
                    "days": days,
                    "avg_stocks_per_day": float(np.mean(n[n >= MIN_STOCKS])) if (n >= MIN_STOCKS).any() else 0.0,
                    "slope_bps_per_unit_s": mean * 1e4 if not np.isnan(mean) else np.nan,
                    "nw_t_stat": t, "p_value": two_sided_p(t),
                })

    for condition, a, b in DIFFERENCES:
        for h in HORIZONS:
            slopes_a, slopes_b = daily[(condition, a, h)], daily[(condition, b, h)]
            both_days = ~np.isnan(slopes_a) & ~np.isnan(slopes_b)
            if both_days.sum() >= 30:
                # Same days in both groups: test the day-by-day difference.
                mean, se, t, days = newey_west_mean(slopes_a - slopes_b, lags=h)
            else:
                # Groups that never share a day (e.g. calm vs stressed markets):
                # compare the two averages, treating them as independent.
                mean_a, se_a, _, days_a = newey_west_mean(slopes_a, lags=h)
                mean_b, se_b, _, days_b = newey_west_mean(slopes_b, lags=h)
                mean = mean_a - mean_b
                se = math.sqrt(se_a ** 2 + se_b ** 2)
                t = mean / se if se > 0 else np.nan
                days = days_a + days_b
            diff_rows.append({
                "condition": condition, "comparison": f"{a} minus {b}", "horizon_days": h,
                "days": days,
                "difference_bps_per_unit_s": mean * 1e4 if not np.isnan(mean) else np.nan,
                "nw_t_stat": t, "p_value": two_sided_p(t),
            })

    slopes_table = pd.DataFrame(rows)
    slopes_table["q_value_bh"] = benjamini_hochberg(slopes_table["p_value"])
    differences_table = pd.DataFrame(diff_rows)
    differences_table["q_value_bh"] = benjamini_hochberg(differences_table["p_value"])
    return slopes_table, differences_table


# ============================================================
# 3. FIGURE
# ============================================================

def plot_news_curve(s, residual_values, groups, model, path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, muted, grid, axis = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
    series = [
        ("no news-like move", "No news-like move", "#2a78d6"),
        ("news-like move in last 5 days", "News-like move in last 5 days", "#eb6834"),
    ]

    curves = {key: [] for key, _, _ in series}
    for h in CURVE_HORIZONS:
        forward = forward_sums(residual_values, h)
        for key, _, _ in series:
            slopes, _ = daily_slopes(s, forward, groups["news"][key])
            mean, se, _, _ = newey_west_mean(slopes, lags=h)
            curves[key].append((h, mean * 1e4, se * 1e4))

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for key, label, color in series:
        h, mean, se = map(np.array, zip(*curves[key]))
        ax.fill_between(h, mean - 1.96 * se, mean + 1.96 * se, color=color, alpha=0.12, linewidth=0)
        ax.plot(h, mean, color=color, linewidth=2, label=label)
        ax.annotate(label.split(" in ")[0], xy=(h[-1], mean[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=ink_2)

    ax.axhline(0, color=axis, linewidth=1)
    ax.text(0.6, 0, "  below zero = mispricings correct", va="bottom", fontsize=8, color=muted)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(axis)
    ax.tick_params(colors=muted, labelsize=9)
    ax.set_xlim(0.5, max(CURVE_HORIZONS) + 5)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_xlabel("Trading days after the signal", color=ink_2, fontsize=10)
    ax.set_ylabel("Return per 1 SD of s-score (bps), 95% band", color=ink_2, fontsize=10)
    ax.set_title(f"Do mispricings correct after news? ({model.upper()} residuals)",
                 color=ink, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink_2, loc="best")

    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def format_grid(table, value_column):
    """One row per group, one column per horizon: 'value (t)', * if q < 0.10."""
    def cell(row):
        star = "*" if row["q_value_bh"] < 0.10 else ""
        return f"{row[value_column]:7.2f} ({row['nw_t_stat']:5.2f}){star}"
    keys = ["condition", "group"] if "group" in table else ["condition", "comparison"]
    grid = table.assign(cell=table.apply(cell, axis=1)).pivot_table(
        index=keys, columns="horizon_days", values="cell", aggfunc="first", sort=False)
    grid.columns = [f"{h} day" + ("s" if h > 1 else "") for h in grid.columns]
    return grid


def run_model(model, residuals, returns, universe, details, b_critical):

    print("\n" + "=" * 70)
    print(f"MODEL: {model}")
    print("=" * 70)

    residual_wide = (residuals.pivot(index="date", columns="ticker", values=model)
                              .sort_index().dropna(how="all"))
    signals = pd.read_parquet(PROCESSED_DIR / f"ou_signals_{model}.parquet",
                              columns=["date", "ticker", "s_score", "b"])
    s_wide = to_wide(signals, "s_score", residual_wide.index, residual_wide.columns)
    b_wide = to_wide(signals, "b", residual_wide.index, residual_wide.columns)

    s = np.clip(s_wide.to_numpy(), -S_CAP, S_CAP)
    values = residual_wide.to_numpy()
    forwards = {h: forward_sums(values, h) for h in HORIZONS}

    print("  Building groups...")
    groups = build_groups(residual_wide, b_wide, returns, universe, details, b_critical)

    news_share = np.nanmean(groups["news"]["news-like move in last 5 days"][~np.isnan(s)])
    print(f"  Share of stock-days with a news-like move in the last 5 days: {news_share:.1%}")

    print("  Running tests...")
    slopes_table, differences_table = run_tests(s, forwards, groups)
    slopes_table.to_csv(TABLES_DIR / f"conditional_slopes_{model}.csv", index=False)
    differences_table.to_csv(TABLES_DIR / f"conditional_differences_{model}.csv", index=False)

    print("  Drawing the news chart...")
    plot_news_curve(s, values, groups, model, FIGURES_DIR / f"news_vs_no_news_{model}.png")

    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 60)
    print("\n  SLOPE OF FORWARD RESIDUAL RETURN ON S-SCORE, BY GROUP")
    print("  bps per 1 SD of s-score (Newey-West t).  Negative = mispricings correct.")
    print("  * = significant after the multiple-testing correction (q < 0.10)\n")
    print(format_grid(slopes_table, "slope_bps_per_unit_s").to_string())

    print("\n  DIFFERENCES BETWEEN GROUPS (first minus second)\n")
    print(format_grid(differences_table, "difference_bps_per_unit_s").to_string())


def main():

    models = sys.argv[1:] or DEFAULT_MODELS

    print("=" * 70)
    print("STEP 6 - WHEN DO MISPRICINGS CORRECT?")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    residuals = pd.read_parquet(RESIDUALS_FILE, columns=["date", "ticker"] + models)
    returns = pd.read_parquet(RETURNS_FILE, columns=["date", "ticker", "ret", "volume"])
    universe = pd.read_csv(UNIVERSE_FILE)
    details = pd.read_csv(DETAILS_FILE, usecols=["month", "ticker", "market_cap", "sic_code"],
                          dtype={"sic_code": "string"})
    b_critical = random_walk_b_critical()

    for model in models:
        run_model(model, residuals, returns, universe, details, b_critical)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Tables:  {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
