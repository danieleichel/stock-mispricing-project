"""
Step 5: how fast do stock mispricings correct?

For every stock on every day, take the last 30 days of its residual returns
(from Step 4), add them up into a "residual price" path, and fit the classic
mean-reversion model (Ornstein-Uhlenbeck, estimated as an AR(1)):

    X[l+1] = a + b * X[l] + error

From the fit:
    b             persistence; mean-reverting only if 0 < b < 1
    half-life     -ln(2) / ln(b): days until half of a deviation is gone
    mu, sigma_eq  long-run level a/(1-b) and its equilibrium spread
    s-score       (X_today - mu) / sigma_eq: how far the stock is from normal,
                  in standard deviations. Large positive = "too high",
                  large negative = "too low".

Then the model-free test: do the s-scores actually predict what happens next?
Everything known at the close of day t is used to build the signal; the
forward residual returns cover days t+1 ... t+h only. No look-ahead.

Run from the project folder:
  python src/05_mispricing_correction.py               PCA-5 and FF-5 (main)
  python src/05_mispricing_correction.py pca1 ff3      any models from Step 4

Creates (for each model):
  data/processed/ou_signals_<model>.parquet          one row per stock-day
  outputs/tables/halflife_summary_<model>.csv        half-lives overall and by year
  outputs/tables/sscore_forward_returns_<model>.csv  next-h-day returns by s-score group
  outputs/tables/sscore_slope_<model>.csv            daily cross-sectional regressions
  outputs/figures/correction_curve_<model>.png       the main picture
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

WINDOW = 30                       # days of residuals in each OU fit (as in the paper)
HORIZONS = list(range(1, 21))     # forward days for the correction curve
SAVED_HORIZONS = [1, 5, 10, 20]   # forward returns saved in the signal file
S_CAP = 4.0                       # cap s-scores at +/-4 in the slope regressions
MIN_STOCKS = 5                    # a group needs this many stocks on a day to count
NULL_SIMULATIONS = 200_000        # simulated random walks for the no-reversion benchmark

BUCKETS = [
    ("s <= -2", -np.inf, -2.0),
    ("-2 < s <= -1", -2.0, -1.0),
    ("-1 < s < 1", -1.0, 1.0),
    ("1 <= s < 2", 1.0, 2.0),
    ("s >= 2", 2.0, np.inf),
]

DEFAULT_MODELS = ["pca5", "ff5"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
RESIDUALS_FILE = PROCESSED_DIR / "residuals_winsorized_betas.parquet"


# ============================================================
# STATISTICS HELPERS
# ============================================================

def newey_west_mean(series, lags):
    """Mean of a daily series and its t-statistic with Newey-West standard errors.

    Forward returns over h days overlap from one day to the next, so ordinary
    standard errors would be too small. Newey-West corrects for that.
    """
    x = np.asarray(series, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 10:
        return np.nan, np.nan, n

    mean = x.mean()
    e = x - mean
    variance = e @ e / n
    for k in range(1, min(lags, n - 1) + 1):
        weight = 1 - k / (lags + 1)
        variance += 2 * weight * (e[k:] @ e[:-k]) / n

    se = np.sqrt(variance / n)
    return mean, (mean / se if se > 0 else np.nan), n


# ============================================================
# 1. OU FITS
# ============================================================

def fit_ou(residual_wide):
    """Fit the OU / AR(1) model on every stock-day with a full 30-day window."""

    values = residual_wide.to_numpy()
    T, N = values.shape

    out = {name: np.full((T, N), np.nan) for name in
           ["b", "half_life", "mu", "sigma_eq", "s_score", "r2", "deviation"]}

    for t in range(WINDOW - 1, T):

        window = values[t - WINDOW + 1:t + 1]          # days t-29 ... t
        ok = ~np.isnan(window).any(axis=0)
        if not ok.any():
            continue

        X = np.cumsum(window[:, ok], axis=0)           # residual "price" path
        x, y = X[:-1], X[1:]

        x_mean, y_mean = x.mean(axis=0), y.mean(axis=0)
        sxx = ((x - x_mean) ** 2).sum(axis=0)
        sxy = ((x - x_mean) * (y - y_mean)).sum(axis=0)
        valid = sxx > 0

        b = np.where(valid, sxy / np.where(valid, sxx, 1), np.nan)
        a = y_mean - b * x_mean
        e = y - (a + b * x)
        sse = (e ** 2).sum(axis=0)
        sst = ((y - y_mean) ** 2).sum(axis=0)
        r2 = np.where(sst > 0, 1 - sse / sst, np.nan)

        mean_reverting = valid & (b > 0) & (b < 1)
        with np.errstate(divide="ignore", invalid="ignore"):
            half_life = np.where(mean_reverting, -np.log(2) / np.log(b), np.nan)
            mu = np.where(mean_reverting, a / (1 - b), np.nan)
            error_var = sse / (len(e) - 2)
            sigma_eq = np.where(mean_reverting, np.sqrt(error_var / (1 - b ** 2)), np.nan)
            deviation = X[-1] - mu
            s_score = np.where(sigma_eq > 0, deviation / sigma_eq, np.nan)

        columns = np.flatnonzero(ok)
        out["b"][t, columns] = b
        out["half_life"][t, columns] = half_life
        out["mu"][t, columns] = mu
        out["sigma_eq"][t, columns] = sigma_eq
        out["s_score"][t, columns] = s_score
        out["r2"][t, columns] = r2
        out["deviation"][t, columns] = deviation

    return {k: pd.DataFrame(v, index=residual_wide.index, columns=residual_wide.columns)
            for k, v in out.items()}


# ============================================================
# 2. FORWARD RESIDUAL RETURNS
# ============================================================

def forward_sums(residual_wide, horizon):
    """Sum of residual returns over days t+1 ... t+h, only if all h days exist."""

    values = residual_wide.to_numpy()
    filled = np.nan_to_num(values)
    present = (~np.isnan(values)).astype(int)

    zero_row = np.zeros((1, values.shape[1]))
    c_sum = np.vstack([zero_row, np.cumsum(filled, axis=0)])
    c_cnt = np.vstack([zero_row, np.cumsum(present, axis=0)])

    T = values.shape[0]
    result = np.full(values.shape, np.nan)
    if horizon < T:
        idx = np.arange(T - horizon)
        total = c_sum[idx + 1 + horizon] - c_sum[idx + 1]
        count = c_cnt[idx + 1 + horizon] - c_cnt[idx + 1]
        result[idx] = np.where(count == horizon, total, np.nan)

    return result


# ============================================================
# 3. ANALYSIS
# ============================================================

def random_walk_null(simulations=NULL_SIMULATIONS, seed=0):
    """Distribution of the fitted b when there is NO mean reversion at all.

    A 30-day AR(1) fit is biased toward b < 1, so even a pure random walk
    often looks "mean-reverting" with a short half-life. Simulating random
    walks shows how often that happens by chance, and gives a 5% critical
    value: a stock-day only counts as significant mean reversion if its b is
    lower than 95% of the random-walk b's (the Dickey-Fuller idea, by simulation).
    b does not depend on the volatility, so unit-variance noise is enough.
    """
    rng = np.random.default_rng(seed)
    X = np.cumsum(rng.standard_normal((WINDOW, simulations)), axis=0)
    x, y = X[:-1], X[1:]
    xc = x - x.mean(axis=0)
    b = (xc * (y - y.mean(axis=0))).sum(axis=0) / (xc ** 2).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        half_life = np.where((b > 0) & (b < 1), -np.log(2) / np.log(b), np.nan)
    return {
        "b_critical_5pct": float(np.quantile(b, 0.05)),
        "share_b_below_1": float(np.mean((b > 0) & (b < 1))),
        "median_half_life": float(np.nanmedian(half_life)),
    }


def halflife_summary(fits, null):

    b = fits["b"].stack().rename("b")
    hl = fits["half_life"].stack().rename("half_life")
    frame = b.to_frame().join(hl, how="left").reset_index()
    frame["year"] = frame["date"].dt.year
    critical = null["b_critical_5pct"]

    def describe(group):
        half_lives = group["half_life"].dropna()
        return pd.Series({
            "stock_days_fitted": len(group),
            "share_mean_reverting": (group["half_life"].notna()).mean(),
            "share_significant_5pct": ((group["b"] > 0) & (group["b"] < critical)).mean(),
            "median_half_life_days": half_lives.median(),
            "p25_half_life_days": half_lives.quantile(0.25),
            "p75_half_life_days": half_lives.quantile(0.75),
            "share_half_life_under_5_days": (half_lives < 5).mean(),
            "share_half_life_under_15_days": (half_lives < WINDOW / 2).mean(),
        })

    overall = describe(frame).to_frame().T.assign(period="all")
    by_year = frame.groupby("year").apply(describe, include_groups=False).reset_index()
    by_year["period"] = by_year.pop("year").astype(str)

    table = pd.concat([overall, by_year], ignore_index=True)
    return table[["period"] + [c for c in table.columns if c != "period"]]


def bucket_matrix(s_score):
    """Integer group number (0-4) for each stock-day's s-score, -1 if none."""
    s = s_score.to_numpy()
    groups = np.full(s.shape, -1)
    groups[s <= -2] = 0
    groups[(s > -2) & (s <= -1)] = 1
    groups[(s > -1) & (s < 1)] = 2
    groups[(s >= 1) & (s < 2)] = 3
    groups[s >= 2] = 4
    return groups


def forward_return_tables(residual_wide, s_score):

    groups = bucket_matrix(s_score)
    s = np.clip(s_score.to_numpy(), -S_CAP, S_CAP)

    curve_rows, table_rows, slope_rows = [], [], []
    saved = {}

    for h in HORIZONS:

        forward = forward_sums(residual_wide, h)
        if h in SAVED_HORIZONS:
            saved[h] = forward

        daily_means = {}
        for k, (label, _, _) in enumerate(BUCKETS):
            mask = (groups == k) & ~np.isnan(forward)
            count = mask.sum(axis=1)
            total = np.where(mask, forward, 0).sum(axis=1)
            mean = np.where(count >= MIN_STOCKS, total / np.maximum(count, 1), np.nan)
            daily_means[label] = mean

            avg, t_stat, n_days = newey_west_mean(mean, lags=h)
            curve_rows.append({"horizon_days": h, "group": label,
                               "mean_forward_bps": avg * 1e4, "nw_t_stat": t_stat})
            if h in SAVED_HORIZONS:
                table_rows.append({
                    "horizon_days": h, "group": label, "days": n_days,
                    "avg_stocks_per_day": float(np.mean(count[count >= MIN_STOCKS]))
                    if (count >= MIN_STOCKS).any() else 0.0,
                    "mean_forward_bps": avg * 1e4, "nw_t_stat": t_stat,
                })

        # Low-minus-high: buy the most "too low", sell the most "too high".
        spread = daily_means["s <= -2"] - daily_means["s >= 2"]
        avg, t_stat, n_days = newey_west_mean(spread, lags=h)
        curve_rows.append({"horizon_days": h, "group": "low minus high",
                           "mean_forward_bps": avg * 1e4, "nw_t_stat": t_stat})
        if h in SAVED_HORIZONS:
            table_rows.append({"horizon_days": h, "group": "low minus high",
                               "days": n_days, "avg_stocks_per_day": np.nan,
                               "mean_forward_bps": avg * 1e4, "nw_t_stat": t_stat})

            # Daily cross-sectional regression of forward return on s-score
            # (Fama-MacBeth): the average slope says how much a 1-unit higher
            # s-score changes the next h days' residual return.
            slopes = []
            for t in range(len(s)):
                ok = ~np.isnan(s[t]) & ~np.isnan(forward[t])
                if ok.sum() >= 30:
                    slopes.append(np.polyfit(s[t, ok], forward[t, ok], 1)[0])
            avg, t_stat, n_days = newey_west_mean(slopes, lags=h)
            slope_rows.append({"horizon_days": h, "days": n_days,
                               "slope_bps_per_unit_s": avg * 1e4, "nw_t_stat": t_stat})

    return (pd.DataFrame(curve_rows), pd.DataFrame(table_rows),
            pd.DataFrame(slope_rows), saved)


# ============================================================
# 4. FIGURE
# ============================================================

def plot_correction_curve(curve, model, path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, muted, grid, axis = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
    lines = [
        ("s <= -2", "Far below normal (s ≤ −2)", "#2a78d6", "-"),
        ("-1 < s < 1", "Near normal (−1 < s < 1)", "#898781", "--"),
        ("s >= 2", "Far above normal (s ≥ 2)", "#e34948", "-"),
    ]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for key, label, color, style in lines:
        data = curve[curve["group"] == key].sort_values("horizon_days")
        x = np.concatenate([[0], data["horizon_days"]])
        y = np.concatenate([[0], data["mean_forward_bps"]])
        ax.plot(x, y, color=color, linewidth=2, linestyle=style, label=label)
        ax.annotate(label.split(" (")[0], xy=(x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=9, color=ink_2)

    ax.axhline(0, color=axis, linewidth=1)
    ax.grid(axis="y", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color(axis)
    ax.tick_params(colors=muted, labelsize=9)

    ax.set_xlim(0, max(HORIZONS) + 5)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("Trading days after the signal", color=ink_2, fontsize=10)
    ax.set_ylabel("Average cumulative residual return (bps)", color=ink_2, fontsize=10)
    ax.set_title(f"What happens after a stock moves away from normal ({model.upper()} residuals)",
                 color=ink, fontsize=11, loc="left")
    ax.legend(frameon=False, fontsize=9, labelcolor=ink_2, loc="upper left")

    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def run_model(residuals, model):

    print("\n" + "=" * 70)
    print(f"MODEL: {model}")
    print("=" * 70)

    wide = residuals.pivot(index="date", columns="ticker", values=model).sort_index()
    wide = wide.dropna(how="all")
    print(f"  {wide.shape[0]:,} days x {wide.shape[1]:,} stocks, "
          f"{wide.index.min().date()} to {wide.index.max().date()}")

    print("  Fitting OU models...")
    fits = fit_ou(wide)

    print("  Measuring what happens next...")
    curve, table, slopes, saved = forward_return_tables(wide, fits["s_score"])
    null = random_walk_null()
    halflives = halflife_summary(fits, null)
    benchmark = pd.DataFrame([{
        "period": "random-walk benchmark (no reversion)",
        "share_mean_reverting": null["share_b_below_1"],
        "share_significant_5pct": 0.05,
        "median_half_life_days": null["median_half_life"],
    }])
    halflives = pd.concat([halflives, benchmark], ignore_index=True)

    # ----- Save one row per stock-day -----
    signals = pd.concat(
        {name: frame.stack(future_stack=True) for name, frame in fits.items()}, axis=1)
    for h, forward in saved.items():
        signals[f"fwd_{h}d"] = pd.DataFrame(forward, index=wide.index,
                                            columns=wide.columns).stack(future_stack=True)
    signals = signals.dropna(subset=["b"]).reset_index()
    signals.to_parquet(PROCESSED_DIR / f"ou_signals_{model}.parquet", index=False)

    halflives.to_csv(TABLES_DIR / f"halflife_summary_{model}.csv", index=False)
    table.to_csv(TABLES_DIR / f"sscore_forward_returns_{model}.csv", index=False)
    slopes.to_csv(TABLES_DIR / f"sscore_slope_{model}.csv", index=False)
    plot_correction_curve(curve, model, FIGURES_DIR / f"correction_curve_{model}.png")

    # ----- Print the key results -----
    pd.set_option("display.width", 140)
    print("\n  HALF-LIVES (days)")
    print(halflives.round(3).to_string(index=False))
    print(f"  A stock-day counts as significant mean reversion if b < "
          f"{null['b_critical_5pct']:.3f} (the 5% critical value from "
          f"{NULL_SIMULATIONS:,} simulated random walks).")
    print("  Compare each row with the random-walk benchmark: only the excess "
          "over it is real mean reversion.")

    print("\n  NEXT-DAYS RESIDUAL RETURN BY S-SCORE GROUP (basis points; |t| > 2 is significant)")
    print(table.round(2).to_string(index=False))

    print("\n  SLOPE OF FORWARD RETURN ON S-SCORE (negative = mispricings correct)")
    print(slopes.round(3).to_string(index=False))


def main():

    models = sys.argv[1:] or DEFAULT_MODELS

    print("=" * 70)
    print("STEP 5 - HOW FAST DO MISPRICINGS CORRECT?")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    residuals = pd.read_parquet(RESIDUALS_FILE, columns=["date", "ticker"] + models)

    for model in models:
        run_model(residuals, model)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Tables:  {TABLES_DIR}")
    print(f"Figures: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
