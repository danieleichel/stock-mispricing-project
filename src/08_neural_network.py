"""
Step 8: a neural network that reads the raw 30-day path (PyTorch).

Steps 5-7 described each stock's recent past with numbers we designed
(s-score, 5-day return, ...). The paper's main idea is to let a network
learn its own patterns from the raw path. This step does a small version:

  input    3 channels x 30 days, for each stock-day:
             1. daily residual return / the stock's 30-day residual volatility
             2. cumulative residual path (same scaling), starting at 0
             3. log of volume vs. its 30-day median
  network  1D convolutional neural network (CNN): two convolution layers
           slide small filters along the 30 days looking for short shapes
           (e.g. "two down days, then a bounce"), then two dense layers
           (about 9,000 parameters in total)
  target   next 5 days' residual return (same as Step 7), demeaned across
           stocks each day and scaled by the stock's volatility
  ensemble 3 networks with different random starts, averaged (networks are
           noisy; averaging is standard practice)

Same rows, target and walk-forward schedule as Step 7, so all five models
are compared on exactly the same out-of-sample stock-days:
  - each quarter: train on all earlier data (5-day embargo), predict the quarter
  - inside each training set, the last 3 months are held out to decide when
    to stop training (early stopping on validation rank IC)

Before using real data, the script runs a self-test: it trains the network
on simulated paths with a known pattern and stops if it cannot learn it.

Install once:   pip install torch
Run from the project folder:
  python src/08_neural_network.py              PCA-5 residuals
  python src/08_neural_network.py --selftest   only the self-test

Needs: Steps 2, 2b, 4, 5, 6, 7.

Creates:
  data/processed/all_predictions_<model>.parquet   all five models' out-of-sample predictions
  outputs/tables/model_comparison_<model>.csv      all five models, with 95% confidence intervals
  outputs/tables/nn_folds_<model>.csv
  outputs/figures/model_comparison_<model>.png
"""

import importlib.util
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError:
    sys.exit("PyTorch is not installed. Run:  pip install torch")


# ============================================================
# SETTINGS
# ============================================================

WINDOW = 30
HORIZON = 5
N_SEEDS = 3
MAX_EPOCHS = 12
PATIENCE = 3
BATCH_SIZE = 1024
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
VALIDATION_MONTHS = 3
TARGET_CLIP = 5.0

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

torch.set_num_threads(max(1, torch.get_num_threads()))


def load_step7():
    """Reuse Step 7's data preparation and statistics, so the comparison is exact."""
    path = Path(__file__).resolve().parent / "07_machine_learning.py"
    spec = importlib.util.spec_from_file_location("step7", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ============================================================
# 1. THE NETWORK
# ============================================================

class PathCNN(nn.Module):
    """Small 1D CNN: 3 channels x 30 days -> one number (expected scaled return)."""

    def __init__(self, channels=3, window=WINDOW):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(channels, 16, kernel_size=3), nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * (window - 4), 16), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.head(self.features(x)).squeeze(-1)


def predict_numpy(model, x):
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(x), 8192):
            out.append(model(torch.from_numpy(x[start:start + 8192])).numpy())
    return np.concatenate(out) if out else np.array([])


def train_one(x_train, y_train, x_valid, valid_frame, seed, score_fn, verbose=False):
    """Train one network; keep the epoch with the best validation rank IC."""

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = PathCNN(channels=x_train.shape[1], window=x_train.shape[2])
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()

    x_t = torch.from_numpy(x_train)
    y_t = torch.from_numpy(y_train)
    best_score, best_state, best_epoch, waited = -np.inf, None, 0, 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = torch.from_numpy(order[start:start + BATCH_SIZE])
            optimizer.zero_grad()
            loss = loss_fn(model(x_t[batch]), y_t[batch])
            loss.backward()
            optimizer.step()

        score = score_fn(valid_frame.assign(p=predict_numpy(model, x_valid)), "p")
        if verbose:
            print(f"      epoch {epoch}: validation IC {score:+.4f}")
        if score > best_score:
            best_score, best_epoch, waited = score, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_epoch, best_score


# ============================================================
# 2. INPUT WINDOWS
# ============================================================

def build_windows(data, model_name, s7):
    """3 x 30 window for every row of Step 7's data (same rows, same order)."""

    residuals = pd.read_parquet(s7.RESIDUALS_FILE, columns=["date", "ticker", model_name])
    R = (residuals.pivot(index="date", columns="ticker", values=model_name)
                  .sort_index().dropna(how="all"))
    returns = pd.read_parquet(s7.RETURNS_FILE, columns=["date", "ticker", "volume"])
    volume = returns.pivot(index="date", columns="ticker", values="volume").reindex(
        index=R.index, columns=R.columns)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_volume_ratio = np.log(volume / volume.shift(1).rolling(30, min_periods=20).median())
    log_volume_ratio = log_volume_ratio.replace([np.inf, -np.inf], np.nan)

    t_index = R.index.get_indexer(data["date"])
    n_index = R.columns.get_indexer(data["ticker"])
    keep = (t_index >= WINDOW - 1) & (n_index >= 0)

    def windows(wide):
        view = np.lib.stride_tricks.sliding_window_view(wide.to_numpy(), WINDOW, axis=0)
        # view[t - WINDOW + 1, n, :] = days t-29 ... t for stock n
        out = np.full((len(data), WINDOW), np.nan, dtype=np.float32)
        out[keep] = view[t_index[keep] - WINDOW + 1, n_index[keep], :]
        return out

    res = windows(R)
    vol = data["res_vol_30d"].to_numpy(dtype=np.float32)[:, None]
    scaled = res / vol
    cumulative = np.cumsum(np.nan_to_num(scaled), axis=1)
    volume_window = np.nan_to_num(windows(log_volume_ratio))

    x = np.stack([np.nan_to_num(scaled), cumulative, volume_window], axis=1).astype(np.float32)
    x = np.clip(x, -10, 10)
    usable = ~np.isnan(res).any(axis=1) & np.isfinite(vol[:, 0]) & (vol[:, 0] > 0)
    return x, usable


def scaled_target(frame):
    """Training target: demeaned 5-day residual return in units of the stock's 5-day volatility."""
    z = frame["target"] / (frame["res_vol_30d"] * math.sqrt(HORIZON))
    return np.clip(z.to_numpy(dtype=np.float32), -TARGET_CLIP, TARGET_CLIP)


# ============================================================
# 3. SELF-TEST
# ============================================================

def self_test(s7):
    """Simulated paths with a purely non-linear pattern a linear model cannot see.

    True rule: after a 3-day move, stocks bounce back when today's volume is
    high and keep going when it is low (return ~ -0.4 x 3-day move x volume).
    The 3-day move and volume each have zero linear relation to the outcome,
    so a linear model scores ~0. A working network must clearly beat it.
    """
    from sklearn.linear_model import Ridge

    print("Self-test on simulated paths... ", end="", flush=True)
    rng = np.random.default_rng(0)
    n_days, n_stocks = 300, 200
    n = n_days * n_stocks
    x = rng.standard_normal((n, 3, WINDOW)).astype(np.float32)
    x[:, 1, :] = np.cumsum(x[:, 0, :], axis=1)
    last3 = x[:, 0, -3:].sum(axis=1)
    volume_today = x[:, 2, -1]
    signal = -0.4 * last3 * volume_today
    y = (signal + rng.standard_normal(n)).astype(np.float32)

    frame = pd.DataFrame({"date": np.repeat(pd.bdate_range("2020-01-01", periods=n_days), n_stocks),
                          "target": y, "truth": signal})
    train = frame.index < n * 0.7
    valid = (frame.index >= n * 0.7) & (frame.index < n * 0.85)
    test = frame.index >= n * 0.85

    model, epoch, _ = train_one(x[train], y[train], x[valid], frame[valid], seed=0,
                                score_fn=lambda f, c: float(s7.daily_rank_ic(f, c).mean()))
    test_frame = frame[test].assign(p=predict_numpy(model, x[test]))
    learned = float(s7.daily_rank_ic(test_frame, "p").mean())
    best_possible = float(s7.daily_rank_ic(test_frame, "truth").mean())

    flat = x.reshape(n, -1)
    linear = Ridge(alpha=10).fit(flat[train], y[train])
    test_frame["linear"] = linear.predict(flat[test])
    linear_ic = float(s7.daily_rank_ic(test_frame, "linear").mean())

    summary = (f"network IC {learned:.3f}, linear model {linear_ic:.3f}, "
               f"best possible {best_possible:.3f}; stopped at epoch {epoch}")
    if not (learned >= 0.20 and learned >= linear_ic + 0.15):
        raise SystemExit(f"Self-test FAILED ({summary}). The network did not learn a "
                         "pattern that a linear model cannot see.")
    print(f"passed ({summary})")


# ============================================================
# 4. WALK-FORWARD
# ============================================================

def walk_forward(data, x, usable, dates, s7):

    start = data["date"].min()
    first_test = start + pd.DateOffset(months=s7.FIRST_TRAIN_MONTHS)
    quarters = pd.period_range(first_test, data["date"].max(), freq="Q")
    score_fn = lambda frame, column: float(s7.daily_rank_ic(frame, column).mean())

    data = data.assign(pred_cnn=np.nan)
    folds = []

    for quarter in quarters:
        q_start, q_end = quarter.start_time, quarter.end_time
        test_mask = ((data["date"] >= max(q_start, first_test)) & (data["date"] <= q_end)
                     & usable).to_numpy()
        if not test_mask.any():
            continue
        test_start = data.loc[test_mask, "date"].min()

        train_rows = s7.training_rows(data, dates, test_start).index.to_numpy()
        train_rows = train_rows[usable[train_rows]]

        # Last 3 months of the training data decide when to stop (with its own embargo).
        train_dates = data.loc[train_rows, "date"]
        valid_start = train_dates.max() - pd.DateOffset(months=VALIDATION_MONTHS)
        valid_start = next(d for d in dates if d >= valid_start)
        fit_rows = s7.training_rows(data.loc[train_rows], dates, valid_start).index.to_numpy()
        valid_rows = train_rows[train_dates.to_numpy() >= valid_start]

        t0 = time.time()
        y_fit = scaled_target(data.loc[fit_rows])
        valid_frame = data.loc[valid_rows, ["date", "target"]]

        predictions, epochs = [], []
        for seed in range(N_SEEDS):
            model, epoch, _ = train_one(x[fit_rows], y_fit, x[valid_rows], valid_frame,
                                        seed=seed, score_fn=score_fn)
            predictions.append(predict_numpy(model, x[test_mask]))
            epochs.append(epoch)

        # Back to return units, so the prediction ranks expected returns, not z-scores.
        vol = data.loc[test_mask, "res_vol_30d"].to_numpy() * math.sqrt(HORIZON)
        data.loc[test_mask, "pred_cnn"] = np.mean(predictions, axis=0) * vol

        ic = score_fn(data.loc[test_mask], "pred_cnn")
        folds.append({"quarter": str(quarter), "train_rows": len(fit_rows),
                      "validation_rows": len(valid_rows), "test_rows": int(test_mask.sum()),
                      "epochs_chosen": epochs, "ic_cnn": ic})
        print(f"    {quarter}: trained on {len(fit_rows):,} rows, tested on {int(test_mask.sum()):,} "
              f"(IC CNN {ic:+.4f}, epochs {epochs})  [{time.time() - t0:.0f}s]")

    return data, pd.DataFrame(folds)


# ============================================================
# 5. COMPARISON AND FIGURE
# ============================================================

MODEL_LABELS = {
    "pred_sscore": "s-score rule (Step 5)",
    "pred_reversal": "5-day reversal",
    "pred_ridge": "ridge regression",
    "pred_boosting": "gradient boosting",
    "pred_cnn": "neural network (CNN)",
}


def compare(all_predictions, s7):
    rows = []
    for column, label in MODEL_LABELS.items():
        ic = s7.daily_rank_ic(all_predictions, column)
        spread = s7.daily_decile_spread(all_predictions, column)
        ic_mean, ic_t, days = s7.newey_west_mean(ic, lags=HORIZON)
        spread_mean, spread_t, _ = s7.newey_west_mean(spread, lags=HORIZON)
        se = ic_mean / ic_t if ic_t and not np.isnan(ic_t) else np.nan
        rows.append({
            "model": label, "days": days, "mean_rank_ic": ic_mean,
            "ic_ci95_low": ic_mean - 1.96 * se, "ic_ci95_high": ic_mean + 1.96 * se,
            "ic_nw_t_stat": ic_t, "share_days_ic_positive": float((ic > 0).mean()),
            "decile_spread_bps_5d": spread_mean * 1e4, "spread_nw_t_stat": spread_t,
        })
    return pd.DataFrame(rows)


def plot_comparison(table, model_name, path):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    surface, ink, ink_2, muted, grid, axis = (
        "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7")
    color = "#2a78d6"

    fig, ax = plt.subplots(figsize=(8, 4.2), dpi=200)
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    y = np.arange(len(table))[::-1]
    ax.hlines(y, table["ic_ci95_low"], table["ic_ci95_high"], color=color, linewidth=2)
    ax.plot(table["mean_rank_ic"], y, "o", color=color, markersize=8,
            markeredgecolor=surface, markeredgewidth=2)
    for yi, (_, row) in zip(y, table.iterrows()):
        ax.annotate(f"{row['mean_rank_ic']:+.4f}", xy=(row["ic_ci95_high"], yi), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=ink_2)

    ax.axvline(0, color=axis, linewidth=1)
    ax.set_yticks(y)
    ax.set_yticklabels(table["model"], color=ink_2, fontsize=10)
    ax.grid(axis="x", color=grid, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(axis)
    ax.tick_params(axis="x", colors=muted, labelsize=9)
    ax.tick_params(axis="y", length=0)
    ax.margins(x=0.25)
    ax.set_xlabel("Mean daily rank IC, out of sample, with 95% confidence interval",
                  color=ink_2, fontsize=10)
    # The headline follows the data: count intervals that exclude zero.
    beats_chance = int(((table["ic_ci95_low"] > 0) | (table["ic_ci95_high"] < 0)).sum())
    headline = ("No model predicts 5-day residual returns" if beats_chance == 0 else
                f"{beats_chance} of {len(table)} models differ from zero skill")
    ax.set_title(f"{headline} ({model_name.upper()})\n"
                 "Intervals that cross zero = not distinguishable from no skill",
                 color=ink, fontsize=11, loc="left")

    fig.tight_layout()
    fig.savefig(path, facecolor=surface)
    plt.close(fig)


# ============================================================
# MAIN
# ============================================================

def main():

    args = sys.argv[1:]
    model_name = next((a for a in args if not a.startswith("--")), "pca5")

    print("=" * 70)
    print(f"STEP 8 - NEURAL NETWORK ON THE RAW PATH ({model_name})")
    print("=" * 70)

    s7 = load_step7()
    self_test(s7)
    if "--selftest" in args:
        return

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    data, dates = s7.build_features(model_name)
    x, usable = build_windows(data, model_name, s7)
    print(f"  Input windows: {x.shape[0]:,} stock-days x {x.shape[1]} channels x {x.shape[2]} days "
          f"({usable.mean():.1%} usable)")

    print("\n  Walk-forward (3 networks per quarter, averaged)...")
    data, folds = walk_forward(data, x, usable, dates, s7)
    folds.to_csv(TABLES_DIR / f"nn_folds_{model_name}.csv", index=False)

    # Put all five models side by side, on the same out-of-sample rows.
    step7 = pd.read_parquet(PROCESSED_DIR / f"ml_predictions_{model_name}.parquet")
    combined = step7.merge(data[["date", "ticker", "pred_cnn"]], on=["date", "ticker"], how="left")
    combined = combined.dropna(subset=list(MODEL_LABELS))
    combined.to_parquet(PROCESSED_DIR / f"all_predictions_{model_name}.parquet", index=False)

    table = compare(combined, s7)
    table.to_csv(TABLES_DIR / f"model_comparison_{model_name}.csv", index=False)
    plot_comparison(table, model_name, FIGURES_DIR / f"model_comparison_{model_name}.png")

    pd.set_option("display.width", 170)
    print("\n" + "=" * 70)
    print(f"ALL FIVE MODELS, SAME OUT-OF-SAMPLE STOCK-DAYS "
          f"({combined['date'].min().date()} to {combined['date'].max().date()}, "
          f"{len(combined):,} stock-days)")
    print("=" * 70)
    shown = table.copy()
    for col in ["mean_rank_ic", "ic_ci95_low", "ic_ci95_high", "share_days_ic_positive"]:
        shown[col] = shown[col].round(4)
    for col in ["ic_nw_t_stat", "decile_spread_bps_5d", "spread_nw_t_stat"]:
        shown[col] = shown[col].round(2)
    print(shown.to_string(index=False))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
