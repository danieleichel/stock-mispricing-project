"""
Step 2: build the stock universe and daily returns.

Reads the raw Massive files from data/raw/massive/ and creates:

  data/processed/returns.parquet
      One row per stock per trading day for every stock that is ever in the
      universe: close, volume, dollar volume, dividend and daily total return.

  data/processed/universe.csv
      For each month, the 500 stocks we study that month and their rank.

  outputs/tables/data_quality_report.csv
      Counts from every cleaning step, so the choices are documented.

Rules (no look-ahead: each month's stocks are chosen with last month's data):
  - Keep common stocks only (type CS), not ETFs, funds or warrants.
  - Total return = (close today + dividend paid today) / close yesterday - 1.
  - A return is only computed when the stock also traded on the previous
    trading day. Otherwise it is left missing instead of spanning a gap.
  - Each month's universe = the 500 stocks with the highest average daily
    dollar volume (close x volume) in the previous month, among stocks that
    traded on every day of that month and closed at $5 or more.

Run from the project folder:
  python src/02_build_universe.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

UNIVERSE_SIZE = 500
MIN_PRICE = 5.0
EXTREME_RETURN = 0.50  # flag daily moves larger than 50% for review

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "massive"
DAILY_DIR = RAW_DIR / "daily"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

CORRECTIONS_FILE = PROJECT_ROOT / "config" / "return_corrections.csv"

RETURNS_OUTPUT = PROCESSED_DIR / "returns.parquet"
UNIVERSE_OUTPUT = PROCESSED_DIR / "universe.csv"
QUALITY_OUTPUT = TABLES_DIR / "data_quality_report.csv"


quality = []


def note(step, value):
    """Record one number for the data-quality report and print it."""
    quality.append({"step": step, "value": value})
    if isinstance(value, (int, np.integer)):
        print(f"  {step}: {value:,}")
    else:
        print(f"  {step}: {value}")


# ============================================================
# 1. COMMON STOCK TICKERS
# ============================================================

def load_common_stock_tickers():

    print("\n1. Common stock tickers")

    active = pd.read_csv(RAW_DIR / "tickers_active.csv", usecols=["ticker", "type"])
    delisted = pd.read_csv(RAW_DIR / "tickers_delisted.csv", usecols=["ticker", "type"])

    tickers = pd.concat([active, delisted])
    tickers = tickers[tickers["type"] == "CS"]

    common = set(tickers["ticker"].dropna())

    note("active common stocks", len(active))
    note("delisted common stocks", len(delisted))
    note("unique common stock tickers", len(common))

    return common


# ============================================================
# 2. DAILY PRICES
# ============================================================

def load_daily_prices(common):

    print("\n2. Daily prices")

    files = sorted(DAILY_DIR.glob("*.csv.gz"))
    if not files:
        raise SystemExit(f"No daily files found in {DAILY_DIR}. Run the download first.")

    note("daily files", len(files))

    frames = []
    total_rows = 0

    for i, path in enumerate(files, start=1):

        day = pd.read_csv(
            path,
            usecols=["date", "ticker", "close", "volume"],
            dtype={"ticker": "string"},
        )
        total_rows += len(day)
        frames.append(day[day["ticker"].isin(common)])

        if i % 250 == 0 or i == len(files):
            print(f"    read {i:,}/{len(files):,} files")

    prices = pd.concat(frames, ignore_index=True)
    prices["date"] = pd.to_datetime(prices["date"])

    note("rows in raw files (all tickers)", total_rows)
    note("rows for common stocks", len(prices))

    # Remove impossible values.
    bad = (prices["close"] <= 0) | prices["close"].isna() | (prices["volume"] < 0)
    note("rows with missing or non-positive price removed", int(bad.sum()))
    prices = prices[~bad]

    duplicates = prices.duplicated(["date", "ticker"], keep="first")
    note("duplicate stock-days removed", int(duplicates.sum()))
    prices = prices[~duplicates]

    prices = prices.sort_values(["ticker", "date"]).reset_index(drop=True)
    prices["dollar_volume"] = prices["close"] * prices["volume"]

    return prices


# ============================================================
# 3. DIVIDENDS AND TOTAL RETURNS
# ============================================================

def add_returns(prices):

    print("\n3. Dividends and total returns")

    dividends = pd.read_csv(
        RAW_DIR / "dividends.csv",
        usecols=["ticker", "ex_dividend_date", "split_adjusted_cash_amount", "currency"],
    )
    dividends = dividends[dividends["currency"].fillna("USD") == "USD"]
    dividends = dividends.dropna(subset=["split_adjusted_cash_amount"])
    dividends["date"] = pd.to_datetime(dividends["ex_dividend_date"])

    # Prices from Massive are split-adjusted to today's share count, so the
    # matching dividend amount is the split-adjusted one.
    dividends = (
        dividends.groupby(["ticker", "date"], as_index=False)["split_adjusted_cash_amount"]
        .sum()
        .rename(columns={"split_adjusted_cash_amount": "dividend"})
    )

    prices = prices.merge(dividends, on=["ticker", "date"], how="left")
    prices["dividend"] = prices["dividend"].fillna(0.0)
    note("stock-days with a dividend", int((prices["dividend"] > 0).sum()))

    # Trading calendar: every date that appears in the data.
    calendar = np.sort(prices["date"].unique())
    position = pd.Series(np.arange(len(calendar)), index=calendar)
    prices["day_index"] = prices["date"].map(position).to_numpy()

    previous_close = prices.groupby("ticker")["close"].shift(1)
    previous_index = prices.groupby("ticker")["day_index"].shift(1)

    # Only compute a return when yesterday was the previous trading day.
    consecutive = previous_index == prices["day_index"] - 1

    prices["ret"] = np.where(
        consecutive,
        (prices["close"] + prices["dividend"]) / previous_close - 1,
        np.nan,
    )

    gaps = previous_index.notna() & ~consecutive
    note("returns left missing because of a trading gap", int(gaps.sum()))

    # Known corporate actions (spin-offs, acquisitions) that the price data
    # cannot represent. Their returns are set to missing, with a reason.
    if CORRECTIONS_FILE.exists():
        corrections = pd.read_csv(CORRECTIONS_FILE, parse_dates=["date"])
        key = prices.set_index(["ticker", "date"]).index
        fix = key.isin(corrections.set_index(["ticker", "date"]).index)
        prices.loc[fix, "ret"] = np.nan
        note("returns set to missing for known corporate actions", int(fix.sum()))
        if fix.sum() < len(corrections):
            print("    Warning: some corrections did not match any stock-day.")

    extreme = prices["ret"].abs() > EXTREME_RETURN
    note(f"daily returns larger than {EXTREME_RETURN:.0%} (flagged)", int(extreme.sum()))
    prices["extreme_return_flag"] = extreme

    return prices, calendar


# ============================================================
# 4. MONTHLY UNIVERSE
# ============================================================

def build_universe(prices, calendar):

    print("\n4. Monthly universe")

    prices["month"] = prices["date"].dt.to_period("M")

    days_per_month = (
        pd.Series(calendar).dt.to_period("M").value_counts().rename("trading_days")
    )

    monthly = prices.groupby(["month", "ticker"]).agg(
        days_traded=("date", "size"),
        mean_dollar_volume=("dollar_volume", "mean"),
        last_close=("close", "last"),
    ).reset_index()

    monthly = monthly.merge(days_per_month, left_on="month", right_index=True)

    eligible = monthly[
        (monthly["days_traded"] == monthly["trading_days"])
        & (monthly["last_close"] >= MIN_PRICE)
    ].copy()

    # Rank within each month, then use that ranking for the NEXT month.
    eligible["rank"] = eligible.groupby("month")["mean_dollar_volume"].rank(
        ascending=False, method="first"
    )
    chosen = eligible[eligible["rank"] <= UNIVERSE_SIZE].copy()
    chosen["month"] = chosen["month"] + 1
    chosen["rank"] = chosen["rank"].astype(int)

    # Drop the first month (formed from a partial month of data) and any
    # month after the data ends.
    months = sorted(prices["month"].unique())
    chosen = chosen[(chosen["month"] > months[1]) & (chosen["month"] <= months[-1])]

    universe = chosen[["month", "ticker", "rank", "mean_dollar_volume", "last_close"]]
    universe = universe.sort_values(["month", "rank"]).reset_index(drop=True)

    sizes = universe.groupby("month").size()
    note("months in universe", len(sizes))
    note("first universe month", str(sizes.index.min()))
    note("last universe month", str(sizes.index.max()))
    note("smallest monthly universe", int(sizes.min()))
    note("unique stocks ever in universe", universe["ticker"].nunique())

    turnover = (
        universe.groupby("month")["ticker"].apply(set)
        .pipe(lambda s: [len(s.iloc[i] - s.iloc[i - 1]) for i in range(1, len(s))])
    )
    note("average new stocks entering per month", round(float(np.mean(turnover)), 1))

    return universe


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 2 - BUILD UNIVERSE AND RETURNS")
    print("=" * 70)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    common = load_common_stock_tickers()
    prices = load_daily_prices(common)
    prices, calendar = add_returns(prices)
    universe = build_universe(prices, calendar)

    # Keep full history for every stock that is ever in the universe, so the
    # later steps have the past data they need (for example 252 days for PCA).
    keep = set(universe["ticker"])
    returns = prices[prices["ticker"].isin(keep)][[
        "date", "ticker", "close", "volume", "dollar_volume",
        "dividend", "ret", "extreme_return_flag",
    ]]

    print("\n5. Save")
    note("stock-days saved for universe stocks", len(returns))

    returns.to_parquet(RETURNS_OUTPUT, index=False)
    universe.assign(month=universe["month"].astype(str)).to_csv(UNIVERSE_OUTPUT, index=False)
    pd.DataFrame(quality).to_csv(QUALITY_OUTPUT, index=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Returns:        {RETURNS_OUTPUT}")
    print(f"Universe:       {UNIVERSE_OUTPUT}")
    print(f"Quality report: {QUALITY_OUTPUT}")


if __name__ == "__main__":
    main()
