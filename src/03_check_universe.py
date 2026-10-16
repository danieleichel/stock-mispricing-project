"""
Step 3: check data quality inside the universe we will actually study.

Step 2 flagged large daily moves and gaps across ALL common stocks. Most of
those are tiny stocks that never make the top 500. This script answers:
how many of them fall inside the 500 stocks we study each month, and do the
biggest ones look like real events or like data errors?

Run from the project folder:
  python src/03_check_universe.py

Creates:
  outputs/tables/universe_extreme_returns.csv   every >50% move inside the universe
  outputs/tables/universe_summary.csv           counts used in the report
"""

from pathlib import Path

import pandas as pd


EXTREME_RETURN = 0.50

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"

RETURNS_FILE = PROCESSED_DIR / "returns.parquet"
UNIVERSE_FILE = PROCESSED_DIR / "universe.csv"

EXTREME_OUTPUT = TABLES_DIR / "universe_extreme_returns.csv"
SUMMARY_OUTPUT = TABLES_DIR / "universe_summary.csv"


def main():

    print("=" * 70)
    print("STEP 3 - CHECK DATA INSIDE THE UNIVERSE")
    print("=" * 70)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    returns = pd.read_parquet(RETURNS_FILE)
    universe = pd.read_csv(UNIVERSE_FILE)

    returns = returns.sort_values(["ticker", "date"])
    returns["previous_close"] = returns.groupby("ticker")["close"].shift(1)
    returns["month"] = returns["date"].dt.to_period("M").astype(str)

    # Keep only the stock-days where the stock is in that month's universe.
    inside = returns.merge(
        universe[["month", "ticker", "rank"]], on=["month", "ticker"], how="inner"
    )

    summary = {}

    summary["stock-days inside universe"] = len(inside)
    summary["missing returns inside universe"] = int(inside["ret"].isna().sum())
    summary["missing returns inside universe (%)"] = round(
        100 * inside["ret"].isna().mean(), 3
    )

    extreme = inside[inside["ret"].abs() > EXTREME_RETURN].copy()
    summary["moves larger than 50% inside universe"] = len(extreme)

    sizes = universe.groupby("month").size()
    summary["smallest monthly universe"] = int(sizes.min())
    summary["largest monthly universe"] = int(sizes.max())

    print()
    for key, value in summary.items():
        print(f"  {key}: {value:,}" if isinstance(value, int) else f"  {key}: {value}")

    # ----- Biggest moves inside the universe -----
    extreme["abs_ret"] = extreme["ret"].abs()
    extreme = extreme.sort_values("abs_ret", ascending=False)
    extreme_table = extreme[[
        "date", "ticker", "rank", "previous_close", "close", "dividend", "ret"
    ]].copy()
    extreme_table["ret"] = (extreme_table["ret"] * 100).round(1)
    extreme_table = extreme_table.rename(columns={"ret": "return_percent"})
    extreme_table["date"] = extreme_table["date"].dt.date

    print("\nBiggest daily moves inside the universe (up to 20):")
    if extreme_table.empty:
        print("  none")
    else:
        print(extreme_table.head(20).to_string(index=False))

    # ----- Sanity check: who is at the top of the latest month? -----
    latest = universe["month"].max()
    top = universe[universe["month"] == latest].nsmallest(15, "rank")
    print(f"\nTop 15 stocks by dollar volume, universe month {latest}:")
    print("  " + ", ".join(top["ticker"]))

    extreme_table.to_csv(EXTREME_OUTPUT, index=False)
    pd.DataFrame(
        [{"check": k, "value": v} for k, v in summary.items()]
    ).to_csv(SUMMARY_OUTPUT, index=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Saved: {EXTREME_OUTPUT}")
    print(f"Saved: {SUMMARY_OUTPUT}")


if __name__ == "__main__":
    main()
