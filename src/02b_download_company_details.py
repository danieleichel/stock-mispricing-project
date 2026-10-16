"""
Step 2b: download company size and industry for every stock in the universe.

For each month and each of that month's 500 stocks, this asks Massive for
the company's details as of the last day of the PREVIOUS month, so size is
measured with information available before the month starts.

Saves (inside data/raw/massive/):
  company_details.csv   month, ticker, market cap, shares outstanding,
                        industry (SIC code and description), listing date

Needs: data/processed/universe.csv (run src/02_build_universe.py first)
       a Massive plan with unlimited requests (Stocks Starter or higher)

Run from the project folder:
  python src/02b_download_company_details.py

It asks for your API key; the key is never saved. You can stop it any time
(Ctrl + C) and run it again: stock-months already saved are skipped.
"""

import csv
import getpass
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://api.massive.com"
WORKERS = 8
MAX_RETRIES = 5
TIMEOUT_SECONDS = 60

PROJECT_ROOT = Path(__file__).resolve().parents[1]
UNIVERSE_FILE = PROJECT_ROOT / "data" / "processed" / "universe.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "raw" / "massive" / "company_details.csv"

COLUMNS = [
    "month", "ticker", "as_of_date", "status",
    "name", "market_cap", "weighted_shares_outstanding",
    "share_class_shares_outstanding", "sic_code", "sic_description",
    "list_date", "active", "type", "cik", "composite_figi",
]


# ============================================================
# HTTP
# ============================================================

_local = threading.local()


def session():
    """One requests session per thread (sessions are not shared between threads)."""
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = "stat-arb-student-project/1.0"
    return _local.session


def fetch_details(ticker, as_of, api_key):
    """Return (status, results dict) for one ticker on one date."""

    url = f"{BASE_URL}/v3/reference/tickers/{ticker}"
    params = {"date": as_of, "apiKey": api_key}

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session().get(url, params=params, timeout=TIMEOUT_SECONDS)
        except requests.RequestException:
            time.sleep(2 ** attempt)
            continue

        if response.status_code == 200:
            return "ok", response.json().get("results", {}) or {}

        if response.status_code == 404:
            return "not_found", {}

        if response.status_code in (401, 403):
            return "not_authorized", {}

        if response.status_code == 429 or response.status_code >= 500:
            time.sleep(2 ** attempt)
            continue

        return f"error_{response.status_code}", {}

    return "failed", {}


# ============================================================
# WORK LIST
# ============================================================

def build_work_list():

    universe = pd.read_csv(UNIVERSE_FILE, usecols=["month", "ticker"])

    # Last calendar day of the previous month, e.g. month 2023-04 -> 2023-03-31.
    month_start = pd.PeriodIndex(universe["month"], freq="M").to_timestamp()
    universe["as_of_date"] = (month_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    done = set()
    if OUTPUT_FILE.exists():
        previous = pd.read_csv(OUTPUT_FILE, usecols=["month", "ticker", "status"])
        # Retry anything that did not succeed last time.
        previous = previous[previous["status"].isin(["ok", "not_found"])]
        done = set(zip(previous["month"], previous["ticker"]))

    todo = [
        (row.month, row.ticker, row.as_of_date)
        for row in universe.itertuples(index=False)
        if (row.month, row.ticker) not in done
    ]
    return universe, todo


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("STEP 2b - DOWNLOAD COMPANY SIZE AND INDUSTRY")
    print("=" * 70)

    if not UNIVERSE_FILE.exists():
        sys.exit("universe.csv not found. Run python src/02_build_universe.py first.")

    universe, todo = build_work_list()
    print(f"Stock-months in universe: {len(universe):,}")
    print(f"Already saved:            {len(universe) - len(todo):,}")
    print(f"To download:              {len(todo):,}")

    if not todo:
        print("\nNothing to download. Done.")
        return

    api_key = getpass.getpass("\nPaste your Massive API key (it will not show), then press Enter: ").strip()
    if not api_key:
        sys.exit("No API key entered. Stopping.")

    status, _ = fetch_details("AAPL", todo[0][2], api_key)
    if status == "not_authorized":
        sys.exit("Massive rejected the API key. Copy it again from the Keys page.")
    print("API key works.\n")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUTPUT_FILE.exists()
    counts = {}
    started = time.time()

    with open(OUTPUT_FILE, "a", newline="") as file:

        writer = csv.DictWriter(file, fieldnames=COLUMNS, extrasaction="ignore")
        if new_file:
            writer.writeheader()

        with ThreadPoolExecutor(max_workers=WORKERS) as pool:

            futures = {
                pool.submit(fetch_details, ticker, as_of, api_key): (month, ticker, as_of)
                for month, ticker, as_of in todo
            }

            for i, future in enumerate(as_completed(futures), start=1):

                month, ticker, as_of = futures[future]
                status, details = future.result()
                counts[status] = counts.get(status, 0) + 1

                writer.writerow({
                    **details,
                    "month": month,
                    "ticker": ticker,
                    "as_of_date": as_of,
                    "status": status,
                })

                if i % 1000 == 0 or i == len(todo):
                    file.flush()
                    elapsed = time.time() - started
                    remaining = elapsed / i * (len(todo) - i)
                    print(f"  {i:,}/{len(todo):,} done - about {remaining / 60:.0f} min left")

    # A re-run appends retried rows. Keep one row per stock-month,
    # preferring a successful answer.
    saved = pd.read_csv(OUTPUT_FILE, dtype={"sic_code": "string"})
    saved["_order"] = saved["status"].map({"ok": 0, "not_found": 1}).fillna(2)
    saved = (saved.sort_values("_order")
                  .drop_duplicates(["month", "ticker"], keep="first")
                  .drop(columns="_order")
                  .sort_values(["month", "ticker"]))
    saved.to_csv(OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    for status, n in sorted(counts.items()):
        print(f"  {status}: {n:,}")
    print(f"\nSaved to: {OUTPUT_FILE}")
    if any(s not in ("ok", "not_found") for s in counts):
        print("Some requests failed. Run the script again to retry only those.")


if __name__ == "__main__":
    main()
