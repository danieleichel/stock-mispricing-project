"""
Download five years of US stock data from Massive (formerly Polygon.io).

What it saves (all inside data/raw/massive/):
  daily/YYYY-MM-DD.csv.gz   every US stock's open/high/low/close/volume for that day
                            (split-adjusted; includes stocks that were later delisted)
  tickers_active.csv        common stocks trading today
  tickers_delisted.csv      common stocks that stopped trading
  dividends.csv             dividend payments (needed to compute total returns)
  splits.csv                stock splits
  download_log.csv          one row per day: rows saved or the error

How to run (from the project folder, with your virtual environment on):
  python src/01_download_massive.py

It asks for your API key when it starts. The key is never saved to a file.
You can stop it any time (Ctrl + C) and run it again: days already saved are skipped.
"""

import csv
import getpass
import gzip
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import requests


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://api.massive.com"

# Stocks Starter includes 5 years of history. Start a few days inside
# that window so the oldest days are not rejected.
YEARS_OF_HISTORY = 5
SAFETY_DAYS = 7

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "massive"
DAILY_DIR = OUTPUT_DIR / "daily"
LOG_FILE = OUTPUT_DIR / "download_log.csv"

DAILY_COLUMNS = ["date", "ticker", "open", "high", "low", "close",
                 "volume", "vwap", "transactions"]

MAX_RETRIES = 5
TIMEOUT_SECONDS = 60


# ============================================================
# HTTP HELPERS
# ============================================================

class NotAuthorizedError(Exception):
    """The plan does not allow this request (for example, a date too far back)."""


def get_json(session, url, params=None):
    """GET a URL and return its JSON, retrying on rate limits and server errors."""

    for attempt in range(1, MAX_RETRIES + 1):

        try:
            response = session.get(url, params=params, timeout=TIMEOUT_SECONDS)
        except requests.RequestException as error:
            wait = 2 ** attempt
            print(f"    Network problem ({error}). Retrying in {wait} s...")
            time.sleep(wait)
            continue

        if response.status_code == 200:
            return response.json()

        if response.status_code in (401, 403):
            raise NotAuthorizedError(response.text[:300])

        if response.status_code == 429 or response.status_code >= 500:
            wait = 2 ** attempt
            print(f"    Server said {response.status_code}. Retrying in {wait} s...")
            time.sleep(wait)
            continue

        raise RuntimeError(
            f"Request failed with status {response.status_code}: {response.text[:300]}"
        )

    raise RuntimeError(f"Gave up after {MAX_RETRIES} attempts: {url}")


def get_all_pages(session, url, params):
    """Follow next_url pagination and return every result."""

    results = []
    data = get_json(session, url, params)
    results.extend(data.get("results", []))

    while data.get("next_url"):
        # next_url already carries the filters; only the key must be added again.
        data = get_json(session, data["next_url"], {"apiKey": params["apiKey"]})
        results.extend(data.get("results", []))
        print(f"    ...{len(results):,} records so far")

    return results


def write_csv(path, rows, columns):
    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# DAILY PRICES
# ============================================================

def trading_day_candidates(start, end):
    """All weekdays between start and end. Holidays simply return no rows."""

    day = start
    while day <= end:
        if day.weekday() < 5:
            yield day
        day += timedelta(days=1)


def download_daily_prices(session, api_key, start, end):

    DAILY_DIR.mkdir(parents=True, exist_ok=True)

    days = list(trading_day_candidates(start, end))
    already = {p.name.split(".")[0] for p in DAILY_DIR.glob("*.csv.gz")}
    logged_empty = set()

    if LOG_FILE.exists():
        with open(LOG_FILE) as file:
            for row in csv.DictReader(file):
                if row["status"] == "no_trading":
                    logged_empty.add(row["date"])

    todo = [d for d in days if d.isoformat() not in already | logged_empty]

    print(f"\nDaily prices: {len(days):,} weekdays from {start} to {end}")
    print(f"Already saved: {len(days) - len(todo):,}   To download: {len(todo):,}\n")

    new_log = not LOG_FILE.exists()
    log = open(LOG_FILE, "a", newline="")
    log_writer = csv.writer(log)
    if new_log:
        log_writer.writerow(["date", "status", "rows"])

    failures = 0
    started = time.time()

    try:
        for i, day in enumerate(todo, start=1):

            day_text = day.isoformat()
            url = f"{BASE_URL}/v2/aggs/grouped/locale/us/market/stocks/{day_text}"

            try:
                data = get_json(session, url, {"adjusted": "true", "apiKey": api_key})
            except NotAuthorizedError as error:
                print(f"  {day_text}: not included in your plan, skipped ({str(error)[:80]})")
                log_writer.writerow([day_text, "not_authorized", 0])
                failures += 1
                continue
            except RuntimeError as error:
                print(f"  {day_text}: FAILED ({error})")
                log_writer.writerow([day_text, "failed", 0])
                failures += 1
                continue

            results = data.get("results") or []

            if not results:
                # Weekend-style gap: a market holiday.
                log_writer.writerow([day_text, "no_trading", 0])
                log.flush()
                continue

            rows = [
                {
                    "date": day_text,
                    "ticker": r.get("T"),
                    "open": r.get("o"),
                    "high": r.get("h"),
                    "low": r.get("l"),
                    "close": r.get("c"),
                    "volume": r.get("v"),
                    "vwap": r.get("vw"),
                    "transactions": r.get("n"),
                }
                for r in results
            ]

            # Write to a temporary name first, so a stopped run never
            # leaves a half-written file that looks complete.
            final_path = DAILY_DIR / f"{day_text}.csv.gz"
            temp_path = DAILY_DIR / f"{day_text}.partial"
            with gzip.open(temp_path, "wt", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=DAILY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows)
            temp_path.rename(final_path)

            log_writer.writerow([day_text, "ok", len(rows)])
            log.flush()

            if i % 25 == 0 or i == len(todo):
                elapsed = time.time() - started
                remaining = elapsed / i * (len(todo) - i)
                print(f"  {i:,}/{len(todo):,} days done "
                      f"({day_text}, {len(rows):,} stocks) "
                      f"- about {remaining / 60:.0f} min left")

    finally:
        log.close()

    return failures


# ============================================================
# REFERENCE DATA
# ============================================================

TICKER_COLUMNS = ["ticker", "name", "type", "primary_exchange", "active",
                  "delisted_utc", "cik", "composite_figi", "share_class_figi",
                  "last_updated_utc"]


def download_tickers(session, api_key):

    url = f"{BASE_URL}/v3/reference/tickers"

    for active, file_name in [("true", "tickers_active.csv"),
                              ("false", "tickers_delisted.csv")]:

        label = "active" if active == "true" else "delisted"
        print(f"\nTicker list ({label} common stocks)...")

        rows = get_all_pages(session, url, {
            "market": "stocks",
            "type": "CS",
            "active": active,
            "limit": 1000,
            "apiKey": api_key,
        })

        write_csv(OUTPUT_DIR / file_name, rows, TICKER_COLUMNS)
        print(f"  Saved {len(rows):,} tickers to {file_name}")


def download_corporate_actions(session, api_key, start):

    print("\nDividends...")
    dividends = get_all_pages(session, f"{BASE_URL}/stocks/v1/dividends", {
        "ex_dividend_date.gte": start.isoformat(),
        "limit": 5000,
        "apiKey": api_key,
    })
    write_csv(OUTPUT_DIR / "dividends.csv", dividends, [
        "ticker", "ex_dividend_date", "cash_amount", "split_adjusted_cash_amount",
        "currency", "declaration_date", "record_date", "pay_date",
        "frequency", "distribution_type", "historical_adjustment_factor", "id",
    ])
    print(f"  Saved {len(dividends):,} dividends")

    print("\nSplits...")
    splits = get_all_pages(session, f"{BASE_URL}/stocks/v1/splits", {
        "execution_date.gte": start.isoformat(),
        "limit": 5000,
        "apiKey": api_key,
    })
    write_csv(OUTPUT_DIR / "splits.csv", splits, [
        "ticker", "execution_date", "adjustment_type", "split_from", "split_to",
        "historical_adjustment_factor", "id",
    ])
    print(f"  Saved {len(splits):,} splits")


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("MASSIVE DATA DOWNLOAD")
    print("=" * 70)

    api_key = getpass.getpass("Paste your Massive API key (it will not show), then press Enter: ").strip()
    if not api_key:
        sys.exit("No API key entered. Stopping.")

    end = date.today() - timedelta(days=1)
    start = date(end.year - YEARS_OF_HISTORY, end.month, end.day) + timedelta(days=SAFETY_DAYS)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers["User-Agent"] = "stat-arb-student-project/1.0"

    # Quick check that the key works before the long download.
    try:
        get_json(session, f"{BASE_URL}/v3/reference/tickers",
                 {"ticker": "AAPL", "apiKey": api_key})
    except NotAuthorizedError:
        sys.exit("Massive rejected the API key. Copy it again from the Keys page.")

    print("API key works.")

    download_tickers(session, api_key)
    download_corporate_actions(session, api_key, start)
    failures = download_daily_prices(session, api_key, start, end)

    saved = len(list(DAILY_DIR.glob("*.csv.gz")))

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
    print(f"Trading days saved: {saved:,}")
    print(f"Days that failed:   {failures:,}")
    print(f"Everything is in:   {OUTPUT_DIR}")
    if failures:
        print("\nSome days failed. Run the script again to retry only those days.")


if __name__ == "__main__":
    main()
