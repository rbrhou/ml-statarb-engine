"""Run this ON YOUR OWN MACHINE (where Yahoo Finance is reachable) to
populate data/ with the return panels the notebooks need, then commit
and push the resulting parquet files.

    python scripts_generate_cache.py
    git add -f data/returns_*.parquet
    git commit -m "Add cached return panels for offline notebook runs"
    git push
"""
from src.data_loader import fetch_equity_returns

NB01 = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AMD","INTC","QCOM",
        "JPM","BAC","WFC","C","GS","XOM","CVX","COP","SLB","EOG"]
NB03 = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AMD","INTC","QCOM",
        "JPM","BAC","WFC","C","GS","MS","XOM","CVX","COP","SLB"]
NB02 = ["AAPL","MSFT","GOOGL","AMZN","NVDA","META","TSLA","AMD","JPM","BAC"]

for name, tickers, start, end in [
    ("notebook 01", NB01, "2023-01-01", "2025-01-01"),
    ("notebooks 03/04", NB03, "2019-01-01", "2025-01-01"),
    ("notebook 02", NB02, "2023-01-01", "2025-01-01"),
]:
    df = fetch_equity_returns(tickers, start_date=start, end_date=end)
    print(f"{name}: {df.shape[0]} days x {df.shape[1]} assets")

print("\nDone. Now run:")
print("  git add -f data/returns_*.parquet && git commit && git push")
