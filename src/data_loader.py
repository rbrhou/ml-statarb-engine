import hashlib
from pathlib import Path
import pandas as pd
import yfinance as yf

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _cache_key(tickers: list[str], start_date: str, end_date: str) -> str:
    """Builds a cache filename that is unique to the REQUEST, not just a
    fixed name. The universe and the date range both change what the
    cached frame contains, so both must key the cache -- otherwise one
    notebook's download silently satisfies another notebook's very
    different request.
    """
    payload = f"{','.join(sorted(tickers))}|{start_date}|{end_date}"
    digest = hashlib.sha1(payload.encode()).hexdigest()[:12]
    return f"returns_{len(tickers)}assets_{start_date}_{end_date}_{digest}.parquet"


def fetch_equity_returns(
    tickers: list[str],
    start_date: str,
    end_date: str,
    filename: str | None = None,
) -> pd.DataFrame:
    """Loads equity returns from the local data directory if cached for this
    exact (tickers, start_date, end_date) request; otherwise downloads via
    yfinance and saves locally.

    :param filename: Optional explicit cache filename. When omitted, a name
        derived from the request itself is used, so differing universes or
        date ranges never collide on one file.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    file_path = DATA_DIR / (filename or _cache_key(tickers, start_date, end_date))

    if file_path.exists():
        cached = pd.read_parquet(file_path)
        missing = [t for t in tickers if t not in cached.columns]
        if missing:
            raise ValueError(
                f"Cache {file_path.name} is missing requested tickers {missing}. "
                f"Delete it to force a fresh download."
            )
        print(f"Loading cached returns from {file_path}")
        return cached[list(tickers)]

    print(f"Downloading data for {len(tickers)} assets via yfinance...")
    downloaded_data = yf.download(tickers, start=start_date, end=end_date)
    if downloaded_data is None or downloaded_data.empty:
        raise RuntimeError(
            "yfinance returned no data. Check network egress to "
            "query1/query2.finance.yahoo.com, or supply a cached parquet "
            f"at {file_path}."
        )
    raw_data = downloaded_data["Close"]
    returns = pd.DataFrame(raw_data.pct_change().dropna())

    missing = [t for t in tickers if t not in returns.columns]
    if missing:
        raise RuntimeError(f"yfinance returned no price series for {missing}.")

    returns = returns[list(tickers)]
    returns.to_parquet(file_path)
    return returns
