from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from python.market_cache import MarketCache  # <-- change if needed

PYTEST_DB_PATH = Path("data/yfinance/yfinance_cache_pytest.duckdb")

TICKERS10 = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META",
    "NVDA", "TSLA", "JPM", "XOM", "UNH",
]

FIELDS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]


def _make_ohlcv_frame(index, base, include_adj_close = True):
    n = len(index)
    open_ = base + np.arange(n) * 0.1
    high = open_ + 0.5
    low = open_ - 0.5
    close = open_ + 0.05
    vol = (1_000_000 + np.arange(n) * 1000).astype(np.int64)

    data = {
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": vol,
    }
    if include_adj_close:
        data["Adj Close"] = close * 0.99

    df = pd.DataFrame(data, index=index)

    cols = ["Open", "High", "Low", "Close"]
    if include_adj_close:
        cols += ["Adj Close"]
    cols += ["Volume"]
    return df[cols]

def _make_yf_download_df(tickers, start, end, interval, *, multiindex_layout = "ticker_field", include_adj_close = True):
    if interval == "1d":
        idx = pd.date_range(start, end - pd.Timedelta(days=1), freq="D")
    else:
        idx = pd.date_range(start, end, freq="H", inclusive="left")

    if len(idx) == 0:
        return pd.DataFrame()

    if len(tickers) == 1:
        return _make_ohlcv_frame(idx, base=100.0, include_adj_close=include_adj_close)

    frames = {}
    for i, t in enumerate(tickers):
        frames[t] = _make_ohlcv_frame(idx, base=100.0 + i * 10.0, include_adj_close=include_adj_close)

    if multiindex_layout == "ticker_field":
        out = pd.concat(frames, axis=1)
        out.columns = pd.MultiIndex.from_tuples(out.columns, names=["Ticker", "Field"])
        return out

    if multiindex_layout == "field_ticker":
        out = pd.concat(frames, axis=1)
        tuples = [(field, t) for (t, field) in out.columns.to_list()]
        out.columns = pd.MultiIndex.from_tuples(tuples, names=["Field", "Ticker"])
        return out

    raise ValueError("unknown layout")

@pytest.fixture
def cache_db_path() -> Path:
    PYTEST_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if PYTEST_DB_PATH.exists():
        PYTEST_DB_PATH.unlink()
    yield PYTEST_DB_PATH
    if PYTEST_DB_PATH.exists():
        PYTEST_DB_PATH.unlink()


@pytest.fixture
def cache(cache_db_path):
    return MarketCache(db_path=str(cache_db_path))


@pytest.fixture
def fake_yf(monkeypatch):
    class _State:
        layout = "ticker_field"
        include_adj_close = True

    state = _State()

    def _download(*, tickers, start, end, interval, group_by, auto_adjust, threads):
        tickers_list = list(tickers) if isinstance(tickers, (list, tuple)) else [str(tickers)]
        return _make_yf_download_df(
            tickers_list,
            pd.Timestamp(start),
            pd.Timestamp(end),
            interval,
            multiindex_layout=state.layout,
            include_adj_close=state.include_adj_close,
        )

    import yfinance as yf
    monkeypatch.setattr(yf, "download", _download)
    return state

def test_db_initialization_creates_tables(cache):
    df = cache.cache_summary()
    assert list(df.columns) == ["ticker", "interval", "start_ts", "end_ts"]
    assert df.empty


@pytest.mark.parametrize("layout", ["ticker_field", "field_ticker"])
def test_get_prices_downloads_and_formats_multi_ticker(cache: MarketCache, fake_yf, layout: str):
    fake_yf.layout = layout
    start = pd.Timestamp("2020-01-01")
    end = pd.Timestamp("2020-01-06")

    out = cache.get_prices(TICKERS10, start, end, interval="1d")

    assert isinstance(out.columns, pd.MultiIndex)
    assert out.columns.names == ["Ticker", "Price"]
    assert out.index.name == "Date"
    assert len(out.index) == 5

    for t in TICKERS10:
        assert (t, "Open") in out.columns
        assert (t, "High") in out.columns
        assert (t, "Low") in out.columns
        assert (t, "Close") in out.columns
        assert (t, "Adj Close") in out.columns
        assert (t, "Volume") in out.columns


def test_get_prices_single_ticker_returns_single_level_columns(cache, fake_yf):
    fake_yf.layout = "ticker_field"
    start = pd.Timestamp("2020-02-01")
    end = pd.Timestamp("2020-02-04")

    out = cache.get_prices("AAPL", start, end, interval="1d")
    assert not isinstance(out.columns, pd.MultiIndex)
    assert out.index.name == "Date"
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    assert len(out) == 3


def test_cache_hit_does_not_redownload(cache, fake_yf, monkeypatch):
    start = pd.Timestamp("2020-03-01")
    end = pd.Timestamp("2020-03-06")
    _ = cache.get_prices(TICKERS10, start, end, interval="1d")

    import yfinance as yf  # noqa

    def _boom(**kwargs):
        raise AssertionError("yfinance.download called on cache hit")

    monkeypatch.setattr(yf, "download", _boom)

    out2 = cache.get_prices(TICKERS10, start, end, interval="1d")
    assert len(out2.index) == 5


def test_cache_gaps_and_has_all_data(cache: MarketCache, fake_yf):
    t = "AAPL"
    start = pd.Timestamp("2020-04-01")
    end = pd.Timestamp("2020-04-06")

    with cache._read_conn() as conn:
        gaps0 = cache._cache_gaps(conn, t, start, end, "1d")
        assert gaps0 == [(start, end)]
        assert cache._cache_has_all_data(conn, t, start, end, "1d") is False

    _ = cache.get_prices([t], start, end, interval="1d")
    with cache._read_conn() as conn:
        gaps1 = cache._cache_gaps(conn, t, start, end, "1d")
        assert gaps1 == []
        assert cache._cache_has_all_data(conn, t, start, end, "1d") is True


def test_merge_coverage_window_merges_overlaps(cache: MarketCache):
    t = "AAPL"
    interval = "1d"
    a0 = pd.Timestamp("2020-01-01")
    a1 = pd.Timestamp("2020-01-10")
    b0 = pd.Timestamp("2020-01-05")
    b1 = pd.Timestamp("2020-01-20")

    with cache._write_conn() as conn:
        cache._merge_coverage_window(conn, t, interval, a0, a1)
        cache._merge_coverage_window(conn, t, interval, b0, b1)

    summ = cache.cache_summary()
    sub = summ[(summ["ticker"] == t) & (summ["interval"] == interval)]
    assert len(sub) == 1
    assert pd.Timestamp(sub.iloc[0]["start_ts"]) == a0
    assert pd.Timestamp(sub.iloc[0]["end_ts"]) == b1


@pytest.mark.parametrize("include_adj_close", [True, False])
def test_download_handles_missing_adj_close(cache, fake_yf, include_adj_close):
    fake_yf.include_adj_close = include_adj_close

    start = pd.Timestamp("2020-05-01")
    end = pd.Timestamp("2020-05-04")
    out = cache.get_prices(TICKERS10, start, end, interval="1d")

    for t in TICKERS10:
        assert (t, "Adj Close") in out.columns
        if not include_adj_close:
            assert out[(t, "Adj Close")].isna().all()

def test_get_trading_window_bounds_works(cache, fake_yf):
    end = "2020-06-10"
    w0, w1 = cache.get_trading_window_bounds(5, end=end, reference_ticker="^GSPC")
    assert isinstance(w0, pd.Timestamp)
    assert isinstance(w1, pd.Timestamp)
    assert w0 <= w1

    days = pd.date_range(w0, w1, freq="D")
    assert len(days) == 5

def test_unsupported_interval_raises(cache):
    with pytest.raises(ValueError):
        cache.get_prices(TICKERS10, "2020-01-01", "2020-01-02", interval="3min")

@pytest.mark.integration
def test_real_yfinance_smoke():
    cache = MarketCache("data/yfinance/yf_real.duckdb")

    df = cache.get_prices(["AAPL", "MSFT"], "2023-01-01", "2023-01-10")
    assert not df.empty
