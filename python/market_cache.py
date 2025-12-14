from __future__ import annotations

from contextlib import contextmanager
from typing import Iterable

import duckdb
import pandas as pd
import yfinance as yf


class MarketCache:
    def __init__(self, db_path : str = "data/yfinance/yfinance_cache.duckdb"):
        self.db_path = db_path
        self._initialize_db()

    @contextmanager
    def _read_conn(self):
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            yield conn
        finally:
            conn.close()

    @contextmanager
    def _write_conn(self):
        conn = duckdb.connect(self.db_path, read_only=False)
        try:
            yield conn
        finally:
            conn.close()

    def _initialize_db(self):
        # Ensure schema exists (requires a read-write connection).
        with self._write_conn() as conn:
            conn.execute("""
        CREATE TABLE IF NOT EXISTS prices (
            ticker      VARCHAR NOT NULL,
            interval    VARCHAR NOT NULL,    
            ts          TIMESTAMP NOT NULL, 

            open        DOUBLE,
            high        DOUBLE,
            low         DOUBLE,
            close       DOUBLE,
            adj_close   DOUBLE,
            volume      BIGINT,

            source      VARCHAR,   
            downloaded_at TIMESTAMP,        

            PRIMARY KEY (ticker, interval, ts)
        );
        """)

            conn.execute("""
        CREATE TABLE IF NOT EXISTS coverage_windows (
            ticker      VARCHAR NOT NULL,
            interval    VARCHAR NOT NULL,
            start_ts    TIMESTAMP NOT NULL,  -- inclusive
            end_ts      TIMESTAMP NOT NULL,  -- exclusive, or inclusive+1bar

            PRIMARY KEY (ticker, interval, start_ts)
            -- invariant: for each (ticker, interval), windows do NOT overlap and are sorted by start_ts
        );
        """)

    def cache_summary(self):
        # Return a summary of cached data coverage, for every ticker. Show all coverage windows, for every ticker
        query = """
            SELECT ticker, interval, start_ts, end_ts
            FROM coverage_windows
            ORDER BY ticker, interval, start_ts
        """
        with self._read_conn() as conn:
            return conn.execute(query).fetchdf()
    
    def get_prices(self, tickers: Iterable[str] | str, start, end, interval="1d"):
        if interval not in {"1d", "1wk", "1mo", "1h", "30m", "15m", "5m", "2m", "1m"}:
            raise ValueError(f"Unsupported interval: {interval}")

        if isinstance(tickers, str):
            tickers = [tickers]
        tickers = list(tickers)
        start = _to_timestamp(start)
        end = _to_timestamp(end)

        gaps = []
        missing_tickers = []

        with self._read_conn() as conn:
            for t in tickers:
                if not self._cache_has_all_data(conn, t, start, end, interval):
                    missing_tickers.append(t)
                    gaps.extend(self._cache_gaps(conn, t, start, end, interval))

        if missing_tickers:
            start_gap = min(g[0] for g in gaps)
            end_gap = max(g[1] for g in gaps)

            self._download_from_yfinance(missing_tickers, start_gap, end_gap, interval)

        query = """
            SELECT ticker, interval, ts, open, high, low, close, adj_close, volume
            FROM prices
            WHERE ticker IN ({})
                AND interval = ?
                AND ts >= ?
                AND ts < ?
            ORDER BY ticker, ts
        """.format(",".join("?" for _ in tickers))
        params = list(tickers) + [interval, start, end]
        with self._read_conn() as conn:
            data = conn.execute(query, params).fetchdf()

        return self._format_like_yf(data, tickers)

    def get_trading_window_bounds(
        self,
        n_trading_days: int,
        *,
        end=None,
        start=None,
        reference_ticker: str = "^GSPC",
    ) -> tuple[pd.Timestamp, pd.Timestamp]:
        """
        Resolve a fixed-length trading-day window using the S&P 500 trading calendar.

        Provide exactly one of:
        - `start`: window starts on the first trading day on/after `start`
        - `end`: window ends on the last trading day on/before `end`

        Returns (window_start, window_end) as trading-day timestamps, inclusive on both sides.
        """
        if n_trading_days <= 0:
            raise ValueError("n_trading_days must be positive")
        if (start is None) == (end is None):
            raise ValueError("Provide exactly one of start or end")

        if start is not None:
            anchor = _to_timestamp(start).normalize()
            window_start = self._align_trading_day(anchor, side="next", reference_ticker=reference_ticker)
            window_end = self._nth_trading_day_forward(window_start, n_trading_days - 1, reference_ticker=reference_ticker)
            return window_start, window_end

        anchor = _to_timestamp(end).normalize()
        window_end = self._align_trading_day(anchor, side="prev", reference_ticker=reference_ticker)
        window_start = self._nth_trading_day_backward(window_end, n_trading_days - 1, reference_ticker=reference_ticker)
        return window_start, window_end

    def get_trading_window_end(
        self,
        n_trading_days: int,
        *,
        end=None,
        start=None,
        reference_ticker: str = "^GSPC",
    ) -> pd.Timestamp:
        _, window_end = self.get_trading_window_bounds(
            n_trading_days, end=end, start=start, reference_ticker=reference_ticker
        )
        return window_end

    def _trading_days(self, reference_ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
        df = self.get_prices([reference_ticker], start, end, interval="1d")
        if df.empty:
            return pd.DatetimeIndex([])

        idx = pd.DatetimeIndex(pd.to_datetime(df.index))
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert(None)
        idx = idx.normalize()
        return pd.DatetimeIndex(sorted(set(idx.to_pydatetime())))

    def _align_trading_day(self, date: pd.Timestamp, *, side: str, reference_ticker: str) -> pd.Timestamp:
        if side not in {"next", "prev"}:
            raise ValueError("side must be 'next' or 'prev'")

        date = _to_timestamp(date).normalize()
        if side == "next":
            start = date
            end = date + pd.Timedelta(days=14)
            days = self._trading_days(reference_ticker, start, end + pd.Timedelta(days=1))
            for d in days:
                if d >= date:
                    return pd.Timestamp(d)
        else:
            start = date - pd.Timedelta(days=14)
            end = date
            days = self._trading_days(reference_ticker, start, end + pd.Timedelta(days=1))
            for d in reversed(days):
                if d <= date:
                    return pd.Timestamp(d)

        raise ValueError(f"No trading day found near {date} for reference ticker {reference_ticker!r}")

    def _nth_trading_day_forward(self, start_day: pd.Timestamp, offset: int, *, reference_ticker: str) -> pd.Timestamp:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        start_day = _to_timestamp(start_day).normalize()

        span_days = max(30, (offset + 1) * 3)
        for _ in range(10):
            end = start_day + pd.Timedelta(days=span_days)
            days = self._trading_days(reference_ticker, start_day, end + pd.Timedelta(days=1))
            days = days[days >= start_day]
            if len(days) > offset:
                return pd.Timestamp(days[offset])
            span_days *= 2

        raise ValueError(f"Could not resolve {offset} trading days forward from {start_day}")

    def _nth_trading_day_backward(self, end_day: pd.Timestamp, offset: int, *, reference_ticker: str) -> pd.Timestamp:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        end_day = _to_timestamp(end_day).normalize()

        span_days = max(30, (offset + 1) * 3)
        for _ in range(10):
            start = end_day - pd.Timedelta(days=span_days)
            days = self._trading_days(reference_ticker, start, end_day + pd.Timedelta(days=1))
            days = days[days <= end_day]
            if len(days) > offset:
                return pd.Timestamp(days[-(offset + 1)])
            span_days *= 2

        raise ValueError(f"Could not resolve {offset} trading days backward from {end_day}")

    def _download_from_yfinance(self, tickers, start, end, interval):
        print(f"Downloading from yfinance: {tickers} {start} - {end} @ {interval}")
        df = yf.download(
            tickers=list(tickers),
            start=start,
            end=end,
            interval=interval,
            group_by='ticker',
            auto_adjust=False,
            threads=True,
        )

        records = []
        downloaded_at = pd.Timestamp.now()

        for ticker in tickers:
            ticker_df = _select_ticker_frame(df, ticker)
            if ticker_df is None:
                continue
            ticker_df = _flatten_columns(ticker_df).reset_index()

            ts_col = "Date" if "Date" in ticker_df.columns else ("Datetime" if "Datetime" in ticker_df.columns else ticker_df.columns[0])
            for _, row in ticker_df.iterrows():
                record = {
                    "ticker": ticker,
                    "interval": interval,
                    "ts": row[ts_col],
                    "open": row['Open'],
                    "high": row['High'],
                    "low": row['Low'],
                    "close": row['Close'],
                    "adj_close": row['Adj Close'],
                    "volume": row['Volume'],
                    "source": "yfinance",
                    "downloaded_at": downloaded_at,
                }
                records.append(record)

        if records:
            # Only hold a write connection during the actual DB modifications.
            with self._write_conn() as conn:
                conn.executemany("""
                    INSERT OR REPLACE INTO prices 
                    (ticker, interval, ts, open, high, low, close, adj_close, volume, source, downloaded_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [(
                    r['ticker'], r['interval'], r['ts'], r['open'], r['high'], r['low'], r['close'],
                    r['adj_close'], r['volume'], r['source'], r['downloaded_at']
                ) for r in records])

                for ticker in sorted({r["ticker"] for r in records}):
                    self._merge_coverage_window(conn, ticker, interval, start, end)

    def _cache_has_all_data(self, conn, ticker, start, end, interval = '1d'):
        return len(self._cache_gaps(conn, ticker, start, end, interval)) == 0

    def _cache_gaps(self, conn, ticker, start, end, interval):
        start = _to_timestamp(start)
        end = _to_timestamp(end)

        query = """
            SELECT start_ts, end_ts
            FROM coverage_windows
            WHERE ticker = ?
                AND interval = ?
                AND end_ts > ?
                AND start_ts < ?
            ORDER BY start_ts
        """

        rows = conn.execute(query, (ticker, interval, start, end)).fetchall()

        windows = [(row[0], row[1]) for row in rows]

        gaps = []
        cursor = start

        for w_start, w_end in windows:
            if cursor < w_start:
                gap_start = cursor
                gap_end = min(w_start, end)
                if gap_start < gap_end:
                    gaps.append((gap_start, gap_end))

            if w_end > cursor:
                cursor = w_end

            if cursor >= end:
                break

        # gap after the last window
        if cursor < end:
            gaps.append((cursor, end))

        return gaps

    
    def _merge_coverage_window(self, conn, ticker: str, interval: str, start, end) -> None:
        start = _to_timestamp(start)
        end = _to_timestamp(end)

        merged_start = start
        merged_end = end

        while True:
            rows = conn.execute(
                """
                SELECT start_ts, end_ts
                FROM coverage_windows
                WHERE ticker = ?
                  AND interval = ?
                  AND end_ts >= ?
                  AND start_ts <= ?
                """,
                (ticker, interval, merged_start, merged_end),
            ).fetchall()

            if not rows:
                break

            conn.execute(
                """
                DELETE FROM coverage_windows
                WHERE ticker = ?
                  AND interval = ?
                  AND end_ts >= ?
                  AND start_ts <= ?
                """,
                (ticker, interval, merged_start, merged_end),
            )

            new_start = min([merged_start] + [r[0] for r in rows])
            new_end = max([merged_end] + [r[1] for r in rows])

            if new_start == merged_start and new_end == merged_end:
                break

            merged_start, merged_end = new_start, new_end

        conn.execute(
            """
            INSERT OR REPLACE INTO coverage_windows (ticker, interval, start_ts, end_ts)
            VALUES (?, ?, ?, ?)
            """,
            (ticker, interval, merged_start, merged_end),
        )

        
    def _format_like_yf(self, data: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
        """
        Convert long-form (ticker, ts, open, high, ...) into a DataFrame
        shaped like yfinance.download:
        - index: DatetimeIndex named 'Date'
        - single ticker: columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
        - multiple tickers: MultiIndex columns (Ticker, Price) with Price in that same list.
        """
        if data.empty:
            # Match yfinance: return empty DataFrame with proper index name
            out = pd.DataFrame()
            out.index.name = "Date"
            return out

        # Ensure ts is datetime and sorted
        data = data.copy()
        data["ts"] = pd.to_datetime(data["ts"])
        data = data.sort_values(["ts", "ticker"])

        # Normalize column names to yfinance spelling
        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
        value_cols = list(rename_map.keys())

        if len(tickers) == 1:
            # Single-ticker case: simple columns, no MultiIndex
            t = tickers[0]
            df = (
                data.loc[data["ticker"] == t, ["ts"] + value_cols]
                .set_index("ts")
                .rename(columns=rename_map)
            )
            df.index.name = "Date"
            return df

        # Multi-ticker case: build MultiIndex columns (Ticker, Price)
        # Long -> wide, then swap levels.
        df = (
            data[["ts", "ticker"] + value_cols]
            .set_index(["ts", "ticker"])
            [value_cols]
            .unstack("ticker")  # columns: (value, ticker)
        )

        # Now columns are (value, ticker); we want (ticker, Price)
        df.columns = df.columns.swaplevel(0, 1)  # (ticker, value)
        df = df.sort_index(axis=1, level=0)

        # Rename 'value' level to proper yfinance-style price labels
        new_cols = []
        for ticker, field in df.columns:
            field_title = rename_map[field]  # e.g. 'open' -> 'Open'
            new_cols.append((ticker, field_title))

        df.columns = pd.MultiIndex.from_tuples(new_cols, names=["Ticker", "Price"])
        df.index.name = "Date"

        return df


def _to_timestamp(value) -> pd.Timestamp:
    return value if isinstance(value, pd.Timestamp) else pd.Timestamp(value)

def _select_ticker_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame | None:
    if df.empty:
        return None

    if not isinstance(df.columns, pd.MultiIndex):
        return df

    for level in range(df.columns.nlevels):
        if ticker in set(df.columns.get_level_values(level)):
            out = df.xs(ticker, axis=1, level=level, drop_level=True)
            if isinstance(out, pd.Series):
                out = out.to_frame()
            return out
    return None

def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(-1)
    return df
