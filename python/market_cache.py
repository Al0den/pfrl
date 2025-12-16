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
                );
            """)

    def cache_summary(self):
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
        tickers = [str(t) for t in tickers]

        query = """
            SELECT ticker, interval, ts, open, high, low, close, adj_close, volume
            FROM prices
            WHERE ticker IN (SELECT * FROM UNNEST(?))
            AND interval = ?
            AND ts >= ?
            AND ts < ?
            ORDER BY ticker, ts
        """
        params = [tickers, interval, start, end]

        with self._read_conn() as conn:
            data = conn.execute(query, params).fetchdf()

        return self._format_like_yf(data, tickers)

    def get_trading_window_bounds(self, n_trading_days, *, end=None, start=None, reference_ticker = "^GSPC"):
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

    def get_trading_window_end(self, n_trading_days, *, end=None, start=None, reference_ticker = "^GSPC"):
        _, window_end = self.get_trading_window_bounds(
            n_trading_days, end=end, start=start, reference_ticker=reference_ticker
        )
        return window_end

    def _trading_days(self, reference_ticker: str, start: pd.Timestamp, end: pd.Timestamp):
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
            group_by="ticker",
            auto_adjust=False,
            threads=True,
        )
        assert df is not None

        downloaded_at = pd.Timestamp.now()

        if isinstance(df.columns, pd.MultiIndex):
            cols0 = df.columns.get_level_values(0)
            cols1 = df.columns.get_level_values(1)

            tick_set = set(tickers)
            if len(tick_set.intersection(set(cols0))) > 0:
                wide = df.copy()
                wide.columns = wide.columns.set_names(["Ticker", "Field"])
                long = (
                    wide.stack(level="Ticker", future_stack=True)
                        .reset_index()
                        .rename(columns={"level_0": "ts"})
                )
            else:
                wide = df.copy()
                wide.columns = wide.columns.swaplevel(0, 1)
                wide.columns = wide.columns.set_names(["Ticker", "Field"])
                long = (
                    wide.stack(level="Ticker", future_stack=True)
                        .reset_index()
                        .rename(columns={"level_0": "ts"})
                )
        else:
            t0 = list(tickers)[0] if isinstance(tickers, (list, tuple)) else str(tickers)
            tmp = df.copy()
            tmp["Ticker"] = t0
            long = tmp.reset_index().rename(columns={tmp.index.name or "index": "ts"})


        long = long.rename(columns={"Date": "ts"})

        want = ["ts", "Ticker", "Open", "High", "Low", "Close", "Adj Close", "Volume"]
        for c in want:
            if c not in long.columns:
                long[c] = pd.NA

        long = long[want].rename(columns={"Ticker": "ticker"})
        long["interval"] = interval
        long["source"] = "yfinance"
        long["downloaded_at"] = downloaded_at

        long["ts"] = pd.to_datetime(long["ts"])
        long["Volume"] = pd.to_numeric(long["Volume"], errors="coerce").astype("Int64")

        long = long.rename(
            columns={
                "Open": "open",
                "High": "high",
                "Low": "low",
                "Close": "close",
                "Adj Close": "adj_close",
                "Volume": "volume",
            }
        )

        long = long.dropna(subset=["ts"])
        if long.empty:
            return

        with self._write_conn() as conn:
            conn.execute("BEGIN TRANSACTION;")
            try:
                conn.register("tmp_prices", long)
                conn.execute("""
                    INSERT OR REPLACE INTO prices
                    SELECT
                        ticker,
                        interval,
                        ts,
                        open,
                        high,
                        low,
                        close,
                        adj_close,
                        volume,
                        source,
                        downloaded_at
                    FROM tmp_prices
                """)

                conn.unregister("tmp_prices")

                inserted_tickers = (
                    conn.execute("SELECT DISTINCT ticker FROM prices WHERE downloaded_at = ?", [downloaded_at])
                        .fetchall()
                )
                inserted_tickers = [r[0] for r in inserted_tickers]
                inserted_ranges = conn.execute("""
                    SELECT ticker, MIN(ts) AS min_ts, MAX(ts) AS max_ts
                    FROM prices
                    WHERE downloaded_at = ?
                    AND interval = ?
                    AND ticker IN ({})
                    GROUP BY ticker
                """.format(",".join("?" for _ in tickers)), [downloaded_at, interval, *list(tickers)]
                ).fetchall()

                for t, min_ts, max_ts in inserted_ranges:
                    if min_ts is None or max_ts is None:
                        continue
                   
                    end_excl = pd.Timestamp(max_ts) + pd.Timedelta(days=1) if interval == "1d" else pd.Timestamp(max_ts)
                    self._merge_coverage_window(conn, t, interval, pd.Timestamp(min_ts), end_excl)

                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise
            
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
        if data.empty:
            out = pd.DataFrame()
            out.index.name = "Date"
            return out

        rename_map = {
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "adj_close": "Adj Close",
            "volume": "Volume",
        }
        ycols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]

        df = data.copy()
        df["ts"] = pd.to_datetime(df["ts"])
        df["ticker"] = df["ticker"].astype(str)

        df = df.sort_values(["ts", "ticker"]).drop_duplicates(subset=["ts", "ticker"], keep="last")

        tickers = [str(t) for t in tickers]

        if len(tickers) == 1:
            t = tickers[0]
            sub = df[df["ticker"] == t].set_index("ts")
            sub = sub.rename(columns=rename_map)
            out = sub.reindex(columns=ycols).sort_index()
            out.index.name = "Date"
            return out

        pieces = []
        for t in tickers:
            sub = df[df["ticker"] == t]
            if sub.empty:
                part = pd.DataFrame(columns=ycols, index=pd.DatetimeIndex([], name="Date"))
            else:
                part = sub.set_index("ts").rename(columns=rename_map)
                part = part.reindex(columns=ycols).sort_index()
                part.index.name = "Date"

            part.columns = pd.MultiIndex.from_product([[t], part.columns], names=["Ticker", "Price"])
            pieces.append(part)

        out = pd.concat(pieces, axis=1).sort_index()
        out.index.name = "Date"
        return out



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
