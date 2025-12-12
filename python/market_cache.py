from __future__ import annotations

from typing import Iterable

import duckdb
import pandas as pd
import yfinance as yf


class MarketCache:
    def __init__(self, db_path : str = "data/yfinance/yfinance_cache.duckdb"):
        self.db_path = db_path
        self.conn = duckdb.connect(self.db_path)
        self._initialize_db()


    def _initialize_db(self):
        self.conn.execute("""
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

        self.conn.execute("""
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
        return self.conn.execute(query).fetchdf()
    
    def get_prices(self, tickers: Iterable[str], start, end, interval="1d"):
        if interval not in {"1d", "1wk", "1mo", "1h", "30m", "15m", "5m", "2m", "1m"}:
            raise ValueError(f"Unsupported interval: {interval}")
        tickers = list(tickers)
        start = _to_timestamp(start)
        end = _to_timestamp(end)

        gaps = []
        missing_tickers = []

        for t in tickers:
            if not self._cache_has_all_data(t, start, end, interval):
                missing_tickers.append(t)
                gaps.extend(self._cache_gaps(t, start, end, interval))

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
        data = self.conn.execute(query, params).fetchdf()

        return self._format_like_yf(data, tickers)

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
            self.conn.executemany("""
                INSERT OR REPLACE INTO prices 
                (ticker, interval, ts, open, high, low, close, adj_close, volume, source, downloaded_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [(
                r['ticker'], r['interval'], r['ts'], r['open'], r['high'], r['low'], r['close'],
                r['adj_close'], r['volume'], r['source'], r['downloaded_at']
            ) for r in records])

            for ticker in sorted({r["ticker"] for r in records}):
                self._merge_coverage_window(ticker, interval, start, end)

    def _cache_has_all_data(self, ticker, start, end, interval = '1d'):
        return len(self._cache_gaps(ticker, start, end, interval)) == 0

    def _cache_gaps(self, ticker, start, end, interval):
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

        rows = self.conn.execute(query, (ticker, interval, start, end)).fetchall()

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

    
    def _merge_coverage_window(self, ticker: str, interval: str, start, end) -> None:
        start = _to_timestamp(start)
        end = _to_timestamp(end)

        merged_start = start
        merged_end = end

        while True:
            rows = self.conn.execute(
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

            self.conn.execute(
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

        self.conn.execute(
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

