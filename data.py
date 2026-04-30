# Replit and ChatGPT used across all functions and concepts for development, troubleshooting, and efficiency
# =============================================================
# data.py  —  Data fetching (Yahoo Finance, Alpha Vantage, FMP)
# =============================================================
# This file is the main data layer for the dashboard.
#
# It is responsible for:
#   1. Fetching stock price history from Yahoo Finance.
#   2. Cleaning/normalizing the price data so the rest of the app
#      receives consistent columns.
#   3. Adding technical indicators such as SMA, EMA, RSI, and volatility.
#   4. Fetching company fundamentals from Yahoo Finance.
#   5. Building the main Market Overview table used in app.py.
#   6. Optionally fetching EMA data from Alpha Vantage.
#   7. Optionally fetching exchange/company profile data from FMP.
#
# There are no classes in this file.
# The code is organized as helper functions and cached data functions.
# =============================================================

from __future__ import annotations

# Optional and Any are used for type hints.
# Optional[float] means the function may return a float or None.
from typing import Optional, Any

# numpy is used for numerical checks and volatility calculations.
import numpy as np

# pandas is used for DataFrame operations, rolling averages, and time series data.
import pandas as pd

# requests is used for API calls to Alpha Vantage and FMP.
import requests

# streamlit is used here mainly for caching and displaying warning messages.
import streamlit as st

# yfinance is used to fetch stock price history and fundamentals.
import yfinance as yf

# Import API keys and endpoint URLs from config.py.
# This keeps sensitive/configurable values out of the data logic.
from config import (
    ALPHA_VANTAGE_KEY,
    FMP_KEY,
    AV_BASE_URL,
    FMP_BASE_URL,
    EXCHANGE_COORDS,
    EXCHANGE_COLOR_MAP,
)
import sqlite3
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _safe_float(value: Any) -> Optional[float]:
    """
    Convert a value to float when possible; otherwise return None.

    This prevents bad API values like None, NaN, infinity, or text
    from crashing later calculations.
    """

    try:
        # None is not usable for calculations.
        if value is None:
            return None

        # Attempt to convert the input into a float.
        value = float(value)

        # Reject NaN and infinite values because they break display/calculation logic.
        if np.isnan(value) or np.isinf(value):
            return None

        return value

    # If conversion fails, return None instead of raising an error.
    except (TypeError, ValueError):
        return None


def _format_market_cap(value: Any) -> str:
    """
    Format market cap as T/B/M.

    Example:
        2500000000000 -> "$2.50T"
        340000000000  -> "$340.00B"
    """

    # Safely convert the value first so bad input does not crash formatting.
    v = _safe_float(value)

    if v is None:
        return "—"

    # Trillion-dollar formatting.
    if v >= 1e12:
        return f"${v / 1e12:.2f}T"

    # Billion-dollar formatting.
    if v >= 1e9:
        return f"${v / 1e9:.2f}B"

    # Million-dollar formatting.
    if v >= 1e6:
        return f"${v / 1e6:.1f}M"

    # Fallback for smaller values.
    return f"${v:,.0f}"


def _dividend_yield_pct(value: Any) -> Optional[float]:
    """
    Convert yfinance dividendYield to percent.

    yfinance usually returns decimal form, like 0.004 = 0.4%.
    If a source already returns a percent-like value greater than 1,
    this keeps it as-is.
    """

    # Safely convert to float.
    v = _safe_float(value)

    # Missing or zero dividend yield is treated as unavailable.
    if v is None or v == 0:
        return None

    # NOTE:
    # This function currently returns the decimal value if abs(v) <= 1.
    # In build_market_overview(), dividend yield is separately converted
    # into a percentage for display.
    if abs(v) <= 1:
        return v

    return v


def _period_return(close: pd.Series, bars: int) -> Optional[float]:
    """
    Return percent change over N bars.

    Args:
        close:
            A pandas Series of closing prices.
        bars:
            Number of periods to look back.

    Returns:
        Percent return over that period, or None if not enough data exists.
    """

    # Remove missing closing prices.
    close = close.dropna()

    # Need at least bars + 1 values to compare start and end.
    if len(close) < bars + 1:
        return None

    # Get the starting and ending prices.
    start = _safe_float(close.iloc[-(bars + 1)])
    end = _safe_float(close.iloc[-1])

    # Avoid bad values or division by zero.
    if start is None or end is None or start == 0:
        return None

    # Percent return formula.
    return (end / start - 1) * 100


def _normalize_price_frame(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Normalize yfinance output so app.py always receives the same columns:
    Open, High, Low, Close, Volume.

    yfinance sometimes returns data in slightly different shapes.
    This function protects the rest of the app from those inconsistencies.
    """

    # If no data was returned, return an empty DataFrame.
    if df is None or df.empty:
        return pd.DataFrame()

    # Work on a copy so the original data is not modified unexpectedly.
    df = df.copy()

    # yfinance sometimes returns MultiIndex columns.
    # Example: ('Close', 'AAPL') instead of just 'Close'.
    # This flattens MultiIndex columns into simple names.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            col[0] if isinstance(col, tuple) else col
            for col in df.columns
        ]

    # Remove duplicate columns if yfinance returns them.
    df = df.loc[:, ~df.columns.duplicated()]

    # These columns are required for the dashboard and indicator calculations.
    required_cols = ["Open", "High", "Low", "Close", "Volume"]

    # If any required column is missing, return empty data.
    for col in required_cols:
        if col not in df.columns:
            return pd.DataFrame()

    # Keep only the required columns in a consistent order.
    df = df[required_cols]

    # Convert all required columns to numeric values.
    # Invalid values become NaN instead of throwing an error.
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Closing price is essential; rows without Close cannot be used.
    df = df.dropna(subset=["Close"])

    if df.empty:
        return pd.DataFrame()

    # Missing volume is less serious, so fill it with 0.
    df["Volume"] = df["Volume"].fillna(0)

    # Ensure the index is datetime so resampling and charting work correctly.
    df.index = pd.to_datetime(df.index)

    # Remove timezone info so resampling and display are consistent.
    try:
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
    except AttributeError:
        pass

    return df


# ─────────────────────────────────────────────────────────────
# Price history  (Yahoo Finance)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=300, show_spinner=False)
def fetch_price_data(tickers: tuple, period: str, interval: str) -> dict:
    """
    Download OHLCV bars for every ticker in the watchlist.

    Returns:
        {
            "AAPL": DataFrame,
            "GOOGL": DataFrame,
            ...
        }

    Each DataFrame includes:
        Open, High, Low, Close, Volume,
        SMA20, SMA50, EMA20, RSI14, Vol30d

    Streamlit cache:
        ttl=300 means this function's result is cached for 300 seconds.
        This prevents repeated API calls every time the page reruns.
    """

    # Final output dictionary:
    # key = ticker symbol
    # value = cleaned DataFrame with indicators
    result = {}

    # Loop through the user's ticker list.
    for raw_ticker in tickers:

        # Clean each ticker by converting it to uppercase and removing spaces.
        ticker = str(raw_ticker).strip().upper()

        # Skip empty ticker entries.
        if not ticker:
            continue

        # Start with an empty DataFrame in case the first download fails.
        df = pd.DataFrame()

        # Primary method: yf.download
        try:
            downloaded = yf.download(
                ticker,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )

            # Normalize the yfinance output so the app receives predictable columns.
            df = _normalize_price_frame(downloaded, ticker)

        except Exception as exc:
            # Show a warning but do not crash the app.
            st.warning(f"Yahoo Finance download failed for {ticker}: {exc}")

        # Fallback method: yf.Ticker().history
        # If yf.download fails or returns unusable data, try the ticker history method.
        if df.empty:
            try:
                fallback = yf.Ticker(ticker).history(
                    period=period,
                    interval=interval,
                    auto_adjust=True,
                )

                df = _normalize_price_frame(fallback, ticker)

            except Exception as exc:
                st.warning(f"Yahoo Finance fallback failed for {ticker}: {exc}")

        # If both methods fail, warn the user and skip this ticker.
        if df.empty:
            st.warning(f"No valid price data returned for {ticker}.")
            continue

        # Add indicators before storing the result.
        result[ticker] = _add_indicators(df)

    return result


def _add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute technical indicators and append them as extra columns.

    Adds:
        SMA20   — 20-period simple moving average
        SMA50   — 50-period simple moving average
        EMA20   — 20-period exponential moving average
        RSI14   — 14-period relative strength index
        Vol30d  — 30-day rolling annualized volatility
    """

    # Work on a copy so the caller's DataFrame is not modified unexpectedly.
    df = df.copy()

    # Close price is the base for most technical indicators.
    close = df["Close"]

    # Simple moving averages smooth price over fixed windows.
    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()

    # Exponential moving average weights recent prices more heavily.
    df["EMA20"] = close.ewm(span=20, adjust=False).mean()

    # RSI calculation starts by finding day-to-day price changes.
    delta = close.diff()

    # Gains keep only positive changes.
    gain = delta.clip(lower=0)

    # Losses keep only negative changes, then flip them positive.
    loss = -delta.clip(upper=0)

    # Wilder-style smoothing for average gains/losses.
    # com=13 corresponds to a 14-period RSI smoothing style.
    avg_gain = gain.ewm(com=13, adjust=False).mean()
    avg_loss = loss.ewm(com=13, adjust=False).mean()

    # Relative strength = average gain / average loss.
    # replace(0, NaN) prevents division by zero.
    rs = avg_gain / avg_loss.replace(0, float("nan"))

    # RSI formula.
    # Missing values are filled with 50, a neutral RSI value.
    df["RSI14"] = (100 - 100 / (1 + rs)).fillna(50)

    # Log returns are used for volatility.
    log_ret = np.log(close / close.shift(1))

    # Annualized volatility:
    # rolling 30-day standard deviation × sqrt(252 trading days) × 100.
    df["Vol30d"] = log_ret.rolling(30).std() * np.sqrt(252) * 100

    return df


# ─────────────────────────────────────────────────────────────
# Fundamental data  (Yahoo Finance)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_fundamentals(ticker: str) -> dict:
    """
    Fetch company fundamentals from Yahoo Finance.

    Returns a dictionary with fields such as:
        marketCap, trailingPE, dividendYield, beta,
        longName, sector, industry, country, exchange, etc.

    Cached for 1 hour because fundamentals do not change as often as price data.
    """

    # Clean ticker input.
    ticker = str(ticker).strip().upper()

    if not ticker:
        return {}

    try:
        # yfinance .info returns a large dictionary of company metadata/fundamentals.
        info = yf.Ticker(ticker).info

        # Only return the result if it is actually a dictionary.
        return info if isinstance(info, dict) else {}

    except Exception as exc:
        # Warn and return an empty dictionary instead of crashing.
        st.warning(f"Yahoo Finance fundamentals failed for {ticker}: {exc}")
        return {}


@st.cache_data(ttl=300, show_spinner=False)

def fetch_all_fundamentals(tickers: tuple) -> dict:
    """
    Fetch fundamentals for all tickers.

    Returns:
        {
            "AAPL": {...},
            "GOOGL": {...},
            ...
        }
    """

    # Dictionary comprehension builds a fundamentals dictionary for every valid ticker.
    return {
        str(t).strip().upper(): fetch_fundamentals(str(t).strip().upper())
        for t in tickers
        if str(t).strip()
    }

# ─────────────────────────────────────────────────────────────
# Exchange map database (SQLite)
# ─────────────────────────────────────────────────────────────

EXCHANGE_DB_PATH = Path("exchange_map.db")


def build_exchange_map_dataframe() -> pd.DataFrame:
    """
    Build the exchange dataset from config.EXCHANGE_COORDS.

    Returns columns:
        Exchange Code
        Exchange Name
        Latitude
        Longitude
        Color
    """
    rows = []

    for code, coords in EXCHANGE_COORDS.items():
        lat, lon, name = coords
        rows.append({
            "Exchange Code": str(code).upper(),
            "Exchange Name": str(name),
            "Latitude": float(lat),
            "Longitude": float(lon),
            "Color": EXCHANGE_COLOR_MAP.get(str(code).upper(), "#94a3b8"),
        })

    df = pd.DataFrame(rows)
    return df.sort_values(["Exchange Name", "Exchange Code"]).reset_index(drop=True)


def save_exchange_map_to_db(exchange_df: pd.DataFrame, db_path: Path = EXCHANGE_DB_PATH) -> None:
    """
    Save the exchange DataFrame into a SQLite database.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        exchange_df.to_sql(
            "exchange_map",
            conn,
            if_exists="replace",
            index=False,
        )


def load_exchange_map_from_db(db_path: Path = EXCHANGE_DB_PATH) -> pd.DataFrame:
    """
    Load the exchange dataset from SQLite and return it as a DataFrame.
    """
    if not db_path.exists():
        return pd.DataFrame()

    with sqlite3.connect(db_path) as conn:
        try:
            df = pd.read_sql_query(
                """
                SELECT
                    "Exchange Code",
                    "Exchange Name",
                    "Latitude",
                    "Longitude",
                    "Color"
                FROM exchange_map
                ORDER BY "Exchange Name", "Exchange Code"
                """,
                conn,
            )
            return df
        except Exception:
            return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_exchange_map_dataframe(refresh: bool = False) -> pd.DataFrame:
    """
    Main helper for the exchange map.

    If the SQLite DB does not exist, or refresh=True, rebuild it from
    EXCHANGE_COORDS and save it. Then load it back into a DataFrame.
    """
    if refresh or not EXCHANGE_DB_PATH.exists():
        fresh_df = build_exchange_map_dataframe()
        save_exchange_map_to_db(fresh_df)

    db_df = load_exchange_map_from_db()

    if db_df.empty:
        fresh_df = build_exchange_map_dataframe()
        save_exchange_map_to_db(fresh_df)
        db_df = load_exchange_map_from_db()

    return db_df
# ─────────────────────────────────────────────────────────────
# Recommendation Classification for Deep Dive
# ─────────────────────────────────────────────────────────────

SUGGESTION_LABELS = {
    "STRONG_BUY": "⬆ Strong Buy",
    "BUY": "↑ Buy",
    "HOLD": "→ Hold",
    "SELL": "↓ Sell",
    "STRONG_SELL": "⬇ Strong Sell",
    "UNKNOWN": "—",
}

SUGGESTION_SCORES = {
    "STRONG_BUY": 2,
    "BUY": 1,
    "HOLD": 0,
    "SELL": -1,
    "STRONG_SELL": -2,
    "UNKNOWN": 0,
}


def _suggestion_label(key: str) -> str:
    return SUGGESTION_LABELS.get(key, "—")


def _score_to_recommendation_key(score: float) -> str:
    if score >= 1.25:
        return "STRONG_BUY"
    if score >= 0.35:
        return "BUY"
    if score > -0.35:
        return "HOLD"
    if score > -1.25:
        return "SELL"
    return "STRONG_SELL"


def _safe_last(series):
    try:
        clean = series.dropna()
        if clean.empty:
            return None
        return float(clean.iloc[-1])
    except Exception:
        return None


def _suggest_trend(df: pd.DataFrame) -> tuple[str, str]:
    price = _safe_last(df["Close"]) if "Close" in df.columns else None
    sma20 = _safe_last(df["SMA20"]) if "SMA20" in df.columns else None
    sma50 = _safe_last(df["SMA50"]) if "SMA50" in df.columns else None

    if price is None or sma20 is None or sma50 is None:
        return "UNKNOWN", "Not enough SMA data"

    price_vs_sma20 = ((price / sma20) - 1) * 100

    if price > sma20 > sma50 and price_vs_sma20 >= 2:
        return "STRONG_BUY", "Price above SMA20 and SMA50 with strong upside confirmation"
    if price > sma20 > sma50:
        return "BUY", "Price above SMA20 and SMA50"
    if price < sma20 < sma50 and price_vs_sma20 <= -2:
        return "STRONG_SELL", "Price below SMA20 and SMA50 with strong downside confirmation"
    if price < sma20 < sma50:
        return "SELL", "Price below SMA20 and SMA50"

    return "HOLD", "Mixed moving-average trend"


def _suggest_rsi(df: pd.DataFrame) -> tuple[str, str]:
    rsi = _safe_last(df["RSI14"]) if "RSI14" in df.columns else None

    if rsi is None:
        return "UNKNOWN", "RSI unavailable"
    if rsi <= 25:
        return "STRONG_BUY", f"RSI {rsi:.1f} is deeply oversold"
    if rsi < 30:
        return "BUY", f"RSI {rsi:.1f} is oversold"
    if rsi >= 75:
        return "STRONG_SELL", f"RSI {rsi:.1f} is strongly overbought"
    if rsi > 70:
        return "SELL", f"RSI {rsi:.1f} is overbought"

    return "HOLD", f"RSI {rsi:.1f} is neutral"


def _suggest_bollinger(df: pd.DataFrame) -> tuple[str, str]:
    if "Close" not in df.columns or len(df["Close"].dropna()) < 20:
        return "UNKNOWN", "Not enough data for Bollinger Bands"

    close = df["Close"].dropna()
    price = float(close.iloc[-1])

    sma = close.rolling(20).mean()
    std = close.rolling(20).std()

    upper = float((sma + 2 * std).iloc[-1])
    lower = float((sma - 2 * std).iloc[-1])

    if upper == lower:
        return "HOLD", "Bollinger Band width is too narrow to classify"

    band_position = (price - lower) / (upper - lower)

    if price <= lower or band_position <= 0.05:
        return "STRONG_BUY", "Price is at or below lower Bollinger Band"
    if band_position <= 0.20:
        return "BUY", "Price is near lower Bollinger Band"
    if price >= upper or band_position >= 0.95:
        return "STRONG_SELL", "Price is at or above upper Bollinger Band"
    if band_position >= 0.80:
        return "SELL", "Price is near upper Bollinger Band"

    return "HOLD", "Price is inside normal Bollinger range"


def _suggest_volatility(df: pd.DataFrame) -> tuple[str, str]:
    vol = _safe_last(df["Vol30d"]) if "Vol30d" in df.columns else None

    if "Close" not in df.columns:
        return "UNKNOWN", "Close data unavailable"

    close = df["Close"].dropna()

    if vol is None or len(close) < 22:
        return "UNKNOWN", "Not enough volatility data"

    ret_1m = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100

    if vol >= 65 and ret_1m < -5:
        return "STRONG_SELL", f"High volatility {vol:.1f}% with weak 1M return"
    if vol >= 50 and ret_1m < 0:
        return "SELL", f"High volatility {vol:.1f}% with negative 1M return"
    if vol <= 20 and ret_1m > 10:
        return "STRONG_BUY", f"Very low volatility {vol:.1f}% with strong 1M return"
    if vol <= 25 and ret_1m > 5:
        return "BUY", f"Low volatility {vol:.1f}% with positive 1M return"

    return "HOLD", f"Volatility {vol:.1f}% does not strongly change recommendation"


def _final_recommendation(
    ema_key: str,
    trend_key: str,
    rsi_key: str,
    bollinger_key: str,
    volatility_key: str,
) -> tuple[str, float]:
    weighted_score = (
        1.25 * SUGGESTION_SCORES.get(ema_key, 0)
        + 1.25 * SUGGESTION_SCORES.get(trend_key, 0)
        + 1.00 * SUGGESTION_SCORES.get(rsi_key, 0)
        + 1.00 * SUGGESTION_SCORES.get(bollinger_key, 0)
        + 0.75 * SUGGESTION_SCORES.get(volatility_key, 0)
    )

    total_weight = 1.25 + 1.25 + 1.00 + 1.00 + 0.75
    final_score = weighted_score / total_weight

    return _score_to_recommendation_key(final_score), final_score
# ─────────────────────────────────────────────────────────────
# Market Overview builder
# ─────────────────────────────────────────────────────────────

def build_market_overview(
    price_data: dict,
    fundamentals: dict,
    signal_data: dict,
) -> pd.DataFrame:
    """
    Convert price data, fundamentals, EMA signals, and Deep Dive indicators
    into one clean DataFrame for the Market Overview section.

    This is one of the most important functions in the project because
    app.py uses its output to build the main table and summary visuals.
    """

    rows = []

    # Loop through each ticker's price DataFrame.
    for ticker, df in price_data.items():

        # Skip unusable data.
        if df is None or df.empty or "Close" not in df.columns:
            continue

        close = df["Close"].dropna()

        if close.empty:
            continue

        # Latest close price.
        price = float(close.iloc[-1])

        # Previous close is used for daily change.
        prev = float(close.iloc[-2]) if len(close) >= 2 else price

        # Absolute and percent daily change.
        change_abs = price - prev
        change_pct = (change_abs / prev * 100) if prev else 0

        # Recent period returns.
        ret_5d = ((close.iloc[-1] / close.iloc[-6]) - 1) * 100 if len(close) >= 6 else None
        ret_1m = ((close.iloc[-1] / close.iloc[-22]) - 1) * 100 if len(close) >= 22 else None

        # Pull this ticker's fundamentals and signal data.
        info = fundamentals.get(ticker, {})
        sig = signal_data.get(ticker, {})

        # Internal EMA recommendation key.
        ema_key = sig.get("signal", "UNKNOWN")

        # Calculate additional recommendation categories.
        trend_key, trend_reason = _suggest_trend(df)
        rsi_key, rsi_reason = _suggest_rsi(df)
        bollinger_key, bollinger_reason = _suggest_bollinger(df)
        volatility_key, volatility_reason = _suggest_volatility(df)

        # Combine all categories into one final recommendation.
        final_key, final_score = _final_recommendation(
            ema_key=ema_key,
            trend_key=trend_key,
            rsi_key=rsi_key,
            bollinger_key=bollinger_key,
            volatility_key=volatility_key,
        )

        # Extract common fundamental fields.
        market_cap = info.get("marketCap")
        pe_value = info.get("trailingPE")
        dividend_yield = info.get("dividendYield")

        # yfinance usually returns dividend yield as decimal, ex: 0.005 = 0.5%.
        dividend_yield_pct = None

        if dividend_yield is not None:
            dividend_yield = float(dividend_yield)
            dividend_yield_pct = dividend_yield if abs(dividend_yield) <= 1 else dividend_yield

        # Latest volume value.
        volume = float(df["Volume"].dropna().iloc[-1]) if "Volume" in df.columns and not df["Volume"].dropna().empty else None

        # Append one row of clean dashboard data.
        rows.append({
            "Ticker": ticker,
            "Name": info.get("longName", ticker),

            "Price": price,
            "Change $": change_abs,
            "Return 1D %": change_pct,
            "Return 5D %": ret_5d,
            "Return 1M %": ret_1m,

            "Market Cap": float(market_cap) if market_cap is not None else None,
            "Market Cap Display": _format_market_cap(market_cap),
            "Volume": volume,
            "P/E": float(pe_value) if pe_value is not None else None,
            "Dividend Yield %": dividend_yield_pct,

            "EMA Signal": sig.get("label", "—"),
            "EMA Key": ema_key,

            "Trend Suggestion": _suggestion_label(trend_key),
            "Trend Key": trend_key,
            "Trend Reason": trend_reason,

            "RSI Suggestion": _suggestion_label(rsi_key),
            "RSI Key": rsi_key,
            "RSI Reason": rsi_reason,

            "Bollinger Suggestion": _suggestion_label(bollinger_key),
            "Bollinger Key": bollinger_key,
            "Bollinger Reason": bollinger_reason,

            "Volatility Suggestion": _suggestion_label(volatility_key),
            "Volatility Key": volatility_key,
            "Volatility Reason": volatility_reason,

            "Final Recommendation": _suggestion_label(final_key),
            "Final Key": final_key,
            "Final Score": final_score,
        })

    # Convert collected rows into a DataFrame.
    overview = pd.DataFrame(rows)

    if overview.empty:
        return overview

    # Sort alphabetically by ticker for consistent display.
    return overview.sort_values("Ticker").reset_index(drop=True)


# ─────────────────────────────────────────────────────────────
# EMA data from Alpha Vantage  (optional)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_ema_alphavantage(ticker: str, window: int) -> list:
    """
    Fetch weekly EMA values from Alpha Vantage for a single ticker.

    Returns:
        [
            {"date": "2024-01-05", "value": 183.24},
            ...
        ]

    Returns [] if no API key is configured or request fails.

    Cached for 24 hours because Alpha Vantage has free-tier rate limits.
    """

    ticker = str(ticker).strip().upper()

    # If no API key exists or ticker is blank, skip the API call.
    if not ALPHA_VANTAGE_KEY or not ticker:
        return []

    # Request parameters for Alpha Vantage EMA endpoint.
    params = {
        "function": "EMA",
        "symbol": ticker,
        "interval": "weekly",
        "time_period": window,
        "series_type": "close",
        "apikey": ALPHA_VANTAGE_KEY,
    }

    try:
        # Send GET request to Alpha Vantage.
        resp = requests.get(AV_BASE_URL, params=params, timeout=10)
        resp.raise_for_status()

        # Parse JSON response.
        raw = resp.json()

        # Alpha Vantage stores EMA results under this specific key.
        series = raw.get("Technical Analysis: EMA", {})

        if not series:
            return []

        # Convert raw API dictionary into a clean list of dictionaries.
        # sorted(..., reverse=True) puts newest dates first.
        return [
            {"date": date_str, "value": float(entry["EMA"])}
            for date_str, entry in sorted(series.items(), reverse=True)
            if "EMA" in entry
        ]

    except Exception as exc:
        st.warning(f"Alpha Vantage EMA failed for {ticker}: {exc}")
        return []


# ─────────────────────────────────────────────────────────────
# Exchange / company profile from FMP  (optional)
# ─────────────────────────────────────────────────────────────

@st.cache_data(ttl=86400, show_spinner=False)
def fetch_exchange_fmp(ticker: str) -> dict:
    """
    Fetch company profile from Financial Modeling Prep.

    Returns {} if no FMP key exists or if the request fails.

    This data is mainly used for the Exchanges & Map tab.
    """

    ticker = str(ticker).strip().upper()

    # Skip the API call if no FMP key exists.
    if not FMP_KEY or not ticker:
        return {}

    try:
        # FMP profile endpoint for one ticker.
        url = f"{FMP_BASE_URL}/profile/{ticker}"

        resp = requests.get(
            url,
            params={"apikey": FMP_KEY},
            timeout=10,
        )

        resp.raise_for_status()

        payload = resp.json()

        # FMP usually returns a list with one company profile dictionary.
        if isinstance(payload, list) and payload:
            return payload[0]

        return {}

    except Exception as exc:
        st.warning(f"FMP profile failed for {ticker}: {exc}")
        return {}