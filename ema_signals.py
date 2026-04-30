# Replit and ChatGPT used across all functions and concepts for development, troubleshooting, and efficiency
# =============================================================
# signals.py  —  EMA dual-crossover signal calculation
# =============================================================
# Computes a weekly 10-period vs 26-period EMA crossover signal
# for every ticker, classified into five bands:
#
#   STRONG_BUY  — spread > +1.0%  AND  price > EMA10 by >2%
#   BUY         — spread > +0.5%
#   HOLD        — |spread| ≤ 0.5%  (EMAs converging, no trend)
#   SELL        — spread < -0.5%
#   STRONG_SELL — spread < -1.0%  AND  price < EMA10 by >2%
#
# Key formulas:
#   k        = 2 / (N + 1)           smoothing multiplier
#   EMA_t    = price_t × k + EMA_{t-1} × (1-k)   recursive EMA
#   spread   = (EMA10 - EMA26) / EMA26 × 100      crossover magnitude
#   price_gap= (price  - EMA10) / EMA10 × 100      price vs short EMA
#
# Data source:
#   Primary — Alpha Vantage weekly EMA API (if key configured)
#   Fallback — local pandas .ewm() on yfinance weekly resampled data# =============================================================
# signals.py  —  EMA dual-crossover signal calculation
# =============================================================
# This file is responsible for creating the app's EMA-based trading signal.
# It does not build the dashboard UI directly. Instead, app.py calls
# compute_all_signals(), and the returned signal dictionary is later displayed
# in the Market Overview table and Administration tab.
#
# Main idea:
#   - Calculate or fetch a 10-week EMA and a 26-week EMA.
#   - Compare the two EMAs to determine trend direction.
#   - Compare current price to EMA10 to confirm strong signals.
#   - Return one signal classification per ticker.
#
# Signal categories:
#   STRONG_BUY  — EMA10 is well above EMA26 and price confirms upside momentum.
#   BUY         — EMA10 is moderately above EMA26.
#   HOLD        — EMAs are close together, meaning no clear trend.
#   SELL        — EMA10 is moderately below EMA26.
#   STRONG_SELL — EMA10 is well below EMA26 and price confirms downside momentum.
#
# Key formulas:
#   k         = 2 / (N + 1)
#   EMA_t     = price_t × k + EMA_previous × (1 - k)
#   spread    = (EMA10 - EMA26) / EMA26 × 100
#   price_gap = (price - EMA10) / EMA10 × 100
#
# Data source behavior:
#   - If ALPHA_VANTAGE_KEY exists, the app tries to use Alpha Vantage EMA data.
#   - If not, it computes EMA locally from Yahoo Finance price data.
# =============================================================

# Allows modern Python type hints without forcing annotations to be evaluated immediately.
from __future__ import annotations

# pandas is used for resampling daily price data into weekly prices
# and for calculating exponential moving averages.
import pandas as pd

# Import signal thresholds, display labels, and colors from the central config file.
# This keeps signal settings in one place instead of hardcoding them here.
from config import (
    ALPHA_VANTAGE_KEY,
    STRONG_BUY_SPREAD, BUY_SPREAD, SELL_SPREAD, STRONG_SELL_SPREAD,
    STRONG_PRICE_GAP, SIGNAL_LABELS, SIGNAL_COLORS,
)

# Reuses the Alpha Vantage EMA-fetching function from data.py.
# This avoids duplicating API request code in two different files.
from data import fetch_ema_alphavantage


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _weekly_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Resample daily OHLCV data to weekly closes, then compute EMA.

    This is an internal helper function. The underscore means it is
    intended to be used only inside this file.

    Args:
        df:
            A price DataFrame containing at least a "Close" column.
        window:
            The EMA period, such as 10 or 26.

    Returns:
        A pandas Series containing weekly EMA values.
    """

    # Resample daily close prices into weekly closes.
    # "W-FRI" means each weekly bar ends on Friday, which matches normal market weeks.
    # .last() takes the final available close in each weekly period.
    # .dropna() removes weeks where no close price exists.
    weekly = df["Close"].resample("W-FRI").last().dropna()

    # .ewm(span=window, adjust=False).mean() calculates the exponential moving average.
    # adjust=False uses the recursive EMA formula usually taught in finance.
    return weekly.ewm(span=window, adjust=False).mean()


def _classify(spread: float, price_gap: float | None) -> tuple:
    """
    Convert EMA spread and price confirmation into a signal key and explanation.

    Args:
        spread:
            Percentage distance between EMA10 and EMA26.
            Positive spread means EMA10 is above EMA26.
            Negative spread means EMA10 is below EMA26.

        price_gap:
            Percentage distance between current price and EMA10.
            Positive price_gap means price is above EMA10.
            Negative price_gap means price is below EMA10.

    Returns:
        A tuple:
            (signal_key, reason)

        Example:
            ("BUY", "EMA10 0.75% above EMA26 — golden cross")
    """

    # If price_gap is missing, use 0.0 so the comparison math still works.
    # This prevents None from causing a crash in threshold comparisons.
    pg = price_gap if price_gap is not None else 0.0

    # Strong Buy requires both:
    #   1. EMA10 meaningfully above EMA26
    #   2. Current price meaningfully above EMA10
    # This helps confirm the trend is not just a weak crossover.
    if spread >= STRONG_BUY_SPREAD and pg >= STRONG_PRICE_GAP:
        return "STRONG_BUY", (
            f"spread +{spread:.2f}% > {STRONG_BUY_SPREAD}% ✓  "
            f"and price +{pg:.2f}% above EMA10 > {STRONG_PRICE_GAP}% ✓"
        )

    # Buy means EMA10 is above EMA26 by enough to show a bullish crossover.
    elif spread >= BUY_SPREAD:
        return "BUY", f"EMA10 {spread:.2f}% above EMA26 — golden cross"

    # Strong Sell requires both:
    #   1. EMA10 meaningfully below EMA26
    #   2. Current price meaningfully below EMA10
    # This confirms stronger downside momentum.
    elif spread <= STRONG_SELL_SPREAD and pg <= -STRONG_PRICE_GAP:
        return "STRONG_SELL", (
            f"spread {spread:.2f}% < {STRONG_SELL_SPREAD}% ✓  "
            f"and price {pg:.2f}% below EMA10 < -{STRONG_PRICE_GAP}% ✓"
        )

    # Sell means EMA10 is below EMA26 by enough to show a bearish crossover.
    elif spread <= SELL_SPREAD:
        return "SELL", f"EMA10 {abs(spread):.2f}% below EMA26 — death cross"

    # If none of the buy/sell thresholds are met, the stock is treated as neutral.
    else:
        return "HOLD", f"|spread| = {abs(spread):.2f}% — EMAs converging, no clear momentum"


def _unknown(reason: str) -> dict:
    """
    Return a standard placeholder signal when calculation cannot be completed.

    This prevents the app from crashing when a ticker has missing data,
    too little history, failed API calls, or other issues.

    Args:
        reason:
            Human-readable explanation of why the signal is unknown.

    Returns:
        A dictionary with the same keys as a normal signal result,
        but with values set to None, UNKNOWN, or empty lists.
    """

    # Keeping the dictionary shape consistent is important because the UI expects
    # these keys to exist even when the signal cannot be calculated.
    return {
        "ema10": None,
        "ema26": None,
        "spread": None,
        "price": None,
        "price_gap": None,
        "signal": "UNKNOWN",
        "label": SIGNAL_LABELS["UNKNOWN"],
        "color": SIGNAL_COLORS["UNKNOWN"],
        "reason": reason,
        "ema10_history": [],
        "ema26_history": [],
    }


# ─────────────────────────────────────────────────────────────
# Public function — compute signals for all tickers
# ─────────────────────────────────────────────────────────────

def compute_all_signals(price_data: dict) -> dict:
    """
    Compute EMA10 / EMA26 dual-crossover signals for every ticker.

    This is the main function from this file that app.py uses.

    Args:
        price_data:
            A dictionary created by data.fetch_price_data().
            Expected shape:
                {
                    "AAPL": DataFrame,
                    "GOOGL": DataFrame,
                    ...
                }

    Returns:
        A dictionary with one signal result per ticker.

        Example:
            {
                "AAPL": {
                    "ema10": 180.25,
                    "ema26": 175.10,
                    "spread": 2.94,
                    "price": 185.00,
                    "price_gap": 2.63,
                    "signal": "STRONG_BUY",
                    "label": "⬆ Strong Buy",
                    "color": "#10b981",
                    "reason": "...",
                    "ema10_history": [...],
                    "ema26_history": [...],
                }
            }
    """

    # If an Alpha Vantage API key exists, use Alpha Vantage as the preferred EMA source.
    # Otherwise, calculate EMA locally using the yfinance price data already downloaded.
    use_av = bool(ALPHA_VANTAGE_KEY)

    # This dictionary will store the final signal result for every ticker.
    results = {}

    # Loop through each ticker and its corresponding price DataFrame.
    for ticker, df in price_data.items():

        # If no usable data exists, store an UNKNOWN result and move to the next ticker.
        if df is None or df.empty:
            results[ticker] = _unknown("No price data available")
            continue

        try:
            # These variables will hold the latest EMA values.
            # They start as None so the code can tell whether API data succeeded.
            ema10_val = ema26_val = None

            # These lists hold recent EMA history for display in the Administration tab.
            ema10_hist = ema26_hist = []

            # ── Path A: Alpha Vantage API ──────────────────────────
            # If an API key is configured, try getting EMA10 and EMA26 from Alpha Vantage.
            if use_av:
                av10 = fetch_ema_alphavantage(ticker, 10)
                av26 = fetch_ema_alphavantage(ticker, 26)

                # Only use the API path if both EMA series returned data.
                if av10 and av26:

                    # Alpha Vantage data is sorted newest-first in data.py,
                    # so index 0 is the latest EMA value.
                    ema10_val = av10[0]["value"]
                    ema26_val = av26[0]["value"]

                    # Keep the 8 newest EMA values for the Administration tab.
                    ema10_hist = [(d["date"], d["value"]) for d in av10[:8]]
                    ema26_hist = [(d["date"], d["value"]) for d in av26[:8]]

            # ── Path B: local pandas EWM on yfinance data ──────────
            # If Alpha Vantage is unavailable or returned no valid EMA data,
            # calculate weekly EMAs locally using the price history DataFrame.
            if ema10_val is None:

                # Calculate 10-week and 26-week EMAs from weekly resampled closes.
                s10 = _weekly_ema(df, 10)
                s26 = _weekly_ema(df, 26)

                # EMA26 needs enough weekly data to stabilize.
                # If fewer than 26 weekly bars exist, the signal is marked unknown.
                if len(s26) < 26:
                    results[ticker] = _unknown(
                        "Insufficient history for EMA26 (need ≥26 weekly bars — try period ≥ 1y)"
                    )
                    continue

                # Latest EMA values are the final entries in the Series.
                ema10_val = float(s10.iloc[-1])
                ema26_val = float(s26.iloc[-1])

                # Create recent EMA history for display.
                # dates[-8:][::-1] takes the last 8 dates and reverses them newest-first.
                dates = s10.index.strftime("%Y-%m-%d").tolist()
                ema10_hist = list(zip(dates[-8:][::-1], s10.iloc[-8:][::-1].tolist()))
                ema26_hist = list(zip(dates[-8:][::-1], s26.iloc[-8:][::-1].tolist()))

            # ── Classify the signal ────────────────────────────────
            # Current price is the latest close from the price DataFrame.
            price_val = float(df["Close"].iloc[-1])

            # Spread measures how far EMA10 is above or below EMA26.
            # Positive spread is bullish; negative spread is bearish.
            spread = (ema10_val - ema26_val) / ema26_val * 100 if ema26_val else 0.0

            # price_gap checks whether current price confirms the EMA10 trend.
            # This helps separate regular Buy/Sell from Strong Buy/Strong Sell.
            price_gap = (price_val - ema10_val) / ema10_val * 100 if ema10_val else None

            # Convert spread and price confirmation into a signal key and explanation.
            sig_key, reason = _classify(spread, price_gap)

            # Store the completed result using the exact structure app.py and tabs.py expect.
            results[ticker] = {
                "ema10": ema10_val,
                "ema26": ema26_val,
                "spread": spread,
                "price": price_val,
                "price_gap": price_gap,
                "signal": sig_key,
                "label": SIGNAL_LABELS[sig_key],
                "color": SIGNAL_COLORS[sig_key],
                "reason": reason,
                "ema10_history": ema10_hist,
                "ema26_history": ema26_hist,
            }

        # If anything unexpected happens for one ticker, store UNKNOWN for that ticker
        # instead of crashing the entire dashboard.
        except Exception as exc:
            results[ticker] = _unknown(str(exc))

    # Return all ticker signals back to app.py.
    return results
# =============================================================

from __future__ import annotations
import pandas as pd
from config import (
    ALPHA_VANTAGE_KEY,
    STRONG_BUY_SPREAD, BUY_SPREAD, SELL_SPREAD, STRONG_SELL_SPREAD,
    STRONG_PRICE_GAP, SIGNAL_LABELS, SIGNAL_COLORS,
)
from data import fetch_ema_alphavantage


# ─────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────

def _weekly_ema(df: pd.DataFrame, window: int) -> pd.Series:
    """
    Resample daily OHLCV data to weekly closes then compute EMA.

    Why resample to weekly?
    The crossover strategy uses 10-week and 26-week EMAs (i.e. the
    EMA is computed on weekly bars, not daily bars).  Daily bars with
    span=10 would give a ~2-week EMA — a very different signal.

    pandas .ewm(span=N, adjust=False):
      - span=N → k = 2/(N+1)
      - adjust=False → applies the standard recursive formula
        EMA_t = close_t × k + EMA_{t-1} × (1-k)
      - W-FRI → resample to weekly bars closing on Friday
    """
    weekly = df["Close"].resample("W-FRI").last().dropna()
    return weekly.ewm(span=window, adjust=False).mean()


def _classify(spread: float, price_gap: float | None) -> tuple:
    """
    Map (spread, price_gap) to a signal key and human-readable reason.

    The 0.5% / 1.0% thresholds filter out noise in flat/ranging markets
    where the EMAs cross back and forth without a genuine trend.

    The "Strong" qualifiers require the spot price to also confirm the
    direction — this reduces false signals from lagging EMAs.
    """
    pg = price_gap if price_gap is not None else 0.0

    if spread >= STRONG_BUY_SPREAD and pg >= STRONG_PRICE_GAP:
        return "STRONG_BUY", (
            f"spread +{spread:.2f}% > {STRONG_BUY_SPREAD}% ✓  "
            f"and price +{pg:.2f}% above EMA10 > {STRONG_PRICE_GAP}% ✓"
        )
    elif spread >= BUY_SPREAD:
        return "BUY", f"EMA10 {spread:.2f}% above EMA26 — golden cross"

    elif spread <= STRONG_SELL_SPREAD and pg <= -STRONG_PRICE_GAP:
        return "STRONG_SELL", (
            f"spread {spread:.2f}% < {STRONG_SELL_SPREAD}% ✓  "
            f"and price {pg:.2f}% below EMA10 < -{STRONG_PRICE_GAP}% ✓"
        )
    elif spread <= SELL_SPREAD:
        return "SELL", f"EMA10 {abs(spread):.2f}% below EMA26 — death cross"

    else:
        return "HOLD", f"|spread| = {abs(spread):.2f}% — EMAs converging, no clear momentum"


def _unknown(reason: str) -> dict:
    """Return a uniform 'unknown signal' result dict."""
    return {
        "ema10": None, "ema26": None, "spread": None,
        "price": None, "price_gap": None,
        "signal":        "UNKNOWN",
        "label":         SIGNAL_LABELS["UNKNOWN"],
        "color":         SIGNAL_COLORS["UNKNOWN"],
        "reason":        reason,
        "ema10_history": [],
        "ema26_history": [],
    }


# ─────────────────────────────────────────────────────────────
# Public function — compute signals for all tickers
# ─────────────────────────────────────────────────────────────

def compute_all_signals(price_data: dict) -> dict:
    """
    Compute EMA10 / EMA26 dual-crossover signals for every ticker.

    Signal source selection:
      If ALPHA_VANTAGE_KEY is set → use Alpha Vantage weekly EMA API.
        Pros: full multi-year history → more accurate EMA convergence.
        Cons: 25 req/day limit on free tier.
      Otherwise → compute locally from yfinance weekly bars.
        Pros: no API key required, unlimited calls.
        Cons: accuracy depends on the selected period — needs at least
        6–7 months of data for EMA26 to properly converge.

    Args:
        price_data: { ticker: DataFrame } as returned by data.fetch_price_data()

    Returns:
        { ticker: {
            ema10, ema26, spread, price, price_gap,
            signal, label, color, reason,
            ema10_history,   # list of (date_str, value) newest-first
            ema26_history,
          }
        }
    """
    use_av = bool(ALPHA_VANTAGE_KEY)   # use Alpha Vantage if key is configured
    results = {}

    for ticker, df in price_data.items():
        if df is None or df.empty:
            results[ticker] = _unknown("No price data available")
            continue

        try:
            ema10_val = ema26_val = None
            ema10_hist = ema26_hist = []

            # ── Path A: Alpha Vantage API ──────────────────────────
            if use_av:
                av10 = fetch_ema_alphavantage(ticker, 10)
                av26 = fetch_ema_alphavantage(ticker, 26)
                if av10 and av26:
                    ema10_val  = av10[0]["value"]
                    ema26_val  = av26[0]["value"]
                    # Build 8-row history lists (newest first) for the Admin tab
                    ema10_hist = [(d["date"], d["value"]) for d in av10[:8]]
                    ema26_hist = [(d["date"], d["value"]) for d in av26[:8]]

            # ── Path B: local pandas EWM on yfinance data ──────────
            if ema10_val is None:
                s10 = _weekly_ema(df, 10)
                s26 = _weekly_ema(df, 26)
                if len(s26) < 26:
                    # Not enough weekly bars for EMA26 to converge
                    results[ticker] = _unknown(
                        "Insufficient history for EMA26 (need ≥26 weekly bars — try period ≥ 1y)"
                    )
                    continue
                ema10_val = float(s10.iloc[-1])
                ema26_val = float(s26.iloc[-1])
                # Collect last 8 data points, newest first
                dates = s10.index.strftime("%Y-%m-%d").tolist()
                ema10_hist = list(zip(dates[-8:][::-1], s10.iloc[-8:][::-1].tolist()))
                ema26_hist = list(zip(dates[-8:][::-1], s26.iloc[-8:][::-1].tolist()))

            # ── Classify the signal ────────────────────────────────
            price_val = float(df["Close"].iloc[-1])
            spread    = (ema10_val - ema26_val) / ema26_val * 100  if ema26_val else 0.0
            price_gap = (price_val - ema10_val) / ema10_val * 100  if ema10_val else None

            sig_key, reason = _classify(spread, price_gap)

            results[ticker] = {
                "ema10":         ema10_val,
                "ema26":         ema26_val,
                "spread":        spread,
                "price":         price_val,
                "price_gap":     price_gap,
                "signal":        sig_key,
                "label":         SIGNAL_LABELS[sig_key],
                "color":         SIGNAL_COLORS[sig_key],
                "reason":        reason,
                "ema10_history": ema10_hist,
                "ema26_history": ema26_hist,
            }

        except Exception as exc:
            results[ticker] = _unknown(str(exc))

    return results
