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
#   Fallback — local pandas .ewm() on yfinance weekly resampled data
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
