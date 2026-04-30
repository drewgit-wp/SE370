# Replit and ChatGPT used across all functions and concepts for development, troubleshooting, and efficiency
# =============================================================
# tabs.py  —  All 7 tab rendering functions
# =============================================================
# This file controls what appears inside each Streamlit tab.
#
# Important design point:
#   This file does not fetch new data from the internet.
#   app.py already fetches the data and passes it into these functions.
#
# tabs.py focuses on:
#   - layout
#   - charts
#   - tables
#   - user controls inside each tab
#   - explanations and expanders
#
# There are no classes in this file.
# Each function renders one dashboard tab.
# =============================================================

from __future__ import annotations

# pandas is used to build DataFrames for displayed tables.
import pandas as pd

# streamlit renders the dashboard UI.
import streamlit as st

# Shared constants from config.py.
from config import COLORS, SIGNAL_LABELS, SIGNAL_COLORS, ALPHA_VANTAGE_KEY
import charts
# charts.py contains the Plotly figure-building functions.



# ─────────────────────────────────────────────────────────────
# Tab 1 — Overview
# ─────────────────────────────────────────────────────────────

def tab_overview(price_data: dict, tickers: list) -> None:
    """
    Portfolio-wide snapshot.

    Layout:
      Row 1: Full-width normalized performance line chart.
      Row 2: Latest volume comparison bar chart.

    Args:
        price_data:
            Dictionary of ticker -> price DataFrame.
        tickers:
            List of valid tickers.
    """

    st.subheader("Portfolio Performance")

    st.caption(
        "All series rebased to 100 at the start of the selected period. "
        "A value of 115 means +15% since period start."
    )

    # Normalized performance chart across all tickers.
    fig = charts.comparison_chart(price_data, tickers)
    st.plotly_chart(fig, use_container_width=True)

    # Latest session volume comparison.
    st.subheader("Latest Session — Volume")

    vol_values = []

    # Build a list of latest volume values in the same order as tickers.
    for ticker in tickers:
        df = price_data.get(ticker)

        # Use the most recent Volume value if data exists.
        vol_values.append(float(df["Volume"].iloc[-1]) if df is not None and not df.empty else None)

    fig_vol = charts.metric_bar_chart(tickers, vol_values, "Volume (latest session)", ".3s")
    st.plotly_chart(fig_vol, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 2 — Comparison
# ─────────────────────────────────────────────────────────────

def tab_comparison(price_data: dict, fundamentals: dict, tickers: list) -> None:
    """
    Side-by-side metric comparison across all tickers.

    Six metrics shown:
      1. 1-Month Return (%)
      2. YTD Return (%)
      3. Market Cap (B)
      4. Trailing P/E
      5. 30-Day Volatility (%)
      6. 5-Year Beta

    This function uses nested helper functions because those helpers are
    only needed inside this tab.
    """

    # Creates a two-column layout for the first row of charts.
    col1, col2 = st.columns(2)

    def _get(ticker: str, key: str, scale: float = 1.0):
        """
        Extract a value from the fundamentals dictionary.

        Args:
            ticker:
                Ticker symbol.
            key:
                Fundamental metric key, such as "marketCap".
            scale:
                Optional multiplier for unit conversion.

        Example:
            marketCap × 1e-9 converts market cap to billions.
        """

        v = fundamentals.get(ticker, {}).get(key)
        return float(v) * scale if v is not None else None

    def _return(ticker: str, days: int) -> float | None:
        """
        Compute N-day return from price data.

        Uses closing prices:
            return = latest close / close N days ago - 1
        """

        df = price_data.get(ticker)

        # Need at least days + 1 rows to compare current price to past price.
        if df is None or len(df) < days + 1:
            return None

        close = df["Close"].dropna()

        if len(close) < days + 1 or close.iloc[-(days + 1)] == 0:
            return None

        return (close.iloc[-1] / close.iloc[-(days + 1)] - 1) * 100

    def _ytd(ticker: str) -> float | None:
        """
        Compute year-to-date return.

        Finds the first close in the current year and compares it
        to the latest close.
        """

        df = price_data.get(ticker)

        if df is None or df.empty:
            return None

        close = df["Close"].dropna()

        # Keep only rows from the same year as the latest price.
        year_start = close[close.index.year == close.index[-1].year]

        if len(year_start) < 2 or year_start.iloc[0] == 0:
            return None

        return (year_start.iloc[-1] / year_start.iloc[0] - 1) * 100

    # 1-Month Return chart.
    with col1:
        returns_1m = [_return(t, 21) for t in tickers]
        st.plotly_chart(
            charts.metric_bar_chart(tickers, returns_1m, "1-Month Return (%)", ".1f", "%"),
            use_container_width=True,
        )

    # YTD Return chart.
    with col2:
        returns_ytd = [_ytd(t) for t in tickers]
        st.plotly_chart(
            charts.metric_bar_chart(tickers, returns_ytd, "YTD Return (%)", ".1f", "%"),
            use_container_width=True,
        )

    # Second row of charts.
    col3, col4 = st.columns(2)

    # Market Cap chart.
    with col3:
        mktcaps = [_get(t, "marketCap", 1e-9) for t in tickers]
        st.plotly_chart(
            charts.metric_bar_chart(tickers, mktcaps, "Market Cap (Billions $)", ".1f", "B"),
            use_container_width=True,
        )

    # Trailing P/E chart.
    with col4:
        pes = [_get(t, "trailingPE") for t in tickers]
        st.plotly_chart(
            charts.metric_bar_chart(tickers, pes, "Trailing P/E Ratio", ".1f"),
            use_container_width=True,
        )

    # Third row of charts.
    col5, col6 = st.columns(2)

    # 30-Day Volatility chart.
    with col5:
        vols = []

        for ticker in tickers:
            df = price_data.get(ticker)

            # Vol30d is calculated in data.py.
            v = float(df["Vol30d"].iloc[-1]) if df is not None and "Vol30d" in df.columns else None
            vols.append(v)

        st.plotly_chart(
            charts.metric_bar_chart(tickers, vols, "30-Day Volatility (annualised %)", ".1f", "%"),
            use_container_width=True,
        )

    # Beta chart.
    with col6:
        betas = [_get(t, "beta") for t in tickers]
        st.plotly_chart(
            charts.metric_bar_chart(tickers, betas, "5-Year Beta (vs S&P 500)", ".2f"),
            use_container_width=True,
        )


# ─────────────────────────────────────────────────────────────
# Tab 3 — Deep Dive
# ─────────────────────────────────────────────────────────────

def tab_deep_dive(price_data: dict, tickers: list) -> None:
    """
    Technical analysis for one focused ticker.

    The user picks the ticker from a selectbox, then sees:
      1. Price chart with SMA20 / SMA50.
      2. Bollinger Bands.
      3. RSI-14.
      4. 30-day rolling annualized volatility.
    """

    # User selects one ticker for closer analysis.
    focus = st.selectbox("Focus ticker", tickers, key="deep_dive_select")

    # Get that ticker's price DataFrame.
    df = price_data.get(focus)

    # Assign the selected ticker a consistent chart color.
    color = COLORS[tickers.index(focus) % len(COLORS)]

    if df is None or df.empty:
        st.warning(f"No data available for {focus}.")
        return

    # Price + SMA overlay.
    st.subheader(f"{focus} — Price & Moving Averages")
    st.caption("SMA20 (dashed amber) and SMA50 (dotted violet) shown as trend references.")
    st.plotly_chart(charts.price_chart(df, focus, color), use_container_width=True)

    # Bollinger Bands.
    st.subheader("Bollinger Bands  (SMA20, ±2σ)")
    st.caption(
        "Upper/lower bands = SMA20 ± 2 standard deviations of the last 20 closes. "
        "Wider bands = higher volatility. A squeeze (narrow bands) often precedes a breakout."
    )
    st.plotly_chart(charts.bollinger_chart(df, focus, color), use_container_width=True)

    # RSI and volatility are displayed side by side.
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("RSI (14)")
        st.caption(">70 overbought · <30 oversold · 50 neutral")
        st.plotly_chart(charts.rsi_chart(df, focus, color), use_container_width=True)

    with col2:
        st.subheader("30-Day Volatility (annualised %)")
        st.caption("Rolling std(log-returns, 30) × √252 × 100")
        st.plotly_chart(charts.volatility_chart(df, focus, color), use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Tab 4 — Fundamentals
# ─────────────────────────────────────────────────────────────

def tab_fundamentals(fundamentals: dict, tickers: list) -> None:
    """
    Company fundamentals in a 3-column card layout.

    Each card shows:
      Valuation
      Profitability & Growth
      Trading
      Business Summary
    """

    def fmt_pct(v) -> str:
        """
        Format a decimal fraction as a percentage string.

        Example:
            0.1525 -> "15.25%"
        """

        return f"{float(v)*100:.2f}%" if v is not None else "—"

    def fmt_num(v, decimals=2) -> str:
        """
        Format regular numbers with commas and decimal places.
        """

        return f"{float(v):,.{decimals}f}" if v is not None else "—"

    def fmt_large(v) -> str:
        """
        Format large numbers with T/B/M suffixes.
        """

        if v is None:
            return "—"

        v = float(v)

        if v >= 1e12:
            return f"${v/1e12:.2f}T"

        if v >= 1e9:
            return f"${v/1e9:.2f}B"

        if v >= 1e6:
            return f"${v/1e6:.2f}M"

        return f"${v:,.0f}"

    # Creates up to 3 columns for company cards.
    cols = st.columns(min(len(tickers), 3))

    for idx, ticker in enumerate(tickers):
        info = fundamentals.get(ticker, {})

        # idx % 3 cycles cards across the 3 columns.
        col = cols[idx % 3]

        with col:
            color = COLORS[idx % len(COLORS)]

            # Uses HTML for a styled card header with a colored border.
            st.markdown(
                f"<div style='border-left: 3px solid {color}; padding-left: 10px;'>"
                f"<h4 style='margin:0; color:{color};'>{ticker}</h4>"
                f"<small style='color:#64748b;'>"
                f"{info.get('longName', '')} · {info.get('sector','')}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Valuation block.
            st.markdown("**Valuation**")
            st.write({
                "Trailing P/E": fmt_num(info.get("trailingPE")),
                "Forward P/E": fmt_num(info.get("forwardPE")),
                "Price/Book": fmt_num(info.get("priceToBook")),
                "EV/EBITDA": fmt_num(info.get("enterpriseToEbitda")),
                "Market Cap": fmt_large(info.get("marketCap")),
            })

            # Profitability block.
            st.markdown("**Profitability & Growth**")
            st.write({
                "Profit Margin": fmt_pct(info.get("profitMargins")),
                "ROE": fmt_pct(info.get("returnOnEquity")),
                "ROA": fmt_pct(info.get("returnOnAssets")),
                "Revenue Growth": fmt_pct(info.get("revenueGrowth")),
                "Trailing EPS": fmt_num(info.get("trailingEps")),
            })

            # Trading / market data block.
            st.markdown("**Trading**")

            lo = info.get("fiftyTwoWeekLow")
            hi = info.get("fiftyTwoWeekHigh")

            # Builds 52-week range only if both values are available.
            rng = f"${fmt_num(lo)} – ${fmt_num(hi)}" if lo and hi else "—"

            st.write({
                "52-Week Range": rng,
                "Beta": fmt_num(info.get("beta")),
                "Dividend Yield": fmt_pct(info.get("dividendYield")),
                "Analyst Target": fmt_num(info.get("targetMeanPrice")),
            })

            # Business summary is placed in an expander so it does not overcrowd the tab.
            summary = info.get("longBusinessSummary", "")

            if summary:
                with st.expander("Business Summary"):
                    # Truncate long business summaries to 600 characters.
                    st.write(summary[:600] + ("…" if len(summary) > 600 else ""))

            st.divider()


# ─────────────────────────────────────────────────────────────
# Tab 5 — Raw Data
# ─────────────────────────────────────────────────────────────

def tab_raw_data(price_data: dict, tickers: list) -> None:
    """
    Paginated OHLCV table with technical indicator columns.

    The user picks a ticker and a row limit.
    The DataFrame is displayed with newest rows first.
    The table can also be downloaded as CSV.
    """

    # User chooses which ticker's raw data to inspect.
    focus = st.selectbox("Ticker", tickers, key="raw_data_select")

    df_raw = price_data.get(focus)

    if df_raw is None or df_raw.empty:
        st.warning(f"No data for {focus}.")
        return

    # Select only columns that actually exist in the DataFrame.
    display_cols = [
        c
        for c in [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume",
            "SMA20",
            "SMA50",
            "EMA20",
            "RSI14",
            "Vol30d",
        ]
        if c in df_raw.columns
    ]

    # Copy display data so formatting changes do not affect source data.
    df_show = df_raw[display_cols].copy()

    # Convert datetime index to readable date strings.
    df_show.index = df_show.index.strftime("%Y-%m-%d")

    # Newest rows first.
    df_show = df_show.sort_index(ascending=False)

    # Slider controls how many rows are displayed.
    n = st.slider("Rows to display", 20, min(500, len(df_show)), 50, step=10)

    df_show = df_show.head(n)

    # Round numeric values for easier display.
    df_show = df_show.round(4)

    st.dataframe(df_show, use_container_width=True)

    # Convert table to CSV bytes for download.
    csv = df_show.to_csv().encode("utf-8")

    st.download_button(
        label=f"Download {focus} data as CSV",
        data=csv,
        file_name=f"{focus}_data.csv",
        mime="text/csv",
    )


# ─────────────────────────────────────────────────────────────
# Tab 6 — Exchanges & Map
# ─────────────────────────────────────────────────────────────

def tab_exchanges_map(exchange_df: pd.DataFrame) -> None:
    """
    Exchange-only map tab backed by a SQLite database.

    exchange_df should come from data.get_exchange_map_dataframe().
    """

    st.subheader("Global Exchange Centres")
    st.caption(
        "This map is driven by a SQLite database built from config.EXCHANGE_COORDS. "
        "The app loads the database back into a pandas DataFrame for plotting."
    )

    if exchange_df is None or exchange_df.empty:
        st.warning("No exchange map data is available.")
        return

    # Plot from the DB-loaded DataFrame
    fig = charts.world_map_chart(exchange_df)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Exchange Dataset (Loaded from Database)")

    def _style_exchange_code(val):
        try:
            color = charts.exchange_color(str(val))
            return f"color: {color}; font-weight: 700;"
        except Exception:
            return ""

    def _style_color_cell(val):
        try:
            return f"background-color: {val}; color: white; font-weight: 700;"
        except Exception:
            return ""

    styled_exchange_df = (
        exchange_df.style
        .applymap(_style_exchange_code, subset=["Exchange Code"])
        .applymap(_style_color_cell, subset=["Color"])
    )

    st.dataframe(styled_exchange_df, use_container_width=True)

    csv = exchange_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download exchange database table as CSV",
        data=csv,
        file_name="exchange_map_database_table.csv",
        mime="text/csv",
    )

# ─────────────────────────────────────────────────────────────
# Tab 7 — Administration (EMA methodology)
# ─────────────────────────────────────────────────────────────

def tab_administration(signals: dict, tickers: list) -> None:
    """
    Educational / transparency tab explaining the EMA crossover system.

    Four sections:
      1. What is EMA?
      2. Signal system.
      3. Live calculation breakdown.
      4. Limitations.

    This tab helps explain why each EMA signal was assigned.
    """

    # ── Section 1: EMA theory ────────────────────────────────
    with st.expander("① What is EMA? — Exponential Moving Average", expanded=True):
        st.markdown("""
An **Exponential Moving Average (EMA)** weights recent prices more heavily than older ones,
making it more responsive to new data than a Simple Moving Average (SMA).

| Formula | Expression |
|---------|-----------|
| Smoothing multiplier | `k = 2 / (N + 1)` |
| EMA on day t | `EMAₜ = Priceₜ × k + EMAₜ₋₁ × (1 − k)` |
| Seed value | `EMA₁ = SMA of first N periods` |
| k for EMA-10 | `k = 2/11 ≈ 0.1818` — 18.2% weight on today |
| k for EMA-26 | `k = 2/27 ≈ 0.0741` —  7.4% weight on today |

> **Why weekly?** Using weekly closes smooths daily noise and makes 10W/26W
> correspond to roughly 2.5 months and 6 months of history.
        """)

    # ── Section 2: Signal system ──────────────────────────────
    with st.expander("② Dual EMA Crossover Signal — 10W vs 26W", expanded=True):
        st.markdown("""
**spread** = `(EMA10 − EMA26) / EMA26 × 100`  
**price_gap** = `(Price − EMA10) / EMA10 × 100`

| Signal | Condition | Rationale |
|--------|-----------|-----------|
| ⬆ Strong Buy | spread > 1% AND price_gap > 2% | Golden cross confirmed by price momentum |
| ↑ Buy | spread > 0.5% | Bullish crossover — EMA10 above EMA26 |
| → Hold | \\|spread\\| ≤ 0.5% | EMAs converging — no clear directional bias |
| ↓ Sell | spread < −0.5% | Bearish crossover — death cross forming |
| ⬇ Strong Sell | spread < −1% AND price_gap < −2% | Death cross confirmed by price below EMA10 |

> The 0.5%/1.0% bands filter out noise in sideways/ranging markets.
        """)

    # ── Section 3: Live per-ticker breakdown ──────────────────
    st.subheader("③ Live Calculation Breakdown")

    if not signals:
        st.info("No signal data available. Make sure at least one valid ticker is loaded.")
        return

    # Loop through each ticker and display its EMA signal breakdown.
    for ticker in tickers:
        sig = signals.get(ticker, {})

        if not sig:
            continue

        # Extract signal fields from the dictionary returned by signals.py.
        sig_key = sig.get("signal", "UNKNOWN")
        sig_color = sig.get("color", "#6b7280")
        sig_label = sig.get("label", "—")
        ema10 = sig.get("ema10")
        ema26 = sig.get("ema26")
        spread = sig.get("spread")
        price_gap = sig.get("price_gap")
        reason = sig.get("reason", "")
        e10h = sig.get("ema10_history", [])
        e26h = sig.get("ema26_history", [])

        # Each ticker gets an expandable detail section.
        with st.expander(
            f"**{ticker}** — {sig_label}  "
            f"{'  |  spread ' + f'{spread:+.2f}%' if spread is not None else ''}",
            expanded=False,
        ):
            cols = st.columns([1, 1, 1, 2])

            with cols[0]:
                st.metric("EMA 10", f"{ema10:.4f}" if ema10 else "—")

            with cols[1]:
                st.metric("EMA 26", f"{ema26:.4f}" if ema26 else "—")

            with cols[2]:
                st.metric("Spread", f"{spread:+.3f}%" if spread is not None else "—")

            with cols[3]:
                # HTML allows the signal label to be color-coded.
                st.markdown(
                    f"<span style='color:{sig_color}; font-weight:600; font-size:15px;'>"
                    f"{sig_label}</span><br>"
                    f"<small style='color:#64748b;'>{reason}</small>",
                    unsafe_allow_html=True,
                )

            # Step-by-step calculation table.
            if ema10 and ema26:
                st.markdown("**Step-by-step**")

                steps = {
                    "① EMA10 (latest weekly)": f"{ema10:.6f}",
                    "② EMA26 (latest weekly)": f"{ema26:.6f}",
                    "③ Spread = (EMA10−EMA26)/EMA26×100": f"{spread:+.4f}%",
                    "④ Price vs EMA10": f"{price_gap:+.2f}%" if price_gap is not None else "—",
                    "⑤ Signal": sig_label,
                }

                # Display each calculation step as a two-column row.
                for k, v in steps.items():
                    sc1, sc2 = st.columns([2, 1])
                    sc1.caption(k)
                    sc2.code(v)

            # 8-week EMA history mini-table.
            if e10h or e26h:
                st.markdown("**Recent weekly EMA history** (newest first)")

                hist_rows = []

                # Build rows even if one history list is longer than the other.
                for j in range(max(len(e10h), len(e26h))):
                    date = e10h[j][0] if j < len(e10h) else (e26h[j][0] if j < len(e26h) else "")
                    v10 = e10h[j][1] if j < len(e10h) else None
                    v26 = e26h[j][1] if j < len(e26h) else None

                    # Calculate spread for each historical row when both EMA values exist.
                    sp = (v10 - v26) / v26 * 100 if v10 and v26 else None

                    hist_rows.append({
                        "Week": date,
                        "EMA 10": f"{v10:.4f}" if v10 else "—",
                        "EMA 26": f"{v26:.4f}" if v26 else "—",
                        "Spread": f"{sp:+.2f}%" if sp is not None else "—",
                    })

                st.dataframe(pd.DataFrame(hist_rows), hide_index=True, use_container_width=True)

#  ───────────────────────────────
# Tab 8: headquarters locations mapping
# ────────────────────────────────

def tab_hq_locations(hq_locations: pd.DataFrame) -> None:
    """
    Plot company headquarters locations gathered by scrape_locations.py.

    This tab is separate from the exchange map. The exchange map shows where
    stocks trade; this tab shows where the companies are headquartered.
    """
    st.subheader("Company Headquarters Locations")

    if hq_locations is None or hq_locations.empty:
        st.warning("No headquarters location data available.")
        return

    found_df = hq_locations[hq_locations["Found"] == True].copy()

    if found_df.empty:
        st.warning("No headquarters coordinates found for the selected tickers.")
        st.dataframe(hq_locations.set_index("Ticker"), use_container_width=True)
        return

    fig = charts.hq_locations_map(found_df)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("HQ Location Detail")
    st.dataframe(
        hq_locations.set_index("Ticker"),
        use_container_width=True,
    )
    
    # ── Section 4: Limitations ────────────────────────────────
    with st.expander("④ Limitations & Caveats"):
        st.markdown("""
- **Lagging indicator** — EMA reacts to price changes after they happen.
  Works well in trending markets but generates false signals in choppy ranges.
- **Weekly timeframe** — smooths daily noise but is slow to react.
  A 26-week EMA takes ~6 months to fully respond to a trend reversal.
- **No fundamentals** — purely technical. A stock can show a bearish EMA
  signal while earnings and cash flow remain strong.
- **API rate limits** — Alpha Vantage free tier: 25 req/day. If the key is
  absent, signals are computed locally from yfinance data (slightly less accurate).
- **Not financial advice** — EMA crossover signals are one input among many.
  Do not use them as the sole basis for any investment decision.
        """)