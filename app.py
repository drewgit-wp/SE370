# CDT Miller, Loughman, Fountain
# Replit and ChatGPT used across all functions and concepts for development, troubleshooting, and efficiency
# =============================================================
# app.py  —  Market Terminal (Python / Streamlit entry point)
# =============================================================
# Launch in terminal with:
#   python -m pip install -r requirements.txt
#   python -m streamlit run app.py
# or via the provided setup scripts (run.sh / run.bat).
#
# Architecture overview:
#   app.py              — page layout, controls, summary cards/table, tabs
#   data.py             — all data fetching (yfinance, Alpha Vantage, FMP)
#   signals.py          — EMA dual-crossover signal calculation
#   charts.py           — all Plotly chart builders
#   tabs.py             — all tab rendering functions
#   config.py           — constants, defaults, colors, thresholds
#   scrape_locations.py — HQ address/coordinate lookup for map plotting
# =============================================================

import streamlit as st
import pandas as pd

import config
import data
import signals
import tabs
import scrape_locations
import plotly.express as px


# ─────────────────────────────────────────────────────────────
# Page config  (must be the FIRST Streamlit call)
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Terminal",
    page_icon="📈",
    layout="wide",                  # use the full browser width
    initial_sidebar_state="collapsed",
)


# ── Custom CSS for a dark, terminal-style look ───────────────
# Streamlit's "dark" theme can be toggled in ☰ → Settings,
# but these overrides ensure the background and card styles
# are consistent regardless of the user's OS/browser theme.
st.markdown("""
<style>
  /* Push the page down so Streamlit's header/banner does not cover the controls */
  .block-container {
    padding-top: 4.25rem !important;
    padding-bottom: 1rem;
  }

  /* Give the top header a dark background so it blends in */
  header[data-testid="stHeader"] {
    background: #0f172a;
  }

  /* Hide the thin Streamlit decoration stripe at the very top */
  [data-testid="stDecoration"] {
    display: none;
  }

  /* Summary metric cards */
  [data-testid="metric-container"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 12px;
    padding: 16px 20px;
  }

  /* Top bar ticker input */
  .stTextInput > label {
    display: none;
  }

  /* Dataframe table header */
  .stDataFrame thead th {
    background: rgba(255,255,255,0.05) !important;
  }

  /* Tab strip */
  .stTabs [data-baseweb="tab-list"] {
    gap: 4px;
  }

  .stTabs [data-baseweb="tab"] {
    padding: 6px 16px;
    border-radius: 8px 8px 0 0;
    font-size: 13px;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Top bar — title + controls
# ─────────────────────────────────────────────────────────────
header_col, controls_col = st.columns([1, 3])

with header_col:
    st.markdown("## 📈 MarketTerminal")

with controls_col:
    ctrl1, ctrl2, ctrl3 = st.columns([3, 1, 1])

    with ctrl1:
        ticker_input = st.text_input(
            "Tickers",
            value=", ".join(config.DEFAULT_TICKERS),
            placeholder="AAPL, GOOGL, TSLA",
            label_visibility="collapsed",
        )

    with ctrl2:
        period = st.selectbox(
            "Period",
            config.PERIODS,
            index=config.PERIODS.index(config.DEFAULT_PERIOD),
            label_visibility="collapsed",
        )

    with ctrl3:
        interval = st.selectbox(
            "Interval",
            config.INTERVALS,
            index=config.INTERVALS.index(config.DEFAULT_INTERVAL),
            label_visibility="collapsed",
        )


# Parse tickers from the text input.
# Split on commas, strip whitespace, uppercase, and remove blank entries.
tickers = [t.strip().upper() for t in ticker_input.split(",") if t.strip()]

if not tickers:
    st.warning("Enter at least one ticker symbol in the search bar.")
    st.stop()

st.divider()


# ─────────────────────────────────────────────────────────────
# Data fetching
# ─────────────────────────────────────────────────────────────
# st.status() shows a progress spinner while loading.
# All data.py functions use @st.cache_data, so repeated runs
# with the same parameters return instantly from cache.

with st.status("Loading market data…", expanded=False) as status:
    st.write("Fetching price history (Yahoo Finance)…")
    price_data = data.fetch_price_data(tuple(tickers), period, interval)

    st.write("Fetching fundamentals…")
    fundamentals = data.fetch_all_fundamentals(tuple(tickers))

    st.write("Computing EMA signals…")
    signal_data = signals.compute_all_signals(price_data)

    st.write("Building recommendation table…")
    overview_df = data.build_market_overview(price_data, fundamentals, signal_data)

    st.write("Fetching exchange data (FMP)…")
    fmp_data = {t: data.fetch_exchange_fmp(t) for t in tickers}

    status.update(label="Data loaded", state="complete")


# Only keep tickers that returned valid price data.
if overview_df.empty:
    st.error("No valid price data returned. Check your ticker symbols and try again.")
    st.stop()

valid_tickers = overview_df["Ticker"].tolist()

# ─────────────────────────────────────────────────────────────
# Headquarters location scraping / geocoding
# ─────────────────────────────────────────────────────────────
# This uses the tickers already validated above.
# scrape_locations.py first tries existing FMP/yfinance data, then geocodes
# an address or company HQ search if coordinates are not already available.
# It also writes hq_locations_cache.json to avoid repeated web calls.

with st.status("Resolving headquarters locations…", expanded=False) as status:
    hq_locations = scrape_locations.fetch_hq_locations(
        valid_tickers,
        fundamentals=fundamentals,
        fmp_data=fmp_data,
    )

    status.update(label="Headquarters locations resolved", state="complete")


# ─────────────────────────────────────────────────────────────
# Summary cards  (4 KPI metrics at the top)
# ─────────────────────────────────────────────────────────────
# Compute the values used in the four summary cards:
#   1. Active Tickers      — count of valid symbols
#   2. Best 1-Month Return — top performer (%) in the watchlist
#   3. Largest Market Cap  — biggest company by market cap
#   4. Average P/E         — mean trailing P/E across the basket

def _1m_return(ticker: str) -> float | None:
    """Return the 21-bar, approximately 1-month, return for a ticker."""
    df = price_data.get(ticker)

    if df is None or len(df) < 22:
        return None

    close = df["Close"].dropna()

    if len(close) < 22 or close.iloc[-22] == 0:
        return None

    return (close.iloc[-1] / close.iloc[-22] - 1) * 100


returns_1m = {t: _1m_return(t) for t in valid_tickers}

best_ticker = max(
    (t for t in valid_tickers if returns_1m.get(t) is not None),
    key=lambda t: returns_1m[t],
    default=None,
)

mkt_caps = {
    t: fundamentals.get(t, {}).get("marketCap")
    for t in valid_tickers
}

largest_ticker = max(
    (t for t in valid_tickers if mkt_caps.get(t)),
    key=lambda t: mkt_caps[t],
    default=None,
)

pe_values = [
    fundamentals.get(t, {}).get("trailingPE")
    for t in valid_tickers
    if fundamentals.get(t, {}).get("trailingPE") is not None
]

avg_pe = (
    sum(float(v) for v in pe_values) / len(pe_values)
    if pe_values
    else None
)


def fmt_mcap(v) -> str:
    """Format a market-cap number as T/B/M."""
    if v is None:
        return "—"

    v = float(v)

    if v >= 1e12:
        return f"${v / 1e12:.2f}T"

    if v >= 1e9:
        return f"${v / 1e9:.2f}B"

    return f"${v / 1e6:.1f}M"


c1, c2, c3, c4 = st.columns(4)

c1.metric("Active Tickers", len(valid_tickers))

c2.metric(
    "Best 1M Return",
    f"{best_ticker}" if best_ticker else "—",
    delta=f"{returns_1m[best_ticker]:+.2f}%" if best_ticker else None,
)

c3.metric(
    "Largest Market Cap",
    f"{largest_ticker}" if largest_ticker else "—",
    delta=fmt_mcap(mkt_caps.get(largest_ticker)) if largest_ticker else None,
    delta_color="off",
)

c4.metric(
    "Average P/E (Trailing)",
    f"{avg_pe:.1f}" if avg_pe else "—",
)

st.divider()

# ─────────────────────────────────────────────────────────────
# Market Overview — recommendations restored
# ─────────────────────────────────────────────────────────────

st.markdown("### Market Overview")

display_cols = [
    "Ticker",
    "Price",
    "Change $",
    "Return 1D %",
    "Return 5D %",
    "Return 1M %",
    "Dividend Yield %",
    "EMA Signal",
    "Trend Suggestion",
    "RSI Suggestion",
    "Bollinger Suggestion",
    "Volatility Suggestion",
    "Final Recommendation",
    "Market Cap Display",
    "Volume",
    "P/E",
]

display_df = overview_df[display_cols].copy()
display_df = display_df.rename(columns={"Market Cap Display": "Market Cap"})
display_df = display_df.set_index("Ticker")


def style_positive_negative(value):
    try:
        if pd.isna(value):
            return ""
        value = float(value)
    except Exception:
        return ""

    if value > 0:
        return "color: #22c55e; font-weight: 700;"
    if value < 0:
        return "color: #ef4444; font-weight: 700;"
    return "color: #cbd5e1;"


def style_dividend(value):
    try:
        if pd.isna(value) or float(value) <= 0:
            return "color: #64748b;"
    except Exception:
        return "color: #64748b;"

    return "color: #22c55e; font-weight: 700;"


def style_recommendation(value):
    text = str(value).lower()

    if "strong buy" in text:
        return "color: #10b981; background-color: rgba(16,185,129,0.16); font-weight: 800;"
    if "buy" in text:
        return "color: #22c55e; background-color: rgba(34,197,94,0.13); font-weight: 700;"
    if "hold" in text:
        return "color: #f59e0b; background-color: rgba(245,158,11,0.13); font-weight: 700;"
    if "strong sell" in text:
        return "color: #dc2626; background-color: rgba(220,38,38,0.18); font-weight: 800;"
    if "sell" in text:
        return "color: #ef4444; background-color: rgba(239,68,68,0.13); font-weight: 700;"

    return "color: #94a3b8;"


styled_display_df = (
    display_df.style
    .map(
        style_positive_negative,
        subset=["Change $", "Return 1D %", "Return 5D %", "Return 1M %"],
    )
    .map(
        style_dividend,
        subset=["Dividend Yield %"],
    )
    .map(
        style_recommendation,
        subset=[
            "EMA Signal",
            "Trend Suggestion",
            "RSI Suggestion",
            "Bollinger Suggestion",
            "Volatility Suggestion",
            "Final Recommendation",
        ],
    )
)

st.dataframe(
    styled_display_df,
    use_container_width=True,
    column_config={
        "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
        "Change $": st.column_config.NumberColumn("Day Change", format="$%.2f"),
        "Return 1D %": st.column_config.NumberColumn("1D %", format="%.2f%%"),
        "Return 5D %": st.column_config.NumberColumn("5D %", format="%.2f%%"),
        "Return 1M %": st.column_config.NumberColumn("1M %", format="%.2f%%"),
        "Dividend Yield %": st.column_config.NumberColumn("Dividend", format="%.2f%%"),
        "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
        "P/E": st.column_config.NumberColumn("P/E", format="%.1f"),
        "EMA Signal": st.column_config.TextColumn("EMA"),
        "Trend Suggestion": st.column_config.TextColumn("Trend"),
        "RSI Suggestion": st.column_config.TextColumn("RSI"),
        "Bollinger Suggestion": st.column_config.TextColumn("Bollinger"),
        "Volatility Suggestion": st.column_config.TextColumn("Volatility"),
        "Final Recommendation": st.column_config.TextColumn("Final"),
    },
)

ema_note = "Final recommendation combines EMA, trend, RSI, Bollinger, and volatility signals."
st.caption(ema_note)

bubble_df = overview_df.copy()
bubble_df["Bubble Volume"] = bubble_df["Volume"].fillna(0).clip(lower=1)

fig_bubble = px.scatter(
    bubble_df,
    x="Return 1D %",
    y="Return 1M %",
    size="Bubble Volume",
    color="Final Recommendation",
    color_discrete_map={
        "⬆ Strong Buy": "#10b981",
        "↑ Buy": "#22c55e",
        "→ Hold": "#f59e0b",
        "↓ Sell": "#ef4444",
        "⬇ Strong Sell": "#dc2626",
        "—": "#94a3b8",
    },
    category_orders={
        "Final Recommendation": [
            "⬆ Strong Buy",
            "↑ Buy",
            "→ Hold",
            "↓ Sell",
            "⬇ Strong Sell",
            "—",
        ]
    },
    hover_name="Ticker",
    hover_data={
        "Price": ":$.2f",
        "Change $": ":$.2f",
        "Return 1D %": ":.2f",
        "Return 5D %": ":.2f",
        "Return 1M %": ":.2f",
        "Market Cap Display": True,
        "Volume": ":,.0f",
        "P/E": ":.1f",
        "Dividend Yield %": ":.2f",
        "EMA Signal": True,
        "Trend Suggestion": True,
        "RSI Suggestion": True,
        "Bollinger Suggestion": True,
        "Volatility Suggestion": True,
        "Final Recommendation": True,
        "Bubble Volume": False,
    },
    title="Market Overview Bubble Chart: 1D Move vs 1M Return",
    labels={
        "Return 1D %": "1D Return (%)",
        "Return 1M %": "1M Return (%)",
        "Final Recommendation": "Final Recommendation",
    },
)

fig_bubble.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color="#cbd5e1"),
    title=dict(
        text="Market Overview Bubble Chart: 1D Move vs 1M Return",
        y=0.98,
        yanchor="top",
        x=0,
        xanchor="left",
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.00,
        xanchor="left",
        x=0,
        bgcolor="rgba(15,23,42,0.35)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
    ),
    margin=dict(l=10, r=10, t=90, b=10),
)

fig_bubble.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.25)")
fig_bubble.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=True, zerolinecolor="rgba(255,255,255,0.25)")

st.plotly_chart(fig_bubble, use_container_width=True)

st.divider()




# ─────────────────────────────────────────────────────────────
# Stock Interactive Heatmap
# ─────────────────────────────────────────────────────────────

def tab_stock_heatmap(overview_df: pd.DataFrame, fundamentals: dict, price_data: dict, valid_tickers: list) -> None:
    """Interactive heatmap grouped by sector with live ticker chart."""

    st.subheader("🔥 Stock Interactive Heatmap")
    st.caption(
        "Green = positive return, red = negative return, white = near flat. "
        "Box size is based on market cap when available, otherwise volume."
    )

    heatmap_df = overview_df.copy()

    heatmap_df["Sector"] = heatmap_df["Ticker"].apply(
        lambda t: fundamentals.get(t, {}).get("sector") or "Unknown"
    )
    heatmap_df["Company"] = heatmap_df["Ticker"].apply(
        lambda t: fundamentals.get(t, {}).get("shortName")
        or fundamentals.get(t, {}).get("longName")
        or t
    )

    heatmap_df["Market Cap Raw"] = heatmap_df["Ticker"].apply(
        lambda t: fundamentals.get(t, {}).get("marketCap")
    )
    heatmap_df["Heatmap Size"] = pd.to_numeric(heatmap_df["Market Cap Raw"], errors="coerce")
    heatmap_df["Heatmap Size"] = heatmap_df["Heatmap Size"].fillna(
        pd.to_numeric(heatmap_df.get("Volume", 1), errors="coerce").fillna(1)
    )
    heatmap_df["Heatmap Size"] = heatmap_df["Heatmap Size"].clip(lower=1)

    col_a, col_b, col_c = st.columns([1, 1, 2])

    with col_a:
        color_metric = st.selectbox(
            "Color by performance",
            ["Return 1D %", "Return 5D %", "Return 1M %"],
            index=0,
        )

    with col_b:
        group_by = st.selectbox(
            "Group by",
            ["Sector", "Final Recommendation"],
            index=0,
        )

    with col_c:
        sector_filter = st.multiselect(
            "Filter by sector",
            sorted(heatmap_df["Sector"].dropna().unique()),
            default=sorted(heatmap_df["Sector"].dropna().unique()),
        )

    heatmap_df = heatmap_df[heatmap_df["Sector"].isin(sector_filter)]

    if heatmap_df.empty:
        st.warning("No stocks match your heatmap filters.")
        return

    st.markdown(
        """
        **Color chart:** 🟢 Up | ⚪ Flat | 🔴 Down  
        **Box size:** larger box = larger market cap, or volume if market cap is unavailable.
        """
    )

    fig_heatmap = px.treemap(
        heatmap_df,
        path=[group_by, "Ticker"],
        values="Heatmap Size",
        color=color_metric,
        color_continuous_scale=["#ef4444", "#f8fafc", "#22c55e"],
        color_continuous_midpoint=0,
        hover_name="Ticker",
        hover_data={
            "Company": True,
            "Sector": True,
            "Price": ":$.2f",
            "Change $": ":$.2f",
            "Return 1D %": ":.2f",
            "Return 5D %": ":.2f",
            "Return 1M %": ":.2f",
            "Final Recommendation": True,
            "Market Cap Display": True,
            "Volume": ":,.0f",
            "Heatmap Size": False,
            "Market Cap Raw": False,
        },
        title=f"Stock Heatmap — colored by {color_metric}",
    )

    fig_heatmap.update_layout(
        height=700,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=10, r=10, t=60, b=10),
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    st.subheader("Heatmap Data Table")
    table_cols = [
        "Ticker", "Company", "Sector", "Price", "Change $",
        "Return 1D %", "Return 5D %", "Return 1M %",
        "Final Recommendation", "Market Cap Display", "Volume", "P/E",
    ]
    st.dataframe(
        heatmap_df[table_cols].sort_values(color_metric, ascending=False),
        use_container_width=True,
        column_config={
            "Price": st.column_config.NumberColumn("Price", format="$%.2f"),
            "Change $": st.column_config.NumberColumn("Change", format="$%.2f"),
            "Return 1D %": st.column_config.NumberColumn("1D %", format="%.2f%%"),
            "Return 5D %": st.column_config.NumberColumn("5D %", format="%.2f%%"),
            "Return 1M %": st.column_config.NumberColumn("1M %", format="%.2f%%"),
            "Volume": st.column_config.NumberColumn("Volume", format="%.0f"),
            "P/E": st.column_config.NumberColumn("P/E", format="%.1f"),
        },
    )

    st.subheader("Live Data Chart")
    chart_ticker = st.selectbox("Choose ticker for chart", heatmap_df["Ticker"].tolist())
    chart_df = price_data.get(chart_ticker)

    if chart_df is None or chart_df.empty:
        st.warning(f"No chart data available for {chart_ticker}.")
        return

    fig_line = px.line(
        chart_df,
        x=chart_df.index,
        y="Close",
        title=f"{chart_ticker} Price Over Selected Timeline",
    )
    fig_line.update_layout(
        height=450,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#cbd5e1"),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig_line.update_xaxes(gridcolor="rgba(255,255,255,0.05)")
    fig_line.update_yaxes(gridcolor="rgba(255,255,255,0.05)")

    st.plotly_chart(fig_line, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# 9-Tab panel
# ─────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "📊 Overview",
    "⚖️ Comparison",
    "🔬 Deep Dive",
    "📋 Fundamentals",
    "🗃️ Raw Data",
    "🗺️ Exchanges Map",
    "📍 HQ Locations",
    "🔥 Heatmap",
    "⚙️ Administration",
])


with tab1:
    tabs.tab_overview(price_data, valid_tickers)


with tab2:
    tabs.tab_comparison(price_data, fundamentals, valid_tickers)


with tab3:
    tabs.tab_deep_dive(price_data, valid_tickers)


with tab4:
    tabs.tab_fundamentals(fundamentals, valid_tickers)


with tab5:
    tabs.tab_raw_data(price_data, valid_tickers)


with tab6:
    tabs.tab_exchanges_map(valid_tickers, fundamentals, hq_locations)

with tab7:
    tabs.tab_hq_locations(hq_locations)

with tab8:
    tab_stock_heatmap(overview_df, fundamentals, price_data, valid_tickers)

with tab9:
    tabs.tab_administration(signal_data, valid_tickers)
