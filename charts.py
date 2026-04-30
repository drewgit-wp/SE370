# Replit and ChatGPT used across all functions and concepts for development, troubleshooting, and efficiency
# =============================================================
# charts.py  —  Plotly chart builders
# =============================================================
# This file contains chart-building functions only.
#
# Every function returns a Plotly Figure.
# Streamlit displays those figures with:
#   st.plotly_chart(fig, use_container_width=True)
#
# There are no classes in this file.
# The functions are separated so tabs.py can stay focused on layout,
# while charts.py stays focused on visualization logic.
#
# Dark-theme consistency:
#   paper_bgcolor = "rgba(0,0,0,0)"  → transparent outer background
#   plot_bgcolor  = "rgba(0,0,0,0)"  → transparent plot area
#   These blend with Streamlit's dark theme.
#
# File layout:
#   _LAYOUT / _base()      — shared style defaults
#   price_chart()          — price line + SMA overlay + volume bars
#   comparison_chart()     — normalized returns for all tickers
#   metric_bar_chart()     — single metric across all tickers
#   rsi_chart()            — RSI(14) with overbought/oversold bands
#   bollinger_chart()      — Bollinger Bands (SMA20 ± 2σ)
#   volatility_chart()     — 30-day rolling annualized volatility
#   world_map_chart()      — exchange locations on a world map
# =============================================================

from __future__ import annotations

# pandas is used for DataFrame/Series manipulation inside charts.
import pandas as pd

# plotly.graph_objects gives more detailed chart control than plotly.express.
import plotly.graph_objects as go

# Shared chart colors and fallback exchange coordinates come from config.py.
from config import COLORS, EXCHANGE_COORDS, EXCHANGE_COLOR_MAP

# ─────────────────────────────────────────────────────────────
# Shared layout
# ─────────────────────────────────────────────────────────────

# These kwargs are spread into every Figure's update_layout call.
# Keeping this in one dictionary makes the whole dashboard visually consistent.
_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#94a3b8", size=11),

    # More top space so title + legend both fit in Streamlit.
    margin=dict(l=10, r=10, t=90, b=10),

    # Horizontal legend placed under the title area.
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.00,
        xanchor="left",
        x=0,
        bgcolor="rgba(15,23,42,0.35)",
        bordercolor="rgba(255,255,255,0.08)",
        borderwidth=1,
        font=dict(size=11),
    ),

    # Hover labels and axis styling match the dark dashboard theme.
    hoverlabel=dict(bgcolor="#1e293b", bordercolor="#334155", font_color="#f1f5f9"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="#334155"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.05)", linecolor="#334155"),
)
EXCHANGE_COLOR_MAP = {
    "NMS": "#22d3ee",   # NASDAQ
    "NGM": "#22d3ee",   # NASDAQ Global Market
    "NCM": "#22d3ee",   # NASDAQ Capital Market
    "NYQ": "#3b82f6",   # NYSE
    "ASE": "#f59e0b",   # NYSE American
    "LSE": "#a855f7",   # London
    "TOR": "#22c55e",   # Toronto
    "JPX": "#ef4444",   # Japan
    "HKG": "#f97316",   # Hong Kong
    "ASX": "#14b8a6",   # Australia
    "BSE": "#eab308",   # Bombay
    "NSE": "#84cc16",   # National Stock Exchange of India
    "SSE": "#ec4899",   # Shanghai
    "SZSE": "#f43f5e",  # Shenzhen
    "FWB": "#8b5cf6",   # Frankfurt
    "EPA": "#06b6d4",   # Paris
    "AMS": "#10b981",   # Amsterdam
    "SWX": "#60a5fa",   # Swiss
    "KRX": "#fb7185",   # Korea
    "TPE": "#34d399",   # Taiwan
}


def exchange_color(exchange_code: str) -> str:
    """
    Return a consistent color for an exchange code.
    Falls back to slate gray if the exchange is unknown.
    """
    if not exchange_code:
        return "#94a3b8"

    return EXCHANGE_COLOR_MAP.get(str(exchange_code).upper(), "#94a3b8")

def _base(title: str = "") -> go.Figure:
    """
    Return an empty Figure with the shared dark-theme layout applied.

    This helper prevents repeated layout code in every chart function.
    """

    fig = go.Figure()

    # **_LAYOUT expands the dictionary into keyword arguments.
    fig.update_layout(
        **_LAYOUT,
        title=dict(
            text=title,
            font=dict(size=13, color="#cbd5e1"),
            y=0.98,
            yanchor="top",
            x=0,
            xanchor="left",
        ),
    )

    return fig


# ─────────────────────────────────────────────────────────────
# price_chart
# ─────────────────────────────────────────────────────────────

def price_chart(df: pd.DataFrame, ticker: str, color: str) -> go.Figure:
    """
    Dual-panel chart for a single ticker:
      Top panel: closing price line + SMA20 + SMA50
      Bottom panel: volume bars colored by up/down day

    Unique behavior:
      Volume is plotted on yaxis2, which creates a second vertical scale
      inside the same figure.
    """

    fig = _base()

    # Main closing-price line.
    # fill="tozeroy" creates a subtle area fill below the line.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Close"],
        mode="lines",
        name=ticker,
        line=dict(color=color, width=2),
        fill="tozeroy",

        # This chain converts known hex colors into rgba fills.
        # It is a manual approach to create translucent fill colors.
        fillcolor=color.replace("#", "rgba(")
                       .replace("6366f1", "99,102,241,0.08)")
                       .replace("22d3ee", "34,211,238,0.08)")
                       .replace("f59e0b", "245,158,11,0.08)")
                       .replace("22c55e", "34,197,94,0.08)")
                       .replace("ef4444", "239,68,68,0.08)")
                       .replace("a78bfa", "167,139,250,0.08)")
                       .replace("3b82f6", "59,130,246,0.08)")
                       .replace("a855f7", "168,85,247,0.08)"),
        hovertemplate="<b>%{x|%b %d %Y}</b><br>Close $%{y:.2f}<extra></extra>",
    ))

    # SMA20 overlay.
    if "SMA20" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["SMA20"],
            mode="lines",
            name="SMA 20",
            line=dict(color="#f59e0b", width=1.2, dash="dash"),
            hovertemplate="SMA20 $%{y:.2f}<extra></extra>",
        ))

    # SMA50 overlay.
    if "SMA50" in df.columns:
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["SMA50"],
            mode="lines",
            name="SMA 50",
            line=dict(color="#a78bfa", width=1.2, dash="dot"),
            hovertemplate="SMA50 $%{y:.2f}<extra></extra>",
        ))

    # Volume bars are colored green if close >= open, otherwise red.
    vol_colors = [
        "#22c55e" if float(c) >= float(o) else "#ef4444"
        for c, o in zip(df["Close"], df["Open"])
    ]

    # Volume is added as a bar chart using the second y-axis.
    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        name="Volume",
        marker_color=vol_colors,
        opacity=0.55,
        yaxis="y2",
        hovertemplate="Vol %{y:,.0f}<extra></extra>",
    ))

    # yaxis2 is placed in the bottom portion of the chart.
    # This creates the appearance of a price panel above a volume panel.
    fig.update_layout(
        yaxis2=dict(
            domain=[0, 0.22],
            gridcolor="rgba(255,255,255,0.03)",
            linecolor="#334155",
        ),
        xaxis=dict(
            domain=[0, 1],
            gridcolor="rgba(255,255,255,0.05)",
        ),
        barmode="overlay",
    )

    return fig


# ─────────────────────────────────────────────────────────────
# comparison_chart — normalized returns
# ─────────────────────────────────────────────────────────────

def comparison_chart(price_data: dict, tickers: list) -> go.Figure:
    """
    Normalized performance chart for the whole watchlist.

    Each series is rebased to 100 at the start of the selected period:
        normalized_t = (close_t / close_0) × 100

    This makes differently priced stocks comparable on the same scale.
    """

    fig = _base("Normalised Performance — rebased to 100 at period start")

    # Loop through tickers and add one normalized line per ticker.
    for i, ticker in enumerate(tickers):
        df = price_data.get(ticker)

        if df is None or df.empty:
            continue

        close = df["Close"].dropna()

        # Avoid division by zero or empty price data.
        if len(close) == 0 or close.iloc[0] == 0:
            continue

        # Rebase each ticker to 100.
        norm = close / close.iloc[0] * 100

        fig.add_trace(go.Scatter(
            x=norm.index,
            y=norm,
            mode="lines",
            name=ticker,
            line=dict(color=COLORS[i % len(COLORS)], width=2),
            hovertemplate=f"<b>{ticker}</b> %{{y:.1f}}<extra></extra>",
        ))

    # Horizontal reference line at the starting baseline.
    fig.add_hline(y=100, line_dash="dot", line_color="rgba(255,255,255,0.18)")

    return fig


# ─────────────────────────────────────────────────────────────
# metric_bar_chart — compare one metric across all tickers
# ─────────────────────────────────────────────────────────────

def metric_bar_chart(
    tickers: list,
    values: list,
    title: str,
    fmt: str = ".2f",
    suffix: str = "",
) -> go.Figure:
    """
    Bar chart comparing a single numeric metric across all tickers.

    Missing values are skipped so one unavailable metric does not break the chart.
    """

    # Filter out None and NaN values.
    # v == v is False for NaN, which is a common Python trick.
    valid = [
        (t, v)
        for t, v in zip(tickers, values)
        if v is not None and v == v
    ]

    # If all values are missing, return an empty chart with a clear title.
    if not valid:
        return _base(f"{title} — no data available")

    labels, vals = zip(*valid)

    # Use each ticker's position to choose a consistent color.
    colors = [COLORS[tickers.index(t) % len(COLORS)] for t in labels]

    fig = _base(title)

    fig.add_trace(go.Bar(
        x=list(labels),
        y=list(vals),
        marker_color=colors,
        hovertemplate=f"%{{x}}: %{{y:{fmt}}}{suffix}<extra></extra>",
    ))

    return fig


# ─────────────────────────────────────────────────────────────
# rsi_chart — RSI(14) with overbought / oversold zones
# ─────────────────────────────────────────────────────────────

def rsi_chart(df: pd.DataFrame, ticker: str, color: str) -> go.Figure:
    """
    RSI-14 momentum indicator chart.

    Interpretation:
        RSI > 70  — potentially overbought
        RSI < 30  — potentially oversold
        RSI = 50  — neutral momentum
    """

    # If RSI was not calculated, return a placeholder chart.
    if "RSI14" not in df.columns:
        return _base("RSI — not available")

    fig = _base(f"{ticker} — RSI (14)")

    # Main RSI line.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["RSI14"],
        mode="lines",
        name="RSI 14",
        line=dict(color=color, width=1.5),
        hovertemplate="RSI %{y:.1f}<extra></extra>",
    ))

    # Shaded overbought zone.
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,0.09)", line_width=0)

    # Shaded oversold zone.
    fig.add_hrect(y0=0, y1=30, fillcolor="rgba(34,197,94,0.09)", line_width=0)

    # Reference lines.
    fig.add_hline(
        y=70,
        line_dash="dot",
        line_color="rgba(239,68,68,0.45)",
        annotation_text="Overbought (70)",
        annotation_font_color="#ef4444",
    )

    fig.add_hline(
        y=30,
        line_dash="dot",
        line_color="rgba(34,197,94,0.45)",
        annotation_text="Oversold (30)",
        annotation_font_color="#22c55e",
    )

    fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.12)")

    # RSI is always interpreted on a 0 to 100 scale.
    fig.update_yaxes(range=[0, 100])

    return fig


# ─────────────────────────────────────────────────────────────
# bollinger_chart — Bollinger Bands
# ─────────────────────────────────────────────────────────────

def bollinger_chart(df: pd.DataFrame, ticker: str, color: str) -> go.Figure:
    """
    Bollinger Bands volatility/range chart.

    Calculation:
        Middle = SMA(20)
        Upper  = SMA(20) + 2 × std(20)
        Lower  = SMA(20) - 2 × std(20)
    """

    fig = _base(f"{ticker} — Bollinger Bands (SMA20, ±2σ)")

    close = df["Close"]

    # Calculate middle band and standard deviation.
    sma = close.rolling(20).mean()
    std = close.rolling(20).std()

    # Upper and lower bands.
    upper = sma + 2 * std
    lower = sma - 2 * std

    # Filled band envelope.
    # x and y are duplicated/reversed to create a closed polygon.
    fig.add_trace(go.Scatter(
        x=df.index.tolist() + df.index.tolist()[::-1],
        y=upper.tolist() + lower.tolist()[::-1],
        fill="toself",
        fillcolor="rgba(99,102,241,0.09)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Band (2σ)",
        showlegend=True,
    ))

    # Upper band line.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=upper,
        mode="lines",
        name="Upper",
        line=dict(color="#6366f1", width=1, dash="dash"),
    ))

    # Middle SMA line.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=sma,
        mode="lines",
        name="SMA 20",
        line=dict(color="#f59e0b", width=1.5),
    ))

    # Lower band line.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=lower,
        mode="lines",
        name="Lower",
        line=dict(color="#6366f1", width=1, dash="dash"),
    ))

    # Actual price line.
    fig.add_trace(go.Scatter(
        x=df.index,
        y=close,
        mode="lines",
        name=ticker,
        line=dict(color=color, width=1.8),
        hovertemplate="$%{y:.2f}<extra></extra>",
    ))

    return fig


# ─────────────────────────────────────────────────────────────
# volatility_chart — 30-day rolling annualized volatility
# ─────────────────────────────────────────────────────────────

def volatility_chart(df: pd.DataFrame, ticker: str, color: str) -> go.Figure:
    """
    Rolling 30-day annualized volatility area chart.

    Calculation:
        log_return_t = ln(close_t / close_t-1)
        vol_30d_t    = std(log_return, window=30) × sqrt(252) × 100
    """

    # Vol30d is created in data.py.
    if "Vol30d" not in df.columns:
        return _base("Volatility — not available")

    fig = _base(f"{ticker} — 30-Day Rolling Volatility (annualised %)")

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["Vol30d"],
        mode="lines",
        name="Vol 30d",
        line=dict(color=color, width=1.5),
        fill="tozeroy",
        fillcolor="rgba(99,102,241,0.1)",
        hovertemplate="Vol %{y:.1f}%<extra></extra>",
    ))

    return fig


# ─────────────────────────────────────────────────────────────
# world_map_chart — exchange / HQ locations on a globe
# ─────────────────────────────────────────────────────────────
def world_map_chart(exchange_df: pd.DataFrame) -> go.Figure:
    """
    Exchange-only world map.

    This chart expects a DataFrame loaded from the SQLite exchange_map table.
    Required columns:
        Exchange Code
        Exchange Name
        Latitude
        Longitude
        Color
    """

    fig = go.Figure()

    if exchange_df is None or exchange_df.empty:
        fig = _base("Global Exchange Centres — no data available")
        return fig

    df = exchange_df.copy()

    df["Latitude"] = pd.to_numeric(df["Latitude"], errors="coerce")
    df["Longitude"] = pd.to_numeric(df["Longitude"], errors="coerce")
    df = df.dropna(subset=["Latitude", "Longitude"]).reset_index(drop=True)

    if "Color" not in df.columns:
        df["Color"] = df["Exchange Code"].apply(exchange_color)

    fig.add_trace(go.Scattergeo(
        lat=df["Latitude"],
        lon=df["Longitude"],
        text=df["Exchange Code"],
        hovertext=[
            f"<b>{row['Exchange Name']}</b><br>"
            f"Code: {row['Exchange Code']}<br>"
            f"Lat: {float(row['Latitude']):.2f}<br>"
            f"Lon: {float(row['Longitude']):.2f}"
            for _, row in df.iterrows()
        ],
        hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        name="Exchanges",
        marker=dict(
            size=9,
            color=df["Color"],
            opacity=0.92,
            line=dict(width=0.8, color="#0f172a"),
        ),
    ))

    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="#334155",
        showland=True,
        landcolor="#1e293b",
        showocean=True,
        oceancolor="#0f172a",
        showlakes=True,
        lakecolor="#0f172a",
        showcountries=True,
        countrycolor="#334155",
        showframe=False,
        projection_type="natural earth",
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=40, b=0),
        height=520,
        title=dict(
            text="Global Exchange Centres",
            x=0,
            xanchor="left",
            y=0.98,
            yanchor="top",
            font=dict(size=13, color="#cbd5e1"),
        ),
        showlegend=False,
    )

    return fig

# ─────────────────────────────────────────────────────────────
# Company headquarters map
# ─────────────────────────────────────────────────────────────

def hq_locations_map(locations_df: pd.DataFrame) -> go.Figure:
    """
    Map company headquarters using coordinates from scrape_locations.py.

    Expected columns:
      Ticker, Name, Address, Latitude, Longitude, Source
    """
    fig = _base("Company Headquarters Locations")

    if locations_df is None or locations_df.empty:
        return fig

    df = locations_df.dropna(subset=["Latitude", "Longitude"]).copy()

    if df.empty:
        return fig

    fig = go.Figure(go.Scattergeo(
        lat=df["Latitude"],
        lon=df["Longitude"],
        text=df["Ticker"],
        hovertext=[
            f"<b>{row['Ticker']}</b><br>"
            f"{row.get('Name', '')}<br>"
            f"{row.get('Address', '')}<br>"
            f"Source: {row.get('Source', '')}"
            for _, row in df.iterrows()
        ],
        hoverinfo="text",
        mode="markers+text",
        textposition="top center",
        marker=dict(
            size=11,
            color=[COLORS[i % len(COLORS)] for i in range(len(df))],
            opacity=0.9,
            line=dict(width=1, color="#0f172a"),
        ),
    ))

    fig.update_geos(
        showcoastlines=True,
        coastlinecolor="#334155",
        showland=True,
        landcolor="#1e293b",
        showocean=True,
        oceancolor="#0f172a",
        showlakes=True,
        lakecolor="#0f172a",
        showcountries=True,
        countrycolor="#334155",
        showframe=False,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        geo=dict(bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=0, r=0, t=32, b=0),
        height=460,
    )

    return fig