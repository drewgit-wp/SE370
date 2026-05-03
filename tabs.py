from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.express as px

import charts
from config import COLORS


def tab_overview(price_data: dict, tickers: list) -> None:
    st.subheader("Portfolio Performance")
    fig = charts.comparison_chart(price_data, tickers)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Session Volume")
    vol_values = []

    for ticker in tickers:
        df = price_data.get(ticker)
        vol_values.append(float(df["Volume"].iloc[-1]) if df is not None and not df.empty else None)

    fig_vol = charts.metric_bar_chart(tickers, vol_values, "Volume", ".3s")
    st.plotly_chart(fig_vol, use_container_width=True)


def tab_comparison(price_data: dict, fundamentals: dict, tickers: list) -> None:
    st.subheader("Ticker Comparison")

    rows = []

    for ticker in tickers:
        df = price_data.get(ticker)
        info = fundamentals.get(ticker, {})

        if df is None or df.empty:
            continue

        close = df["Close"].dropna()
        price = close.iloc[-1]

        ret_1m = None
        if len(close) >= 22:
            ret_1m = (close.iloc[-1] / close.iloc[-22] - 1) * 100

        rows.append({
            "Ticker": ticker,
            "Price": price,
            "1M Return %": ret_1m,
            "Market Cap": info.get("marketCap"),
            "P/E": info.get("trailingPE"),
            "Beta": info.get("beta"),
            "Sector": info.get("sector", "Unknown"),
        })

    df_compare = pd.DataFrame(rows)
    st.dataframe(df_compare, use_container_width=True)

    if not df_compare.empty:
        fig = px.bar(
            df_compare,
            x="Ticker",
            y="1M Return %",
            color="Sector",
            title="1M Return by Ticker",
        )
        st.plotly_chart(fig, use_container_width=True)


def tab_deep_dive(price_data: dict, tickers: list) -> None:
    st.subheader("Deep Dive")

    focus = st.selectbox("Focus ticker", tickers, key="deep_dive_select")
    df = price_data.get(focus)

    if df is None or df.empty:
        st.warning(f"No data available for {focus}.")
        return

    color = COLORS[tickers.index(focus) % len(COLORS)]

    st.plotly_chart(charts.price_chart(df, focus, color), use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.plotly_chart(charts.rsi_chart(df, focus, color), use_container_width=True)

    with col2:
        st.plotly_chart(charts.volatility_chart(df, focus, color), use_container_width=True)


def tab_stock_heatmap(
    overview_df: pd.DataFrame,
    fundamentals: dict,
    price_data: dict,
    tickers: list,
) -> None:
    st.subheader("🔥 Stock Market Heatmap")
    st.caption(
        "TradingView-style heatmap grouped by sector, sized by market cap or volume, "
        "and colored by selected return period."
    )

    if overview_df is None or overview_df.empty:
        st.warning("No market overview data available for the heatmap.")
        return

    heatmap_df = overview_df.copy()

    heatmap_df["Sector"] = heatmap_df["Ticker"].apply(
        lambda t: fundamentals.get(t, {}).get("sector", "Unknown")
    )

    heatmap_df["Industry"] = heatmap_df["Ticker"].apply(
        lambda t: fundamentals.get(t, {}).get("industry", "Unknown")
    )

    heatmap_df["Company"] = heatmap_df["Ticker"].apply(
        lambda t: fundamentals.get(t, {}).get("shortName")
        or fundamentals.get(t, {}).get("longName")
        or t
    )

    heatmap_df["Market Cap"] = heatmap_df["Ticker"].apply(
        lambda t: fundamentals.get(t, {}).get("marketCap", 0)
    )

    def calc_return(ticker: str, bars: int | None) -> float | None:
        df = price_data.get(ticker)

        if df is None or df.empty or "Close" not in df.columns:
            return None

        close = df["Close"].dropna()

        if len(close) < 2:
            return None

        if bars is None:
            start = close.iloc[0]
        else:
            if len(close) <= bars:
                return None
            start = close.iloc[-bars - 1]

        end = close.iloc[-1]

        if start == 0:
            return None

        return (end / start - 1) * 100

    heatmap_df["Return 1D %"] = heatmap_df["Ticker"].apply(lambda t: calc_return(t, 1))
    heatmap_df["Return 1W %"] = heatmap_df["Ticker"].apply(lambda t: calc_return(t, 5))
    heatmap_df["Return 1M %"] = heatmap_df["Ticker"].apply(lambda t: calc_return(t, 21))
    heatmap_df["Return 1Y %"] = heatmap_df["Ticker"].apply(lambda t: calc_return(t, 252))
    heatmap_df["Return 5Y %"] = heatmap_df["Ticker"].apply(lambda t: calc_return(t, 252 * 5))
    heatmap_df["Return Max %"] = heatmap_df["Ticker"].apply(lambda t: calc_return(t, None))

    numeric_cols = [
        "Price",
        "Change $",
        "Return 1D %",
        "Return 1W %",
        "Return 1M %",
        "Return 1Y %",
        "Return 5Y %",
        "Return Max %",
        "Volume",
        "Market Cap",
    ]

    for col in numeric_cols:
        if col in heatmap_df.columns:
            heatmap_df[col] = pd.to_numeric(heatmap_df[col], errors="coerce")

    heatmap_df["Sector"] = heatmap_df["Sector"].fillna("Unknown")
    heatmap_df["Industry"] = heatmap_df["Industry"].fillna("Unknown")
    heatmap_df["Market Cap"] = heatmap_df["Market Cap"].fillna(0)
    heatmap_df["Volume"] = heatmap_df["Volume"].fillna(0)

    heatmap_df["Market Cap Size"] = heatmap_df["Market Cap"].clip(lower=1)
    heatmap_df["Volume Size"] = heatmap_df["Volume"].clip(lower=1)

    c1, c2, c3 = st.columns(3)

    with c1:
        group_by = st.selectbox(
            "Grouping",
            ["Sector", "Industry", "Final Recommendation"],
            index=0,
            key="heatmap_group_by",
        )

    with c2:
        block_color = st.selectbox(
            "Block Color / Return Period",
            [
                "Return 1D %",
                "Return 1W %",
                "Return 1M %",
                "Return 1Y %",
                "Return 5Y %",
                "Return Max %",
            ],
            index=0,
            key="heatmap_block_color",
        )

    with c3:
        block_size = st.selectbox(
            "Block Size",
            ["Market Cap Size", "Volume Size"],
            index=0,
            key="heatmap_block_size",
        )

    groups = sorted(heatmap_df[group_by].fillna("Unknown").unique())

    selected_groups = st.multiselect(
        f"Filter by {group_by}",
        groups,
        default=groups,
        key="heatmap_group_filter",
    )

    heatmap_df = heatmap_df[heatmap_df[group_by].isin(selected_groups)]
    heatmap_df = heatmap_df.dropna(subset=[block_color])

    if heatmap_df.empty:
        st.warning("No stocks match the selected heatmap filters.")
        return

    gainers = int((heatmap_df[block_color] > 0).sum())
    losers = int((heatmap_df[block_color] < 0).sum())
    avg_change = heatmap_df[block_color].mean()

    best_row = heatmap_df.loc[heatmap_df[block_color].idxmax()]
    worst_row = heatmap_df.loc[heatmap_df[block_color].idxmin()]

    m1, m2, m3, m4 = st.columns(4)

    m1.metric("Stocks Loaded", len(heatmap_df))
    m2.metric("Average Return", f"{avg_change:.2f}%")
    m3.metric("Gainers / Losers", f"{gainers} / {losers}")
    m4.metric(
        "Best / Worst",
        f"{best_row['Ticker']} / {worst_row['Ticker']}",
        delta=f"{best_row[block_color]:+.2f}% / {worst_row[block_color]:+.2f}%",
    )

    st.markdown(
        "**Color Meaning:** Dark green = strong gain · Light green = small gain · Gray = flat · Light red = small loss · Dark red = strong loss"
    )

    fig = px.treemap(
        heatmap_df,
        path=[group_by, "Ticker"],
        values=block_size,
        color=block_color,
        color_continuous_scale=[
            (0.00, "#450a0a"),
            (0.15, "#7f1d1d"),
            (0.30, "#dc2626"),
            (0.45, "#fca5a5"),
            (0.50, "#334155"),
            (0.55, "#86efac"),
            (0.70, "#22c55e"),
            (0.85, "#15803d"),
            (1.00, "#052e16"),
        ],
        color_continuous_midpoint=0,
        hover_name="Company",
        hover_data={
            "Ticker": True,
            "Sector": True,
            "Industry": True,
            "Price": ":$.2f",
            "Change $": ":$.2f",
            "Return 1D %": ":.2f",
            "Return 1W %": ":.2f",
            "Return 1M %": ":.2f",
            "Return 1Y %": ":.2f",
            "Return 5Y %": ":.2f",
            "Return Max %": ":.2f",
            "Market Cap": ":,.0f",
            "Volume": ":,.0f",
            "Final Recommendation": True,
            "Market Cap Size": False,
            "Volume Size": False,
        },
    )

    fig.update_traces(
        texttemplate="<b>%{label}</b><br>%{color:+.2f}%",
        textfont=dict(size=18, color="white", family="Arial Black"),
        marker=dict(line=dict(width=2, color="#020617")),
        tiling=dict(pad=4),
    )

    fig.update_layout(
        height=850,
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font=dict(color="#e5e7eb", size=14),
        margin=dict(t=30, l=5, r=5, b=5),
        coloraxis_colorbar=dict(
            title=block_color,
            tickfont=dict(color="#e5e7eb"),
            titlefont=dict(color="#e5e7eb"),
            thickness=16,
        ),
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Heatmap Data")

    table_cols = [
        "Ticker",
        "Company",
        "Sector",
        "Industry",
        "Price",
        "Change $",
        "Return 1D %",
        "Return 1W %",
        "Return 1M %",
        "Return 1Y %",
        "Return 5Y %",
        "Return Max %",
        "Market Cap",
        "Volume",
        "Final Recommendation",
    ]

    table_cols = [c for c in table_cols if c in heatmap_df.columns]

    st.dataframe(
        heatmap_df[table_cols].sort_values(block_color, ascending=False),
        use_container_width=True,
    )

    st.subheader("Stock Performance Timeline")

    selected_ticker = st.selectbox(
        "Choose ticker for price chart",
        heatmap_df["Ticker"].tolist(),
        key="heatmap_selected_ticker",
    )

    selected_df = price_data.get(selected_ticker)

    if selected_df is not None and not selected_df.empty:
        chart_fig = px.line(
            selected_df,
            x=selected_df.index,
            y="Close",
            title=f"{selected_ticker} Price Chart",
        )

        chart_fig.update_layout(
            height=450,
            paper_bgcolor="#020617",
            plot_bgcolor="#020617",
            font=dict(color="#e5e7eb"),
            xaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
            yaxis=dict(gridcolor="rgba(255,255,255,0.08)"),
        )

        st.plotly_chart(chart_fig, use_container_width=True)
    else:
        st.warning(f"No chart data available for {selected_ticker}.")


def tab_fundamentals(fundamentals: dict, tickers: list) -> None:
    st.subheader("Fundamentals")

    rows = []

    for ticker in tickers:
        info = fundamentals.get(ticker, {})

        rows.append({
            "Ticker": ticker,
            "Company": info.get("shortName") or info.get("longName") or ticker,
            "Sector": info.get("sector", "Unknown"),
            "Industry": info.get("industry", "Unknown"),
            "Market Cap": info.get("marketCap"),
            "P/E": info.get("trailingPE"),
            "Dividend Yield": info.get("dividendYield"),
            "Beta": info.get("beta"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)


def tab_raw_data(price_data: dict, tickers: list) -> None:
    st.subheader("Raw Price Data")

    ticker = st.selectbox("Choose ticker", tickers, key="raw_data_ticker")
    df = price_data.get(ticker)

    if df is None or df.empty:
        st.warning(f"No raw data available for {ticker}.")
        return

    st.dataframe(df.tail(250), use_container_width=True)


def tab_exchanges_map(valid_tickers: list, fundamentals: dict, hq_locations) -> None:
    st.subheader("Exchanges / HQ Map")

    if hq_locations is None:
        st.warning("No HQ location data available.")
        return

    if isinstance(hq_locations, pd.DataFrame):
        loc_source = hq_locations.copy()
    elif isinstance(hq_locations, dict):
        loc_source = pd.DataFrame.from_dict(hq_locations, orient="index")
        if "Ticker" not in loc_source.columns:
            loc_source["Ticker"] = loc_source.index
    else:
        st.warning("HQ location data is in an unsupported format.")
        return

    rows = []

    for ticker in valid_tickers:
        info = fundamentals.get(ticker, {})

        if "Ticker" in loc_source.columns:
            match = loc_source[
                loc_source["Ticker"].astype(str).str.upper() == ticker.upper()
            ]
        else:
            match = pd.DataFrame()

        if not match.empty:
            loc = match.iloc[0].to_dict()
        else:
            loc = {}

        rows.append({
            "Ticker": ticker,
            "Company": loc.get("Name") or info.get("shortName") or info.get("longName") or ticker,
            "City": loc.get("City", ""),
            "State": loc.get("State", ""),
            "Country": loc.get("Country", ""),
            "Latitude": loc.get("Latitude"),
            "Longitude": loc.get("Longitude"),
        })

    map_df = pd.DataFrame(rows)
    st.dataframe(map_df, use_container_width=True)

    map_plot_df = map_df.dropna(subset=["Latitude", "Longitude"])

    if not map_plot_df.empty:
        fig = px.scatter_geo(
            map_plot_df,
            lat="Latitude",
            lon="Longitude",
            hover_name="Ticker",
            hover_data=["Company", "City", "State", "Country"],
            title="Company HQ Locations",
        )

        fig.update_layout(height=600)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No location coordinates available to map.")


def tab_hq_locations(hq_locations) -> None:
    st.subheader("HQ Locations")

    if hq_locations is None:
        st.warning("No HQ location data available.")
        return

    if isinstance(hq_locations, pd.DataFrame):
        if hq_locations.empty:
            st.warning("No HQ location data available.")
            return

        st.dataframe(hq_locations, use_container_width=True)
        return

    if isinstance(hq_locations, dict):
        if len(hq_locations) == 0:
            st.warning("No HQ location data available.")
            return

        rows = []

        for ticker, loc in hq_locations.items():
            rows.append({
                "Ticker": ticker,
                "Name": loc.get("Name", ""),
                "Address": loc.get("Address", ""),
                "City": loc.get("City", ""),
                "State": loc.get("State", ""),
                "Country": loc.get("Country", ""),
                "Latitude": loc.get("Latitude"),
                "Longitude": loc.get("Longitude"),
                "Found": loc.get("Found"),
            })

        st.dataframe(pd.DataFrame(rows), use_container_width=True)
        return

    st.warning("HQ location data is in an unsupported format.")


def tab_administration(signal_data: dict, valid_tickers: list) -> None:
    st.subheader("Administration / EMA Signals")

    rows = []

    for ticker in valid_tickers:
        signal = signal_data.get(ticker, {})

        rows.append({
            "Ticker": ticker,
            "Signal": signal.get("label") or signal.get("signal"),
            "EMA10": signal.get("ema10"),
            "EMA26": signal.get("ema26"),
            "Spread": signal.get("spread"),
            "Price": signal.get("price"),
            "Reason": signal.get("reason"),
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)
