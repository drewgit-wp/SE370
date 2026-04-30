# Replit and ChatGPT used across all functions and concepts for development, troubleshooting, and efficiency
# =============================================================
# config.py  —  Configuration & shared constants
# =============================================================
# This file centralizes project settings.
#
# Other files import from config.py so values like default tickers,
# chart colors, API keys, signal thresholds, and API endpoints only
# need to be changed in one place.
#
# There are no classes in this file.
# It mainly contains constants and one helper function.
# =============================================================

# os is used to read environment variables.
import os

# load_dotenv loads variables from a local .env file into os.environ.
from dotenv import load_dotenv

# load_dotenv() reads the .env file in the same directory and
# injects its KEY=value pairs into os.environ so os.getenv works.
load_dotenv()


# ── API Keys (read from .env) ─────────────────────────────────
# These are optional.
# The app works without them, but certain features become limited:
#   - Alpha Vantage improves EMA signal data.
#   - FMP improves exchange/HQ map data.

# Alpha Vantage API key for optional EMA API data.
ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")

# Financial Modeling Prep API key for optional profile/map data.
FMP_KEY: str = os.getenv("FMP_API_KEY", "")


# ── Default watchlist and time controls ──────────────────────

# Default tickers shown in the app when it first loads.
DEFAULT_TICKERS: list = ["GOOGL", "AAPL", "PLTR", "MU", "TSLA"]

# Default yfinance lookback period.
DEFAULT_PERIOD: str = "1y"

# Default yfinance data interval.
DEFAULT_INTERVAL: str = "1d"

# Dropdown choices used in app.py.
PERIODS: list = ["5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]

# Dropdown choices used in app.py.
INTERVALS: list = ["1d", "1wk", "1mo"]


# ── Chart colors (hex, one per ticker, cycles after 8) ────────

# Shared chart palette.
# The app uses these colors consistently across charts and tabs.
COLORS: list = [
    "#6366f1",  # indigo
    "#22d3ee",  # cyan
    "#f59e0b",  # amber
    "#22c55e",  # green
    "#ef4444",  # red
    "#a78bfa",  # violet
    "#3b82f6",  # blue
    "#a855f7",  # purple
]
EXCHANGE_COLOR_MAP: dict = {
    "NMS": "#22d3ee",      # NASDAQ
    "NGM": "#22d3ee",      # NASDAQ Global Market
    "NCM": "#22d3ee",      # NASDAQ Capital Market
    "NYQ": "#3b82f6",      # NYSE
    "ASE": "#f59e0b",      # NYSE American
    "LSE": "#a855f7",      # London
    "TOR": "#22c55e",      # Toronto
    "JPX": "#ef4444",      # Japan
    "HKG": "#f97316",      # Hong Kong
    "ASX": "#14b8a6",      # Australia
    "BSE": "#eab308",      # Bombay
    "SHH": "#ec4899",      # Shanghai
    "FRA": "#8b5cf6",      # Frankfurt
    "XETRA": "#8b5cf6",    # XETRA
    "EURONEXT": "#06b6d4", # Euronext Paris
}

def ticker_color(tickers: list, ticker: str) -> str:
    """
    Return the hex color assigned to a ticker based on its position.

    This keeps one ticker the same color across multiple charts.
    """

    try:
        # Use modulo so the color list cycles if there are more tickers than colors.
        return COLORS[tickers.index(ticker) % len(COLORS)]

    except ValueError:
        # If the ticker is not found, fall back to the first color.
        return COLORS[0]


# ── EMA crossover signal thresholds (%) ──────────────────────
# These values are used by signals.py.
#
# spread    = (EMA10 - EMA26) / EMA26 × 100
# price_gap = (price  - EMA10) / EMA10 × 100
#
# The 0.5% / 1.0% bands filter noise from tiny crossovers.
# "Strong" signals also require current price to confirm the trend.

# Strong buy requires EMA10 to be at least 1% above EMA26.
STRONG_BUY_SPREAD: float = 1.0

# Buy requires EMA10 to be at least 0.5% above EMA26.
BUY_SPREAD: float = 0.5

# Sell requires EMA10 to be at least 0.5% below EMA26.
SELL_SPREAD: float = -0.5

# Strong sell requires EMA10 to be at least 1% below EMA26.
STRONG_SELL_SPREAD: float = -1.0

# Strong signals require price to be at least 2% above/below EMA10.
STRONG_PRICE_GAP: float = 2.0


# ── Signal display labels & badge colors ─────────────────────

# Converts internal signal keys into readable dashboard labels.
SIGNAL_LABELS: dict = {
    "STRONG_BUY": "⬆ Strong Buy",
    "BUY": "↑ Buy",
    "HOLD": "→ Hold",
    "SELL": "↓ Sell",
    "STRONG_SELL": "⬇ Strong Sell",
    "UNKNOWN": "—",
}

# Defines colors for each signal type.
SIGNAL_COLORS: dict = {
    "STRONG_BUY": "#10b981",
    "BUY": "#22c55e",
    "HOLD": "#f59e0b",
    "SELL": "#ef4444",
    "STRONG_SELL": "#dc2626",
    "UNKNOWN": "#6b7280",
}


# ── Alpha Vantage EMA endpoint ────────────────────────────────

# Base URL used by data.py for Alpha Vantage requests.
AV_BASE_URL: str = "https://www.alphavantage.co/query"


# ── Financial Modeling Prep profile endpoint ──────────────────

# Base URL used by data.py for FMP requests.
FMP_BASE_URL: str = "https://financialmodelingprep.com/api/v3"


# ── Approximate exchange centre coordinates for the map ───────
# Falls back to these when FMP data is unavailable.
# Tuple layout:
#   (latitude, longitude, display_name)
#
# These are used by charts.world_map_chart().
EXCHANGE_COORDS: dict = {
    "NMS": (37.33, -121.89, "NASDAQ"),
    "NYQ": (40.71, -74.01, "NYSE"),
    "NGM": (37.33, -121.89, "NASDAQ Global Market"),
    "NCM": (37.33, -121.89, "NASDAQ Capital Market"),
    "LSE": (51.51, -0.09, "London Stock Exchange"),
    "TOR": (43.65, -79.38, "Toronto Stock Exchange"),
    "JPX": (35.69, 139.69, "Japan Exchange Group"),
    "HKG": (22.28, 114.16, "Hong Kong Stock Exchange"),
    "ASX": (-33.87, 151.21, "Australian Securities Exchange"),
    "BSE": (19.08, 72.88, "Bombay Stock Exchange"),
    "SHH": (31.23, 121.47, "Shanghai Stock Exchange"),
    "FRA": (50.11, 8.68, "Frankfurt Stock Exchange"),
    "XETRA": (50.11, 8.68, "XETRA"),
    "EURONEXT": (48.87, 2.33, "Euronext Paris"),
}