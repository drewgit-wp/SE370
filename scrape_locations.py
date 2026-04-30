# Replit and ChatGPT used across all functions and concepts for development, troubleshooting, and efficiency
# =============================================================
# scrape_locations.py — Headquarters address + coordinate lookup
# =============================================================
# Purpose:
#   Takes stock ticker symbols already used by MarketTerminal,
#   resolves each ticker to a company name/address, then attempts
#   to return headquarters latitude/longitude for map plotting.
#
# Data source priority:
#   1. Existing FMP profile data passed in from app.py, if available
#   2. Existing Yahoo/yfinance fundamentals passed in from app.py
#   3. Fresh yfinance lookup, only if app.py did not pass fundamentals
#   4. OpenStreetMap Nominatim geocoding of the address/company HQ query
#
#   This file does NOT scrape Google Maps. Google Maps scraping is brittle
#   and may violate terms. Nominatim is used through a normal HTTP request,
#   with caching and throttling so the app does not spam the service.
# =============================================================

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf


# Local cache prevents repeated geocoding on every Streamlit rerun.
# The file is safe to delete; it will rebuild as tickers are requested.
CACHE_FILE = Path("hq_locations_cache.json")

# Nominatim asks clients to identify themselves with a User-Agent.
USER_AGENT = "MarketTerminal-SE370-HQ-Locator/1.0"

# Small pause between geocode calls so repeated tickers do not overload the service.
GEOCODE_SLEEP_SECONDS = 1.1

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"


# ─────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────

def _safe_str(value: Any) -> str:
    """Return a clean string, or an empty string for None/NaN-like values."""
    if value is None:
        return ""

    text = str(value).strip()

    if text.lower() in {"none", "nan", "null"}:
        return ""

    return text


def _safe_float(value: Any) -> float | None:
    """Convert a value to float when possible; otherwise return None."""
    try:
        if value is None or value == "":
            return None

        return float(value)

    except (TypeError, ValueError):
        return None


def _load_cache() -> dict:
    """Read cached ticker-location results from disk."""
    if not CACHE_FILE.exists():
        return {}

    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))

    except (json.JSONDecodeError, OSError):
        return {}


def _save_cache(cache: dict) -> None:
    """Write cached ticker-location results to disk."""
    try:
        CACHE_FILE.write_text(json.dumps(cache, indent=2), encoding="utf-8")

    except OSError:
        # If the cache cannot be written, the app should still work.
        pass


def _build_address(*parts: Any) -> str:
    """
    Build a readable address from parts while removing blanks and duplicates.

    Example:
        _build_address("1 Apple Park Way", "Cupertino", "CA", "95014", "USA")
    """
    cleaned = []
    seen = set()

    for part in parts:
        text = _safe_str(part)

        if not text:
            continue

        # Avoid repeated pieces like "United States, United States".
        key = text.lower()

        if key not in seen:
            cleaned.append(text)
            seen.add(key)

    return ", ".join(cleaned)


# ─────────────────────────────────────────────────────────────
# Profile lookup
# ─────────────────────────────────────────────────────────────

def ticker_to_profile(ticker: str, fundamentals: dict | None = None) -> dict:
    """
    Resolve a ticker into company profile info.

    app.py already fetches fundamentals through data.fetch_all_fundamentals().
    If that dictionary is passed in, this function reuses it instead of making
    a second yfinance request.
    """
    ticker = ticker.upper().strip()
    info = fundamentals or {}

    # Fallback only happens if app.py did not pass fundamentals for this ticker.
    if not info:
        try:
            info = yf.Ticker(ticker).info or {}

        except Exception:
            info = {}

    name = (
        _safe_str(info.get("longName"))
        or _safe_str(info.get("shortName"))
        or ticker
    )

    address = _build_address(
        info.get("address1"),
        info.get("address2"),
        info.get("city"),
        info.get("state"),
        info.get("zip"),
        info.get("country"),
    )

    return {
        "ticker": ticker,
        "name": name,
        "address": address,
        "city": _safe_str(info.get("city")),
        "state": _safe_str(info.get("state")),
        "country": _safe_str(info.get("country")),
        "website": _safe_str(info.get("website")),
    }


def fetch_fmp_profile(ticker: str, fmp_key: str | None = None) -> dict:
    """
    Optional FMP fallback.

    Your current project already has data.fetch_exchange_fmp(), so app.py can
    pass that result into fetch_hq_locations(). This function is here only so
    scrape_locations.py can also run by itself for testing.
    """
    key = fmp_key or os.environ.get("FMP_API_KEY", "")

    if not key:
        return {}

    try:
        response = requests.get(
            f"{FMP_BASE_URL}/profile/{ticker}",
            params={"apikey": key},
            timeout=10,
        )

        response.raise_for_status()
        raw = response.json()

        return raw[0] if isinstance(raw, list) and raw else {}

    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────
# Geocoding
# ─────────────────────────────────────────────────────────────

def geocode_address(query: str) -> dict:
    """
    Convert an address or company-HQ search query into coordinates.

    Returns:
        {
            "lat": float | None,
            "lon": float | None,
            "display_name": str,
            "source": "Nominatim"
        }
    """
    query = _safe_str(query)

    if not query:
        return {
            "lat": None,
            "lon": None,
            "display_name": "",
            "source": "",
        }

    try:
        response = requests.get(
            NOMINATIM_URL,
            params={
                "q": query,
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )

        response.raise_for_status()
        results = response.json()

        if not results:
            return {
                "lat": None,
                "lon": None,
                "display_name": "",
                "source": "Nominatim",
            }

        top = results[0]

        return {
            "lat": _safe_float(top.get("lat")),
            "lon": _safe_float(top.get("lon")),
            "display_name": _safe_str(top.get("display_name")),
            "source": "Nominatim",
        }

    except Exception:
        return {
            "lat": None,
            "lon": None,
            "display_name": "",
            "source": "Nominatim",
        }


def resolve_hq_location(
    ticker: str,
    fundamentals: dict | None = None,
    fmp_profile: dict | None = None,
) -> dict:
    """
    Resolve one ticker to headquarters data.

    Priority:
        1. FMP lat/lng if provided
        2. Geocode FMP address if provided
        3. Geocode yfinance/Yahoo address
        4. Geocode '[company name] headquarters'
    """
    ticker = ticker.upper().strip()
    profile = ticker_to_profile(ticker, fundamentals=fundamentals)
    fmp = fmp_profile or {}

    company_name = (
        _safe_str(fmp.get("companyName"))
        or _safe_str(fmp.get("companyNameLong"))
        or profile["name"]
        or ticker
    )

    # FMP may include actual coordinates. Use them immediately if present.
    fmp_lat = _safe_float(fmp.get("lat"))
    fmp_lon = _safe_float(
        fmp.get("lng")
        or fmp.get("lon")
        or fmp.get("longitude")
    )

    fmp_address = _build_address(
        fmp.get("address"),
        fmp.get("city"),
        fmp.get("state"),
        fmp.get("zip"),
        fmp.get("country"),
    )

    if fmp_lat is not None and fmp_lon is not None:
        return {
            "Ticker": ticker,
            "Name": company_name,
            "Address": fmp_address or profile["address"],
            "City": _safe_str(fmp.get("city")) or profile["city"],
            "State": _safe_str(fmp.get("state")) or profile["state"],
            "Country": _safe_str(fmp.get("country")) or profile["country"],
            "Latitude": fmp_lat,
            "Longitude": fmp_lon,
            "Resolved Address": fmp_address or profile["address"],
            "Source": "FMP profile lat/lng",
            "Found": True,
        }

    # Try strongest address first.
    search_queries = []

    if fmp_address:
        search_queries.append(fmp_address)

    if profile["address"] and profile["address"] not in search_queries:
        search_queries.append(profile["address"])

    # Last fallback: company name + headquarters.
    search_queries.append(f"{company_name} headquarters")

    for query in search_queries:
        geo = geocode_address(query)

        if geo["lat"] is not None and geo["lon"] is not None:
            time.sleep(GEOCODE_SLEEP_SECONDS)

            return {
                "Ticker": ticker,
                "Name": company_name,
                "Address": fmp_address or profile["address"] or query,
                "City": _safe_str(fmp.get("city")) or profile["city"],
                "State": _safe_str(fmp.get("state")) or profile["state"],
                "Country": _safe_str(fmp.get("country")) or profile["country"],
                "Latitude": geo["lat"],
                "Longitude": geo["lon"],
                "Resolved Address": geo["display_name"],
                "Source": f"Geocoded: {geo['source']}",
                "Found": True,
            }

        time.sleep(GEOCODE_SLEEP_SECONDS)

    return {
        "Ticker": ticker,
        "Name": company_name,
        "Address": fmp_address or profile["address"],
        "City": _safe_str(fmp.get("city")) or profile["city"],
        "State": _safe_str(fmp.get("state")) or profile["state"],
        "Country": _safe_str(fmp.get("country")) or profile["country"],
        "Latitude": None,
        "Longitude": None,
        "Resolved Address": "",
        "Source": "Not found",
        "Found": False,
    }


# ─────────────────────────────────────────────────────────────
# Public function for app.py
# ─────────────────────────────────────────────────────────────

def fetch_hq_locations(
    tickers: list[str] | tuple[str, ...],
    fundamentals: dict | None = None,
    fmp_data: dict | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """
    Main function used by app.py.

    Args:
        tickers:
            Tickers already being used in the dashboard.

        fundamentals:
            Existing dictionary from data.fetch_all_fundamentals(tuple(tickers)).
            Shape:
                {
                    "AAPL": {...},
                    "MSFT": {...}
                }

        fmp_data:
            Existing dictionary from data.fetch_exchange_fmp(ticker).
            Shape:
                {
                    "AAPL": {...},
                    "MSFT": {...}
                }

        refresh:
            If True, ignore hq_locations_cache.json and re-check online.

    Returns:
        DataFrame with:
            Ticker,
            Name,
            Address,
            City,
            State,
            Country,
            Latitude,
            Longitude,
            Resolved Address,
            Source,
            Found
    """
    clean_tickers = []

    for ticker in tickers:
        text = _safe_str(ticker).upper()

        if text and text not in clean_tickers:
            clean_tickers.append(text)

    cache = {} if refresh else _load_cache()
    rows = []

    fundamentals = fundamentals or {}
    fmp_data = fmp_data or {}

    for ticker in clean_tickers:
        if not refresh and ticker in cache:
            rows.append(cache[ticker])
            continue

        # Use already-fetched app.py data when available.
        info = fundamentals.get(ticker, {})
        fmp = fmp_data.get(ticker, {})

        # If no FMP data was passed, optionally try direct FMP lookup.
        if not fmp:
            fmp = fetch_fmp_profile(ticker)

        row = resolve_hq_location(
            ticker,
            fundamentals=info,
            fmp_profile=fmp,
        )

        cache[ticker] = row
        rows.append(row)

    _save_cache(cache)

    columns = [
        "Ticker",
        "Name",
        "Address",
        "City",
        "State",
        "Country",
        "Latitude",
        "Longitude",
        "Resolved Address",
        "Source",
        "Found",
    ]

    return pd.DataFrame(rows, columns=columns)


# ─────────────────────────────────────────────────────────────
# Manual test mode
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Run this file directly with:
    #   python scrape_locations.py
    #
    # This lets you test the file without launching Streamlit.
    sample = ["AAPL", "GOOGL", "TSLA", "MSFT"]

    df = fetch_hq_locations(sample)

    print(df.to_string(index=False))