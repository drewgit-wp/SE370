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
#   1. Forced official-site HTML scrape for selected tickers, currently PLTR
#   2. Existing FMP profile latitude/longitude, if available
#   3. Existing FMP profile address, then Nominatim geocoding
#   4. Existing Yahoo/yfinance fundamentals address, then Nominatim geocoding
#   5. Official-site HTML scrape fallback, then Nominatim geocoding
#   6. Final company-name headquarters geocoding fallback
#
# Important:
#   This file does NOT scrape Google Maps. Google Maps scraping is brittle
#   and may violate terms. The webscraping fallback only reads public HTML
#   from a company's official website, then validates the result by geocoding
#   it through OpenStreetMap Nominatim.
# =============================================================

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import pandas as pd
import requests
from bs4 import BeautifulSoup

try:
    import yfinance as yf
except Exception:  # lets the test mode run even if yfinance is not installed yet
    yf = None


# Local cache prevents repeated geocoding on every Streamlit rerun.
# The file is safe to delete; it will rebuild as tickers are requested.
CACHE_FILE = Path("hq_locations_cache.json")

# Nominatim asks clients to identify themselves with a User-Agent.
USER_AGENT = "MarketTerminal-SE370-HQ-Locator/1.0"

# Small pause between geocode calls so repeated tickers do not overload the service.
GEOCODE_SLEEP_SECONDS = 1.1

# Web endpoints used by this file.
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"

# HTML scraping settings.
HTML_SCRAPE_TIMEOUT_SECONDS = 10
HTML_SCRAPE_SLEEP_SECONDS = 0.4

SCRAPE_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Your default dashboard tickers from config.py.
DEFAULT_LOCATION_TEST_TICKERS = ["GOOGL", "AAPL", "PLTR", "MU", "TSLA"]

# PLTR is forced through the HTML scrape path so the project demonstrates
# real webscraping in the default watchlist.
DEFAULT_FORCE_HTML_SCRAPE_TICKERS = {"PLTR"}

# Common official-site pages that may contain headquarters/contact addresses.
CONTACT_PATHS = [
    "",
    "/contact",
    "/contact-us",
    "/about",
    "/about-us",
    "/company",
    "/investors",
    "/investor-relations",
    "/who-we-are",
]

ADDRESS_KEYWORDS = [
    "headquarters",
    "global headquarters",
    "corporate headquarters",
    "corporate office",
    "head office",
    "office address",
    "mailing address",
    "contact us",
    "address",
]

ADDRESS_STREET_WORDS = (
    "Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Way|Drive|Dr|Lane|Ln|"
    "Parkway|Pkwy|Place|Pl|Circle|Cir|Court|Ct|Park|Plaza|Square|Sq|Trail|Trl"
)


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

        key = text.lower()

        if key not in seen:
            cleaned.append(text)
            seen.add(key)

    return ", ".join(cleaned)


def _clean_page_text(text: str) -> str:
    """Normalize visible text from an HTML page."""
    text = re.sub(r"\s+", " ", _safe_str(text))
    text = text.replace(" ,", ",").replace(" .", ".")
    return text.strip()


def _cache_key(ticker: str, force_html_scrape: bool) -> str:
    """
    Include the scrape mode in the cache key.

    This prevents an older non-scraped PLTR cache row from hiding the forced
    HTML scrape behavior when you rerun Streamlit.
    """
    suffix = "html" if force_html_scrape else "normal"
    return f"{ticker.upper().strip()}::{suffix}"


# ─────────────────────────────────────────────────────────────
# Streamlit / console notice helpers
# ─────────────────────────────────────────────────────────────

def _show_html_scrape_notice(
    ticker: str,
    company_name: str,
    source_url: str,
    address_candidate: str,
    show_scrape_messages: bool = True,
) -> None:
    """
    Show a visible text block when official-site HTML scraping is used.

    In Streamlit, this appears as an info block. When the file is run directly
    from the terminal, it prints the same message.
    """
    if not show_scrape_messages:
        return

    message = (
        f"🌐 HTML webscrape used for {ticker} ({company_name}). "
        f"Scraped location candidate: {address_candidate}. "
        f"Source page: {source_url}"
    )

    try:
        import streamlit as st

        st.info(message)
    except Exception:
        print(message)


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
    if not info and yf is not None:
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


# ─────────────────────────────────────────────────────────────
# Official-site HTML webscraping fallback
# ─────────────────────────────────────────────────────────────

def _allowed_by_robots(url: str) -> bool:
    """
    Check robots.txt before scraping a page.

    If robots.txt cannot be reached, this returns True so class-project usage
    does not fail just because a site blocks or times out on robots.txt.
    """
    try:
        parsed = urlparse(url)

        if not parsed.scheme or not parsed.netloc:
            return False

        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        parser = RobotFileParser()
        parser.set_url(robots_url)
        parser.read()

        return parser.can_fetch(USER_AGENT, url)

    except Exception:
        return True


def _extract_json_ld_addresses(soup: BeautifulSoup) -> list[str]:
    """
    Extract address-like values from JSON-LD script blocks.

    Many corporate websites include structured Organization data in JSON-LD.
    This is still HTML scraping because the JSON-LD is embedded in the page HTML.
    """
    addresses: list[str] = []

    def visit(obj: Any) -> None:
        if isinstance(obj, dict):
            address = obj.get("address")

            if isinstance(address, dict):
                candidate = _build_address(
                    address.get("streetAddress"),
                    address.get("addressLocality"),
                    address.get("addressRegion"),
                    address.get("postalCode"),
                    address.get("addressCountry"),
                )

                if candidate:
                    addresses.append(candidate)

            elif isinstance(address, str):
                candidate = _clean_page_text(address)

                if candidate:
                    addresses.append(candidate)

            for value in obj.values():
                visit(value)

        elif isinstance(obj, list):
            for item in obj:
                visit(item)

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or script.get_text() or "")
            visit(data)
        except Exception:
            continue

    return addresses


def _extract_address_candidate(text: str, company_name: str = "") -> str:
    """
    Try to extract a likely headquarters/contact address from visible page text.

    The logic intentionally stays simple and explainable:
      1. Look near headquarters/contact/address keywords.
      2. Prefer snippets that contain a street-looking phrase.
      3. Fall back to a broader street-address regex.
    """
    clean_text = _clean_page_text(text)

    if not clean_text:
        return ""

    lower_text = clean_text.lower()
    best_candidates: list[str] = []

    for keyword in ADDRESS_KEYWORDS:
        start_index = 0

        while True:
            index = lower_text.find(keyword, start_index)

            if index == -1:
                break

            snippet = clean_text[index:index + 450]
            snippet = re.split(
                r"(privacy policy|terms of use|cookie|©|all rights reserved|subscribe|newsletter)",
                snippet,
                flags=re.IGNORECASE,
            )[0]
            snippet = snippet.strip(" :-|•")

            if re.search(ADDRESS_STREET_WORDS, snippet, flags=re.IGNORECASE):
                best_candidates.append(snippet)

            start_index = index + len(keyword)

    if best_candidates:
        # Shorter keyword snippets usually geocode better than huge chunks.
        return min(best_candidates, key=len)

    street_regex = re.compile(
        rf"\b\d{{1,6}}\s+[A-Za-z0-9 .'\-]+\s+(?:{ADDRESS_STREET_WORDS})\b"
        r"(?:[,\s]+[A-Za-z .'-]+){0,5}"
        r"(?:\b[A-Z]{2}\b)?"
        r"(?:\s+\d{5}(?:-\d{4})?)?",
        flags=re.IGNORECASE,
    )

    match = street_regex.search(clean_text)

    if match:
        return match.group(0).strip(" ,.-")

    return ""


def scrape_official_site_for_hq(
    company_name: str,
    website: str,
    respect_robots: bool = True,
) -> dict:
    """
    Scrape a company's official website for headquarters/contact address text.

    Returns:
        {
            "address": str,
            "source_url": str,
            "source": "Official website HTML scrape",
            "scraped": bool
        }

    This is true webscraping because it downloads HTML, parses the page with
    BeautifulSoup, extracts visible text and JSON-LD, then searches for address
    clues that can be validated by geocoding.
    """
    website = _safe_str(website)

    if not website:
        return {"address": "", "source_url": "", "source": "", "scraped": False}

    if not website.startswith(("http://", "https://")):
        website = "https://" + website

    candidate_urls = []

    for path in CONTACT_PATHS:
        url = urljoin(website.rstrip("/") + "/", path.lstrip("/"))

        if url not in candidate_urls:
            candidate_urls.append(url)

    for url in candidate_urls:
        if respect_robots and not _allowed_by_robots(url):
            continue

        try:
            response = requests.get(
                url,
                headers=SCRAPE_HEADERS,
                timeout=HTML_SCRAPE_TIMEOUT_SECONDS,
            )

            if response.status_code != 200:
                continue

            content_type = response.headers.get("content-type", "").lower()

            if "text/html" not in content_type and content_type:
                continue

            soup = BeautifulSoup(response.text, "lxml")

            json_ld_addresses = _extract_json_ld_addresses(soup)

            if json_ld_addresses:
                return {
                    "address": json_ld_addresses[0],
                    "source_url": url,
                    "source": "Official website HTML scrape",
                    "scraped": True,
                }

            for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
                tag.decompose()

            visible_text = soup.get_text(separator=" ", strip=True)
            address_candidate = _extract_address_candidate(visible_text, company_name)

            if address_candidate:
                return {
                    "address": address_candidate,
                    "source_url": url,
                    "source": "Official website HTML scrape",
                    "scraped": True,
                }

            time.sleep(HTML_SCRAPE_SLEEP_SECONDS)

        except Exception:
            continue

    return {"address": "", "source_url": "", "source": "", "scraped": False}


# ─────────────────────────────────────────────────────────────
# HQ resolution
# ─────────────────────────────────────────────────────────────

def _row(
    ticker: str,
    company_name: str,
    address: str,
    city: str,
    state: str,
    country: str,
    latitude: float | None,
    longitude: float | None,
    resolved_address: str,
    source: str,
    found: bool,
    lookup_method: str,
    html_scrape_used: bool = False,
    scraped_url: str = "",
    scraped_query: str = "",
) -> dict:
    """Return one standardized output row."""
    return {
        "Ticker": ticker,
        "Name": company_name,
        "Address": address,
        "City": city,
        "State": state,
        "Country": country,
        "Latitude": latitude,
        "Longitude": longitude,
        "Resolved Address": resolved_address,
        "Source": source,
        "Lookup Method": lookup_method,
        "HTML Scrape Used": html_scrape_used,
        "Scraped URL": scraped_url,
        "Scraped Query": scraped_query,
        "Found": found,
    }


def _try_geocode_query(
    query: str,
    ticker: str,
    company_name: str,
    city: str,
    state: str,
    country: str,
    source_label: str,
    lookup_method: str,
    html_scrape_used: bool = False,
    scraped_url: str = "",
    scraped_query: str = "",
) -> dict | None:
    """Geocode one query and return a row if coordinates are found."""
    query = _safe_str(query)

    if not query:
        return None

    geo = geocode_address(query)

    if geo["lat"] is None or geo["lon"] is None:
        time.sleep(GEOCODE_SLEEP_SECONDS)
        return None

    time.sleep(GEOCODE_SLEEP_SECONDS)

    return _row(
        ticker=ticker,
        company_name=company_name,
        address=query,
        city=city,
        state=state,
        country=country,
        latitude=geo["lat"],
        longitude=geo["lon"],
        resolved_address=geo["display_name"],
        source=source_label,
        found=True,
        lookup_method=lookup_method,
        html_scrape_used=html_scrape_used,
        scraped_url=scraped_url,
        scraped_query=scraped_query,
    )


def _try_html_scrape_then_geocode(
    ticker: str,
    company_name: str,
    website: str,
    city: str,
    state: str,
    country: str,
    show_scrape_messages: bool = True,
) -> dict | None:
    """Scrape the official website for an address, then validate it by geocoding."""
    scraped = scrape_official_site_for_hq(company_name=company_name, website=website)

    if not scraped.get("address"):
        return None

    address_candidate = scraped["address"]
    source_url = scraped.get("source_url", "")

    _show_html_scrape_notice(
        ticker=ticker,
        company_name=company_name,
        source_url=source_url,
        address_candidate=address_candidate,
        show_scrape_messages=show_scrape_messages,
    )

    return _try_geocode_query(
        query=address_candidate,
        ticker=ticker,
        company_name=company_name,
        city=city,
        state=state,
        country=country,
        source_label="Official website HTML scrape + Nominatim geocode",
        lookup_method="Official-site HTML scrape",
        html_scrape_used=True,
        scraped_url=source_url,
        scraped_query=address_candidate,
    )


def resolve_hq_location(
    ticker: str,
    fundamentals: dict | None = None,
    fmp_profile: dict | None = None,
    force_html_scrape: bool = False,
    show_scrape_messages: bool = True,
) -> dict:
    """
    Resolve one ticker to headquarters data.

    Priority:
        1. Forced official-site HTML scrape for selected tickers, currently PLTR
        2. FMP lat/lng if provided
        3. Geocode FMP address if provided
        4. Geocode yfinance/Yahoo address
        5. Official-site HTML scrape fallback
        6. Geocode '[company name] headquarters'
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

    fmp_address = _build_address(
        fmp.get("address"),
        fmp.get("city"),
        fmp.get("state"),
        fmp.get("zip"),
        fmp.get("country"),
    )

    city = _safe_str(fmp.get("city")) or profile["city"]
    state = _safe_str(fmp.get("state")) or profile["state"]
    country = _safe_str(fmp.get("country")) or profile["country"]
    website = profile.get("website", "") or _safe_str(fmp.get("website"))

    # Force selected tickers, such as PLTR, through actual HTML scraping first.
    if force_html_scrape and website:
        scraped_row = _try_html_scrape_then_geocode(
            ticker=ticker,
            company_name=company_name,
            website=website,
            city=city,
            state=state,
            country=country,
            show_scrape_messages=show_scrape_messages,
        )

        if scraped_row is not None:
            return scraped_row

    # FMP may include actual coordinates. Use them immediately if present.
    fmp_lat = _safe_float(fmp.get("lat"))
    fmp_lon = _safe_float(
        fmp.get("lng")
        or fmp.get("lon")
        or fmp.get("longitude")
    )

    if fmp_lat is not None and fmp_lon is not None:
        return _row(
            ticker=ticker,
            company_name=company_name,
            address=fmp_address or profile["address"],
            city=city,
            state=state,
            country=country,
            latitude=fmp_lat,
            longitude=fmp_lon,
            resolved_address=fmp_address or profile["address"],
            source="FMP profile lat/lng",
            found=True,
            lookup_method="FMP profile coordinates",
        )

    # Strongest structured address source first.
    if fmp_address:
        row = _try_geocode_query(
            query=fmp_address,
            ticker=ticker,
            company_name=company_name,
            city=city,
            state=state,
            country=country,
            source_label="FMP address + Nominatim geocode",
            lookup_method="FMP address geocode",
        )

        if row is not None:
            return row

    if profile["address"] and profile["address"] != fmp_address:
        row = _try_geocode_query(
            query=profile["address"],
            ticker=ticker,
            company_name=company_name,
            city=city,
            state=state,
            country=country,
            source_label="Yahoo/yfinance address + Nominatim geocode",
            lookup_method="Yahoo/yfinance address geocode",
        )

        if row is not None:
            return row

    # Non-forced scrape fallback for tickers with missing structured addresses.
    if not force_html_scrape and website:
        scraped_row = _try_html_scrape_then_geocode(
            ticker=ticker,
            company_name=company_name,
            website=website,
            city=city,
            state=state,
            country=country,
            show_scrape_messages=show_scrape_messages,
        )

        if scraped_row is not None:
            return scraped_row

    # Last fallback: geocode the company name + headquarters.
    fallback_query = f"{company_name} headquarters"
    row = _try_geocode_query(
        query=fallback_query,
        ticker=ticker,
        company_name=company_name,
        city=city,
        state=state,
        country=country,
        source_label="Company headquarters query + Nominatim geocode",
        lookup_method="Company-name headquarters geocode",
    )

    if row is not None:
        return row

    return _row(
        ticker=ticker,
        company_name=company_name,
        address=fmp_address or profile["address"] or fallback_query,
        city=city,
        state=state,
        country=country,
        latitude=None,
        longitude=None,
        resolved_address="",
        source="Not found",
        found=False,
        lookup_method="No location source resolved",
    )


# ─────────────────────────────────────────────────────────────
# Public function for app.py
# ─────────────────────────────────────────────────────────────

def fetch_hq_locations(
    tickers: list[str] | tuple[str, ...],
    fundamentals: dict | None = None,
    fmp_data: dict | None = None,
    refresh: bool = False,
    show_scrape_messages: bool = True,
    force_html_scrape_tickers: set[str] | list[str] | tuple[str, ...] | None = None,
    use_cache: bool = True,
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

        show_scrape_messages:
            If True, show a Streamlit info block when HTML scraping is used.

        force_html_scrape_tickers:
            Tickers that should try official-site HTML scraping before structured
            sources. Defaults to {"PLTR"} so Palantir demonstrates scraping.

        use_cache:
            If False, skip reading/writing hq_locations_cache.json. Useful for tests.

    Returns:
        DataFrame with basic map columns plus scrape-tracking columns:
            Ticker, Name, Address, City, State, Country,
            Latitude, Longitude, Resolved Address, Source,
            Lookup Method, HTML Scrape Used, Scraped URL, Scraped Query, Found
    """
    clean_tickers = []

    for ticker in tickers:
        text = _safe_str(ticker).upper()

        if text and text not in clean_tickers:
            clean_tickers.append(text)

    force_set = {
        _safe_str(t).upper()
        for t in (force_html_scrape_tickers or DEFAULT_FORCE_HTML_SCRAPE_TICKERS)
        if _safe_str(t)
    }

    cache = {} if refresh or not use_cache else _load_cache()
    rows = []

    fundamentals = fundamentals or {}
    fmp_data = fmp_data or {}

    for ticker in clean_tickers:
        force_html_scrape = ticker in force_set
        key = _cache_key(ticker, force_html_scrape)

        if use_cache and not refresh and key in cache:
            rows.append(cache[key])
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
            force_html_scrape=force_html_scrape,
            show_scrape_messages=show_scrape_messages,
        )

        if use_cache:
            cache[key] = row

        rows.append(row)

    if use_cache:
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
        "Lookup Method",
        "HTML Scrape Used",
        "Scraped URL",
        "Scraped Query",
        "Found",
    ]

    return pd.DataFrame(rows, columns=columns)


# ─────────────────────────────────────────────────────────────
# Built-in default-location test
# ─────────────────────────────────────────────────────────────

def _run_default_location_test() -> pd.DataFrame:
    """
    Offline test for the default watchlist.

    Expected behavior:
      - GOOGL, AAPL, MU, TSLA resolve from structured FMP coordinates.
      - PLTR resolves through the official-site HTML scrape path.

    This test uses mock data so it does not depend on live internet access.
    """
    global scrape_official_site_for_hq, geocode_address, fetch_fmp_profile

    original_scrape = scrape_official_site_for_hq
    original_geocode = geocode_address
    original_fetch_fmp_profile = fetch_fmp_profile

    fundamentals = {
        "GOOGL": {
            "longName": "Alphabet Inc.",
            "website": "https://abc.xyz",
        },
        "AAPL": {
            "longName": "Apple Inc.",
            "website": "https://www.apple.com",
        },
        "PLTR": {
            "longName": "Palantir Technologies Inc.",
            "website": "https://www.palantir.com",
        },
        "MU": {
            "longName": "Micron Technology, Inc.",
            "website": "https://www.micron.com",
        },
        "TSLA": {
            "longName": "Tesla, Inc.",
            "website": "https://www.tesla.com",
        },
    }

    fmp_data = {
        "GOOGL": {
            "companyName": "Alphabet Inc.",
            "address": "1600 Amphitheatre Parkway",
            "city": "Mountain View",
            "state": "CA",
            "zip": "94043",
            "country": "US",
            "lat": 37.4220,
            "lng": -122.0841,
        },
        "AAPL": {
            "companyName": "Apple Inc.",
            "address": "One Apple Park Way",
            "city": "Cupertino",
            "state": "CA",
            "zip": "95014",
            "country": "US",
            "lat": 37.3349,
            "lng": -122.0090,
        },
        "MU": {
            "companyName": "Micron Technology, Inc.",
            "address": "8000 South Federal Way",
            "city": "Boise",
            "state": "ID",
            "zip": "83716",
            "country": "US",
            "lat": 43.5300,
            "lng": -116.1500,
        },
        "TSLA": {
            "companyName": "Tesla, Inc.",
            "address": "1 Tesla Road",
            "city": "Austin",
            "state": "TX",
            "zip": "78725",
            "country": "US",
            "lat": 30.2223,
            "lng": -97.6171,
        },
        # PLTR intentionally has no FMP coordinates/address here so the test
        # proves the HTML webscrape branch is used.
        "PLTR": {},
    }

    def fake_scrape(company_name: str, website: str, respect_robots: bool = True) -> dict:
        if "palantir" in company_name.lower() or "palantir" in website.lower():
            return {
                "address": "1555 Blake Street, Denver, CO 80202, United States",
                "source_url": "https://www.palantir.com/contact/",
                "source": "Official website HTML scrape",
                "scraped": True,
            }

        return {"address": "", "source_url": "", "source": "", "scraped": False}

    def fake_geocode(query: str) -> dict:
        if "1555 Blake" in query:
            return {
                "lat": 39.7502,
                "lon": -104.9970,
                "display_name": "1555 Blake Street, Denver, CO 80202, United States",
                "source": "Mock Nominatim",
            }

        return {
            "lat": None,
            "lon": None,
            "display_name": "",
            "source": "Mock Nominatim",
        }

    def fake_fetch_fmp_profile(ticker: str, fmp_key: str | None = None) -> dict:
        return {}

    scrape_official_site_for_hq = fake_scrape
    geocode_address = fake_geocode
    fetch_fmp_profile = fake_fetch_fmp_profile

    try:
        df = fetch_hq_locations(
            DEFAULT_LOCATION_TEST_TICKERS,
            fundamentals=fundamentals,
            fmp_data=fmp_data,
            refresh=True,
            show_scrape_messages=False,
            force_html_scrape_tickers={"PLTR"},
            use_cache=False,
        )

        assert len(df) == 5, "Default test should return five rows."
        assert df["Found"].all(), "All five default tickers should resolve."

        pltr = df.loc[df["Ticker"] == "PLTR"].iloc[0]
        assert bool(pltr["HTML Scrape Used"]) is True, "PLTR should use HTML scraping."
        assert "Official website HTML scrape" in pltr["Source"], "PLTR source should show HTML scrape."
        assert "1555 Blake" in pltr["Scraped Query"], "PLTR scrape query should show the scraped location."

        structured = df[df["Ticker"].isin(["GOOGL", "AAPL", "MU", "TSLA"])]
        assert not structured["HTML Scrape Used"].any(), "GOOGL/AAPL/MU/TSLA should not use HTML scraping in this test."
        assert set(structured["Lookup Method"]) == {"FMP profile coordinates"}, "Other defaults should use structured FMP coordinates."

        return df

    finally:
        scrape_official_site_for_hq = original_scrape
        geocode_address = original_geocode
        fetch_fmp_profile = original_fetch_fmp_profile


# ─────────────────────────────────────────────────────────────
# Manual test mode
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Offline mock test:
    #   python scrape_locations.py
    #
    # Live test using real API/geocoding calls:
    #   python scrape_locations.py --live
    #
    # In the mock test, GOOGL/AAPL/MU/TSLA use structured coordinates and
    # PLTR is forced through the official-site HTML scrape branch.
    if "--live" in sys.argv:
        df_live = fetch_hq_locations(
            DEFAULT_LOCATION_TEST_TICKERS,
            refresh=True,
            show_scrape_messages=True,
            force_html_scrape_tickers={"PLTR"},
            use_cache=False,
        )
        print(df_live.to_string(index=False))
    else:
        df_test = _run_default_location_test()
        print("Default scrape_locations.py test passed.\n")
        print(df_test[[
            "Ticker",
            "Name",
            "Address",
            "Latitude",
            "Longitude",
            "Lookup Method",
            "HTML Scrape Used",
            "Scraped URL",
            "Found",
        ]].to_string(index=False))
