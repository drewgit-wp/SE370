"""Tests for scrape_locations.py.

Run with:
    python -m pytest test_scrape_locations.py
"""

import scrape_locations


def test_default_locations_resolve_and_pltr_uses_html_scrape():
    """Default tickers should resolve, with PLTR forced through HTML scraping."""
    df = scrape_locations._run_default_location_test()

    assert len(df) == 5
    assert df["Found"].all()

    pltr = df.loc[df["Ticker"] == "PLTR"].iloc[0]
    assert pltr["HTML Scrape Used"] is True or bool(pltr["HTML Scrape Used"]) is True
    assert "Official website HTML scrape" in pltr["Source"]
    assert "1555 Blake" in pltr["Scraped Query"]

    other_tickers = df[df["Ticker"].isin(["GOOGL", "AAPL", "MU", "TSLA"])]
    assert not other_tickers["HTML Scrape Used"].any()
    assert set(other_tickers["Lookup Method"]) == {"FMP profile coordinates"}


def test_extract_address_candidate_from_html_text():
    """The HTML-text extraction helper should pull a street-address candidate."""
    text = """
    Contact Us | Palantir Technologies
    Corporate Headquarters: 1555 Blake Street, Denver, CO 80202, United States.
    Privacy Policy Terms of Use
    """

    candidate = scrape_locations._extract_address_candidate(text, "Palantir Technologies Inc.")

    assert "1555 Blake Street" in candidate
    assert "Denver" in candidate