# SE370
Project
1. Open a terminal in the project folder.

2. Install all required packages:
   python -m pip install -r requirements.txt

3. Delete the old HQ cache file if it exists:
   hq_locations_cache.json

4. Test the scraper:
   python -m pytest -q test_scrape_locations.py

5. Run the main Streamlit app:
   python -m streamlit run app.py

6. In the app, test with:
   GOOGL, AAPL, PLTR, MU, TSLA (should be pre-loaded in)

7. Confirm PLTR appears on the HQ Locations map using the BeautifulSoup webscraping/geocoding path.

Web-Scrape test run with palantir hq location and yfinance is new import statement required for financial data and external stock APIs
