"""
Scrapes RBI's Monetary Policy page (rbi.org.in/scripts/annualpolicy.aspx) for
links to each MPC meeting's published minutes, then fetches and saves the
full text of each one.

WHY PLAYWRIGHT: the year tabs (2016-2017, 2017-2018, etc.) on this page load
their content via a client-side AJAX postback when clicked -- a plain
requests.get() only ever sees whichever year is loaded by default. Playwright
drives a real (headless) browser so it can click the tab and wait for the
new content before reading the page.

THIS SCRIPT IS NOT TESTED END-TO-END: the sandbox this was written in cannot
reach rbi.org.in (network egress restriction) or even install a Playwright
browser binary. Expect to debug selectors against the real page -- if a
`page.get_by_text(...)` call can't find something, the most common fix is
adjusting the visible text it's matching (RBI's exact label text may differ
slightly, e.g. punctuation/spacing) or adding a small `page.wait_for_timeout()`
if content hasn't finished rendering yet.

Run standalone:
    python -m src.scrape_minutes
"""

import json
import logging
import re
import time
from datetime import datetime, timedelta

import pandas as pd
from playwright.sync_api import sync_playwright

from src import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MONETARY_POLICY_URL = "https://www.rbi.org.in/scripts/annualpolicy.aspx"
MINUTES_DIR = config.DATA_DIR / "mpc_minutes"
LINKS_CACHE_FILE = config.DATA_DIR / "mpc_minutes_links_raw.json"
INDEX_FILE = config.DATA_DIR / "mpc_minutes_index.csv"

# Match tolerance: the scraped link text usually gives the meeting's last day
# (e.g. "April 6 to 8, 2022" -> April 8), which should be very close to (often
# exactly) the announcement date recorded in processed_cpi_mpc.csv.
DATE_MATCH_TOLERANCE_DAYS = 5

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
# Captures "<Month> <anything> <year>" non-greedily -- the middle group holds
# all the day numbers and connecting words ("6 to 8", "24, 26 and 27", "2 and 4"),
# which parse_last_date() then extracts the day numbers from directly rather
# than trying to match every possible "to"/"and"/"," phrasing in the regex itself.
DATE_RANGE_PATTERN = re.compile(rf"({MONTHS})\s+(.*?),?\s*(\d{{4}})")

# Boilerplate lines that show up on every RBI page (nav, accessibility
# widgets, footer) -- stripped out of the extracted minutes text since they
# add noise without adding meaning for the LLM comparison step.
BOILERPLATE_LINE_PATTERNS = [
    "Skip to main content", "Change Language", "हिंदी", "Search the Website",
    "Beti Bachao Beti Padhao", "Follow RBI", "Sitemap", "Disclaimer",
    "Accessibility Statement", "Website last updated", "Supports: Google Chrome",
    "Increase Letter Spacing", "Font Size", "Dark Theme", "Print this page",
    "© Reserve Bank of India",
]


def financial_years_needed(df_mpc: pd.DataFrame) -> list[str]:
    """India's fiscal year runs April-March. Returns e.g. ['2016-2017', ...]
    covering every meeting date in the dataset."""
    years = set()
    for d in df_mpc["date"]:
        fy_start = d.year if d.month >= 4 else d.year - 1
        years.add(f"{fy_start}-{fy_start + 1}")
    return sorted(years)


def discover_minutes_links(fiscal_years: list[str]) -> dict:
    """Drive a headless browser to click each fiscal-year tab and collect
    every 'Minutes of the Monetary Policy Committee' link and its text.

    Returns {fiscal_year: [{"text": ..., "href": ...}, ...]}.
    Caches to LINKS_CACHE_FILE after each year so a crash partway through
    doesn't lose earlier progress.
    """
    results = {}
    if LINKS_CACHE_FILE.exists():
        results = json.loads(LINKS_CACHE_FILE.read_text())
        logger.info("Loaded %d cached fiscal years from %s", len(results), LINKS_CACHE_FILE)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(MONETARY_POLICY_URL, wait_until="networkidle")

        for fy in fiscal_years:
            if fy in results:
                logger.info("Skipping %s (already cached)", fy)
                continue
            try:
                # The current fiscal year's content is already loaded by
                # default; older years need their tab clicked. Some very old
                # years may sit inside an "Archives" expandable section.
                locator = page.get_by_text(fy, exact=True)
                if locator.count() == 0:
                    logger.info("'%s' not directly visible -- trying to expand Archives first", fy)
                    archives_toggle = page.get_by_text("Archives", exact=False)
                    if archives_toggle.count() > 0:
                        archives_toggle.first.click()
                        page.wait_for_timeout(1000)
                    locator = page.get_by_text(fy, exact=True)

                if locator.count() == 0:
                    logger.warning("Could not find a tab for fiscal year %s -- skipping", fy)
                    results[fy] = []
                    continue

                locator.first.click()
                page.wait_for_timeout(2000)  # AJAX panel render time
                try:
                    page.wait_for_load_state("networkidle", timeout=8000)
                except Exception:
                    pass  # some postbacks don't trigger a clean networkidle; the fixed wait above is the real safeguard

                links = page.locator("a", has_text="Minutes of the Monetary Policy Committee")
                fy_results = []
                for i in range(links.count()):
                    el = links.nth(i)
                    fy_results.append({"text": el.inner_text(), "href": el.get_attribute("href")})

                results[fy] = fy_results
                logger.info("Fiscal year %s: found %d minutes links", fy, len(fy_results))

            except Exception as exc:
                logger.warning("Failed on fiscal year %s (%s) -- skipping, will retry on next run", fy, exc)
                continue

            LINKS_CACHE_FILE.write_text(json.dumps(results, indent=2))
            time.sleep(1.5)  # be polite to RBI's servers between year clicks

        browser.close()

    return results


def parse_last_date(text: str) -> datetime | None:
    """Extract the last (latest) calendar date mentioned in a minutes link's
    text, e.g. "April 6 to 8, 2022" -> April 8, 2022;
    "March 24, 26 and 27, 2020" -> March 27, 2020.
    """
    match = None
    for match in DATE_RANGE_PATTERN.finditer(text):
        pass  # keep the last (month, day-phrase, year) match found in the string
    if match is None:
        return None

    month_name, day_phrase, year = match.groups()
    day_numbers = [int(d) for d in re.findall(r"\d{1,2}", day_phrase)]
    if not day_numbers:
        return None
    last_day = max(day_numbers)

    try:
        return datetime.strptime(f"{month_name} {last_day} {year}", "%B %d %Y")
    except ValueError:
        return None


def match_links_to_meetings(raw_links: dict, df_mpc: pd.DataFrame) -> pd.DataFrame:
    """Match each scraped link to the closest known meeting date in
    processed_cpi_mpc.csv, within DATE_MATCH_TOLERANCE_DAYS."""
    rows = []
    unmatched = []
    known_dates = df_mpc["date"].tolist()

    for fy, links in raw_links.items():
        for link in links:
            parsed_date = parse_last_date(link["text"])
            if parsed_date is None:
                unmatched.append(link)
                continue

            closest = min(known_dates, key=lambda d: abs((d - parsed_date).days))
            if abs((closest - parsed_date).days) <= DATE_MATCH_TOLERANCE_DAYS:
                rows.append({
                    "meeting_date": closest,
                    "scraped_text": link["text"],
                    "url": link["href"],
                })
            else:
                unmatched.append(link)

    if unmatched:
        logger.warning(
            "%d scraped links could not be matched to a known meeting date within %d days -- "
            "review these manually (likely off-cycle meetings, date-format edge cases, or "
            "meetings outside processed_cpi_mpc.csv's range).",
            len(unmatched), DATE_MATCH_TOLERANCE_DAYS,
        )
        for u in unmatched:
            logger.warning("  Unmatched: %s -> %s", u["text"], u["href"])

    return pd.DataFrame(rows)


def clean_minutes_text(raw_text: str) -> str:
    lines = [line.strip() for line in raw_text.split("\n")]
    lines = [
        line for line in lines
        if line and not any(bp.lower() in line.lower() for bp in BOILERPLATE_LINE_PATTERNS)
    ]
    return "\n".join(lines)


def fetch_and_save_minutes_text(matched: pd.DataFrame) -> None:
    """Visit each matched minutes URL and save its cleaned text to disk."""
    MINUTES_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        for _, row in matched.iterrows():
            date_str = pd.Timestamp(row["meeting_date"]).strftime("%Y-%m-%d")
            out_path = MINUTES_DIR / f"{date_str}.txt"
            if out_path.exists():
                continue
            try:
                page.goto(row["url"], wait_until="networkidle", timeout=20000)
                raw_text = page.inner_text("body")
                cleaned = clean_minutes_text(raw_text)
                out_path.write_text(cleaned, encoding="utf-8")
                logger.info("Saved minutes text for %s (%d chars) -> %s", date_str, len(cleaned), out_path)
            except Exception as exc:
                logger.warning("Failed to fetch minutes for %s (%s): %s", date_str, row["url"], exc)
            time.sleep(1.5)

        browser.close()


def run() -> pd.DataFrame:
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_mpc = pd.read_csv(config.PROCESSED_CPI_MPC_FILE)
    df_mpc["date"] = pd.to_datetime(df_mpc["date"])

    fiscal_years = financial_years_needed(df_mpc)
    logger.info("Need minutes for fiscal years: %s", fiscal_years)

    raw_links = discover_minutes_links(fiscal_years)
    matched = match_links_to_meetings(raw_links, df_mpc)
    matched.to_csv(INDEX_FILE, index=False)
    logger.info("Matched %d/%d known meetings to a minutes link. Index saved to %s",
                len(matched), len(df_mpc), INDEX_FILE)

    fetch_and_save_minutes_text(matched)
    logger.info("Done. Minutes text saved under %s", MINUTES_DIR)
    return matched


if __name__ == "__main__":
    run()