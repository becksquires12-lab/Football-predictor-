"""Understat web scraper for xG (expected goals) data."""

import json
import re

from bs4 import BeautifulSoup

from src.data.base_client import BaseClient
from src.data.rate_limiter import RateLimiter
from src.utils.logger import get_logger

logger = get_logger("scraper_understat")


class UnderstatScraper(BaseClient):
    """Scrape expected goals (xG) data from Understat.

    Understat embeds match data as JSON in script tags, making it
    reliable to parse. Covers top 5 European leagues.
    """

    BASE_URL = "https://understat.com"

    # Understat league slugs
    LEAGUE_MAP = {
        "EPL": "EPL",
        "LA_LIGA": "La_liga",
        "BUNDESLIGA": "Bundesliga",
        "SERIE_A": "Serie_A",
        "LIGUE_1": "Ligue_1",
    }

    def __init__(self):
        super().__init__(
            base_url=self.BASE_URL,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                ),
            },
        )
        self._limiter = RateLimiter(requests_per_minute=6)

    def get_matches(self, league, season):
        """Fetch match xG data for a league and season.

        Args:
            league: League key (e.g. "EPL")
            season: Season start year (e.g. 2024)

        Returns:
            List of match dicts with xG data.
        """
        slug = self.LEAGUE_MAP.get(league)
        if not slug:
            raise ValueError(
                f"Unknown league: {league}. Options: {list(self.LEAGUE_MAP)}"
            )

        self._limiter.wait()
        html = self._get_html(f"/league/{slug}/{season}")

        return self._parse_league_page(html, league)

    def _parse_league_page(self, html, league):
        """Extract match data from embedded JSON in the league page."""
        soup = BeautifulSoup(html, "lxml")
        matches = []

        for script in soup.find_all("script"):
            text = script.string or ""
            if "datesData" not in text:
                continue

            json_str = self._extract_json(text, "datesData")
            if not json_str:
                continue

            data = json.loads(json_str)
            for match_data in data:
                match = self._normalize_match(match_data, league)
                if match:
                    matches.append(match)
            break

        logger.info(f"Parsed {len(matches)} matches from Understat for {league}")
        return matches

    def _normalize_match(self, raw, league):
        """Convert Understat match data to standardized dict."""
        try:
            return {
                "source": "understat",
                "source_id": raw.get("id"),
                "league": league,
                "date": raw.get("datetime", "")[:10],
                "home_team": raw.get("h", {}).get("title"),
                "away_team": raw.get("a", {}).get("title"),
                "home_goals": self._safe_int(raw.get("goals", {}).get("h")),
                "away_goals": self._safe_int(raw.get("goals", {}).get("a")),
                "home_xg": self._safe_float(raw.get("xG", {}).get("h")),
                "away_xg": self._safe_float(raw.get("xG", {}).get("a")),
                "is_result": raw.get("isResult", False),
            }
        except (KeyError, TypeError) as e:
            logger.debug(f"Error parsing Understat match: {e}")
            return None

    @staticmethod
    def _extract_json(script_text, variable_name):
        """Extract JSON string from JavaScript variable assignment."""
        pattern = rf"{variable_name}\s*=\s*JSON\.parse\('(.+?)'\)"
        match = re.search(pattern, script_text)
        if match:
            # Understat encodes the JSON with hex escapes
            encoded = match.group(1)
            decoded = encoded.encode().decode("unicode_escape")
            return decoded
        return None

    @staticmethod
    def _safe_int(value):
        try:
            return int(value) if value is not None else None
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_float(value):
        try:
            return float(value) if value is not None else None
        except (ValueError, TypeError):
            return None
