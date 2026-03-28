"""FBRef web scraper for corner counts, xG, and advanced match stats."""

import time
from datetime import datetime

from bs4 import BeautifulSoup

from src.data.base_client import BaseClient
from src.data.rate_limiter import RateLimiter
from src.utils.logger import get_logger

logger = get_logger("scraper_fbref")


class FBRefScraper(BaseClient):
    """Scrape match data from FBRef (fbref.com).

    FBRef provides detailed match stats including corners, shots, xG,
    and possession that aren't always available via free APIs.
    """

    BASE_URL = "https://fbref.com"

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
        """Fetch match results with stats for a league season.

        Args:
            league: League key (e.g. "EPL")
            season: Season string (e.g. "2024-2025")

        Returns:
            List of match dicts with corner and xG data.
        """
        from src.utils.constants import LEAGUE_CODES

        league_info = LEAGUE_CODES.get(league)
        if not league_info:
            raise ValueError(f"Unknown league: {league}")

        slug = league_info["fbref_slug"]
        self._limiter.wait()
        html = self._get_html(f"/en/comps/{self._league_fbref_id(league)}/{season}/schedule/{season}-{slug}-Scores-and-Fixtures")

        return self._parse_scores_page(html, league)

    def get_match_details(self, match_url):
        """Scrape detailed stats from an individual match report page.

        Returns:
            Dict with corners, shots, xG, possession, etc.
        """
        self._limiter.wait()
        html = self._get_html(match_url)
        return self._parse_match_report(html)

    def _parse_scores_page(self, html, league):
        """Parse the scores and fixtures page to extract match results."""
        soup = BeautifulSoup(html, "lxml")
        matches = []

        table = soup.find("table", {"id": lambda x: x and "sched" in str(x)})
        if not table:
            logger.warning("Could not find schedule table on page")
            return matches

        tbody = table.find("tbody")
        if not tbody:
            return matches

        for row in tbody.find_all("tr", class_=lambda c: c != "thead"):
            cells = row.find_all(["td", "th"])
            if len(cells) < 8:
                continue

            match = self._parse_score_row(cells, league)
            if match:
                matches.append(match)

        return matches

    def _parse_score_row(self, cells, league):
        """Parse a single row from the scores table."""
        try:
            date_cell = cells[1].get_text(strip=True) if len(cells) > 1 else None
            if not date_cell:
                return None

            home_team = cells[3].get_text(strip=True) if len(cells) > 3 else None
            score_text = cells[5].get_text(strip=True) if len(cells) > 5 else None
            away_team = cells[7].get_text(strip=True) if len(cells) > 7 else None

            home_goals, away_goals = None, None
            if score_text and "–" in score_text:
                parts = score_text.split("–")
                home_goals = int(parts[0].strip())
                away_goals = int(parts[1].strip())

            # Match report link for detailed stats
            report_link = None
            for cell in cells:
                link = cell.find("a", string=lambda s: s and "Match Report" in s)
                if link:
                    report_link = link.get("href")
                    break

            return {
                "source": "fbref",
                "league": league,
                "date": datetime.strptime(date_cell, "%Y-%m-%d"),
                "home_team": home_team,
                "away_team": away_team,
                "home_goals": home_goals,
                "away_goals": away_goals,
                "report_url": report_link,
                "home_corners": None,  # Available in match report
                "away_corners": None,
            }
        except (ValueError, IndexError) as e:
            logger.debug(f"Error parsing row: {e}")
            return None

    def _parse_match_report(self, html):
        """Parse an individual match report for detailed stats."""
        soup = BeautifulSoup(html, "lxml")
        stats = {}

        # Look for the extra stats table (contains corners, fouls, etc.)
        extra_stats = soup.find("div", {"id": "team_stats_extra"})
        if extra_stats:
            stats.update(self._extract_extra_stats(extra_stats))

        # Look for xG in the scorebox
        scorebox = soup.find("div", class_="scorebox")
        if scorebox:
            xg_values = scorebox.find_all("div", class_="score_xg")
            if len(xg_values) >= 2:
                try:
                    stats["home_xg"] = float(xg_values[0].get_text(strip=True))
                    stats["away_xg"] = float(xg_values[1].get_text(strip=True))
                except ValueError:
                    pass

        return stats

    def _extract_extra_stats(self, container):
        """Extract stats from the team_stats_extra div."""
        stats = {}
        divs = container.find_all("div")

        for i, div in enumerate(divs):
            text = div.get_text(strip=True).lower()
            if "corner" in text:
                values = self._get_stat_pair(divs, i)
                if values:
                    stats["home_corners"] = values[0]
                    stats["away_corners"] = values[1]
            elif "foul" in text:
                values = self._get_stat_pair(divs, i)
                if values:
                    stats["home_fouls"] = values[0]
                    stats["away_fouls"] = values[1]

        return stats

    @staticmethod
    def _get_stat_pair(divs, label_index):
        """Get home/away stat values from adjacent divs around a label."""
        try:
            home_val = int(divs[label_index - 1].get_text(strip=True))
            away_val = int(divs[label_index + 1].get_text(strip=True))
            return (home_val, away_val)
        except (ValueError, IndexError):
            return None

    @staticmethod
    def _league_fbref_id(league):
        """Map league key to FBRef competition ID."""
        mapping = {
            "EPL": 9,
            "LA_LIGA": 12,
            "BUNDESLIGA": 20,
            "SERIE_A": 11,
            "LIGUE_1": 13,
        }
        return mapping.get(league, 9)
