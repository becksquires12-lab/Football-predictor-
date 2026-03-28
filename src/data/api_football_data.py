"""Client for football-data.org v4 API (free tier: 10 req/min)."""

from datetime import datetime

from src.data.base_client import BaseClient
from src.data.rate_limiter import RateLimiter
from src.utils.constants import LEAGUE_CODES


class FootballDataClient(BaseClient):
    """Fetch fixtures, results, and standings from football-data.org."""

    BASE_URL = "https://api.football-data.org/v4"

    def __init__(self, api_key):
        super().__init__(
            base_url=self.BASE_URL,
            headers={"X-Auth-Token": api_key},
        )
        self._limiter = RateLimiter(requests_per_minute=10)

    def _rate_limited_get(self, endpoint, params=None):
        self._limiter.wait()
        return self._get(endpoint, params)

    def get_matches(self, league, season):
        """Fetch all matches for a league and season.

        Args:
            league: League key (e.g. "EPL", "LA_LIGA")
            season: Season year (e.g. 2024 for 2024-25)

        Returns:
            List of normalized match dicts.
        """
        league_info = LEAGUE_CODES.get(league)
        if not league_info:
            raise ValueError(f"Unknown league: {league}. Options: {list(LEAGUE_CODES)}")

        code = league_info["football_data_id"]
        data = self._rate_limited_get(
            f"/competitions/{code}/matches",
            params={"season": season},
        )

        matches = []
        for m in data.get("matches", []):
            matches.append(self._normalize_match(m, league))
        return matches

    def get_upcoming(self, league, limit=10):
        """Fetch upcoming scheduled matches for a league."""
        league_info = LEAGUE_CODES.get(league)
        if not league_info:
            raise ValueError(f"Unknown league: {league}")

        code = league_info["football_data_id"]
        data = self._rate_limited_get(
            f"/competitions/{code}/matches",
            params={"status": "SCHEDULED", "limit": limit},
        )

        return [self._normalize_match(m, league) for m in data.get("matches", [])]

    def get_standings(self, league, season):
        """Fetch current standings for a league."""
        league_info = LEAGUE_CODES.get(league)
        if not league_info:
            raise ValueError(f"Unknown league: {league}")

        code = league_info["football_data_id"]
        data = self._rate_limited_get(
            f"/competitions/{code}/standings",
            params={"season": season},
        )
        return data.get("standings", [])

    def _normalize_match(self, raw, league):
        """Convert API response to a standardized match dict."""
        score = raw.get("score", {})
        full_time = score.get("fullTime", {})

        return {
            "source": "football_data",
            "source_id": raw.get("id"),
            "league": league,
            "season": self._format_season(raw.get("season", {}).get("startDate")),
            "matchday": raw.get("matchday"),
            "date": datetime.fromisoformat(raw["utcDate"].replace("Z", "+00:00")),
            "status": raw.get("status", "SCHEDULED"),
            "home_team": raw.get("homeTeam", {}).get("name"),
            "away_team": raw.get("awayTeam", {}).get("name"),
            "home_team_id": raw.get("homeTeam", {}).get("id"),
            "away_team_id": raw.get("awayTeam", {}).get("id"),
            "home_goals": full_time.get("home"),
            "away_goals": full_time.get("away"),
            "home_corners": None,  # Not available from this API
            "away_corners": None,
        }

    @staticmethod
    def _format_season(start_date):
        """Convert season start date to '2024-25' format."""
        if not start_date:
            return "unknown"
        year = int(start_date[:4])
        return f"{year}-{str(year + 1)[-2:]}"
