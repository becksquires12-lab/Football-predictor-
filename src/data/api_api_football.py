"""Client for API-Football via RapidAPI (free tier: 100 req/day)."""

from datetime import datetime

from src.data.base_client import BaseClient
from src.data.rate_limiter import RateLimiter
from src.utils.constants import LEAGUE_CODES, INTERNATIONAL_COMPS


class APIFootballClient(BaseClient):
    """Fetch fixtures with detailed stats (including corners) from API-Football."""

    BASE_URL = "https://api-football-v1.p.rapidapi.com/v3"

    def __init__(self, api_key):
        super().__init__(
            base_url=self.BASE_URL,
            headers={
                "X-RapidAPI-Key": api_key,
                "X-RapidAPI-Host": "api-football-v1.p.rapidapi.com",
            },
        )
        self._limiter = RateLimiter(requests_per_minute=10)

    def _rate_limited_get(self, endpoint, params=None):
        self._limiter.wait()
        return self._get(endpoint, params)

    def get_matches(self, league, season):
        """Fetch all fixtures for a league and season.

        Args:
            league: League key (e.g. "EPL") or international comp key
            season: Season year (e.g. 2024)

        Returns:
            List of normalized match dicts with corner data.
        """
        league_id = self._resolve_league_id(league)

        data = self._rate_limited_get(
            "/fixtures",
            params={"league": league_id, "season": season},
        )

        matches = []
        for fixture in data.get("response", []):
            matches.append(self._normalize_fixture(fixture, league))
        return matches

    def get_fixture_stats(self, fixture_id):
        """Fetch detailed stats for a specific fixture (includes corners)."""
        data = self._rate_limited_get(
            "/fixtures/statistics",
            params={"fixture": fixture_id},
        )
        return self._parse_stats(data.get("response", []))

    def get_upcoming(self, league, next_count=10):
        """Fetch next N upcoming fixtures for a league."""
        league_id = self._resolve_league_id(league)

        data = self._rate_limited_get(
            "/fixtures",
            params={"league": league_id, "next": next_count},
        )

        return [
            self._normalize_fixture(f, league) for f in data.get("response", [])
        ]

    def get_international_matches(self, competition, season):
        """Fetch international competition matches."""
        comp_info = INTERNATIONAL_COMPS.get(competition)
        if not comp_info:
            raise ValueError(
                f"Unknown competition: {competition}. "
                f"Options: {list(INTERNATIONAL_COMPS)}"
            )

        data = self._rate_limited_get(
            "/fixtures",
            params={"league": comp_info["api_football_id"], "season": season},
        )

        matches = []
        for fixture in data.get("response", []):
            normalized = self._normalize_fixture(fixture, competition)
            normalized["is_international"] = True
            matches.append(normalized)
        return matches

    def _resolve_league_id(self, league):
        """Get the API-Football league ID from a league key."""
        if league in LEAGUE_CODES:
            return LEAGUE_CODES[league]["api_football_id"]
        if league in INTERNATIONAL_COMPS:
            return INTERNATIONAL_COMPS[league]["api_football_id"]
        raise ValueError(f"Unknown league/competition: {league}")

    def _normalize_fixture(self, raw, league):
        """Convert API-Football response to standardized match dict."""
        fixture = raw.get("fixture", {})
        teams = raw.get("teams", {})
        goals = raw.get("goals", {})

        return {
            "source": "api_football",
            "source_id": fixture.get("id"),
            "league": league,
            "season": str(raw.get("league", {}).get("season", "")),
            "matchday": raw.get("league", {}).get("round"),
            "date": datetime.fromisoformat(
                fixture.get("date", "").replace("Z", "+00:00")
            )
            if fixture.get("date")
            else None,
            "status": self._map_status(fixture.get("status", {}).get("short")),
            "home_team": teams.get("home", {}).get("name"),
            "away_team": teams.get("away", {}).get("name"),
            "home_team_api_id": teams.get("home", {}).get("id"),
            "away_team_api_id": teams.get("away", {}).get("id"),
            "home_goals": goals.get("home"),
            "away_goals": goals.get("away"),
            "home_corners": None,  # Populated via get_fixture_stats
            "away_corners": None,
            "is_international": False,
        }

    def _parse_stats(self, response):
        """Extract stats from fixture statistics response."""
        stats = {"home": {}, "away": {}}

        for i, team_data in enumerate(response[:2]):
            side = "home" if i == 0 else "away"
            for stat in team_data.get("statistics", []):
                stat_type = stat.get("type", "").lower().replace(" ", "_")
                stats[side][stat_type] = stat.get("value")

        return {
            "home_corners": stats["home"].get("corner_kicks"),
            "away_corners": stats["away"].get("corner_kicks"),
            "home_shots": stats["home"].get("total_shots"),
            "away_shots": stats["away"].get("total_shots"),
            "home_shots_on_target": stats["home"].get("shots_on_goal"),
            "away_shots_on_target": stats["away"].get("shots_on_goal"),
            "home_possession": self._parse_pct(
                stats["home"].get("ball_possession")
            ),
            "away_possession": self._parse_pct(
                stats["away"].get("ball_possession")
            ),
            "home_fouls": stats["home"].get("fouls"),
            "away_fouls": stats["away"].get("fouls"),
            "home_yellow_cards": stats["home"].get("yellow_cards"),
            "away_yellow_cards": stats["away"].get("yellow_cards"),
            "home_red_cards": stats["home"].get("red_cards"),
            "away_red_cards": stats["away"].get("red_cards"),
        }

    @staticmethod
    def _parse_pct(value):
        """Parse percentage string like '55%' to float 55.0."""
        if value is None:
            return None
        if isinstance(value, str):
            return float(value.replace("%", ""))
        return float(value)

    @staticmethod
    def _map_status(short_status):
        """Map API-Football status codes to our standard statuses."""
        mapping = {
            "FT": "FINISHED",
            "AET": "FINISHED",
            "PEN": "FINISHED",
            "NS": "SCHEDULED",
            "PST": "POSTPONED",
            "CANC": "CANCELLED",
            "TBD": "SCHEDULED",
        }
        return mapping.get(short_status, "IN_PLAY")
