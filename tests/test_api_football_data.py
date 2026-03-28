"""Tests for the football-data.org API client."""

import pytest
import responses

from src.data.api_football_data import FootballDataClient


class TestFootballDataClient:
    @responses.activate
    def test_get_matches(self):
        responses.add(
            responses.GET,
            "https://api.football-data.org/v4/competitions/PL/matches",
            json={
                "matches": [
                    {
                        "id": 1,
                        "utcDate": "2024-08-17T14:00:00Z",
                        "status": "FINISHED",
                        "matchday": 1,
                        "season": {"startDate": "2024-08-01"},
                        "homeTeam": {"id": 57, "name": "Arsenal FC"},
                        "awayTeam": {"id": 65, "name": "Manchester City FC"},
                        "score": {
                            "fullTime": {"home": 2, "away": 1},
                        },
                    }
                ]
            },
            status=200,
        )

        client = FootballDataClient(api_key="test_key")
        matches = client.get_matches("EPL", 2024)

        assert len(matches) == 1
        assert matches[0]["home_team"] == "Arsenal FC"
        assert matches[0]["away_team"] == "Manchester City FC"
        assert matches[0]["home_goals"] == 2
        assert matches[0]["away_goals"] == 1
        assert matches[0]["status"] == "FINISHED"
        assert matches[0]["source"] == "football_data"

    @responses.activate
    def test_get_upcoming(self):
        responses.add(
            responses.GET,
            "https://api.football-data.org/v4/competitions/PL/matches",
            json={
                "matches": [
                    {
                        "id": 2,
                        "utcDate": "2025-03-29T15:00:00Z",
                        "status": "SCHEDULED",
                        "matchday": 30,
                        "season": {"startDate": "2024-08-01"},
                        "homeTeam": {"id": 57, "name": "Arsenal FC"},
                        "awayTeam": {"id": 61, "name": "Chelsea FC"},
                        "score": {"fullTime": {"home": None, "away": None}},
                    }
                ]
            },
            status=200,
        )

        client = FootballDataClient(api_key="test_key")
        matches = client.get_upcoming("EPL", limit=5)

        assert len(matches) == 1
        assert matches[0]["status"] == "SCHEDULED"
        assert matches[0]["home_goals"] is None

    def test_invalid_league_raises(self):
        client = FootballDataClient(api_key="test_key")
        with pytest.raises(ValueError, match="Unknown league"):
            client.get_matches("INVALID", 2024)

    def test_format_season(self):
        assert FootballDataClient._format_season("2024-08-01") == "2024-25"
        assert FootballDataClient._format_season("2023-08-01") == "2023-24"
        assert FootballDataClient._format_season(None) == "unknown"
