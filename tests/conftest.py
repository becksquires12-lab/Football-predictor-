"""Shared test fixtures and mock data."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


@pytest.fixture
def sample_team_stats():
    """Sample team stats as returned by FeatureEngineer.compute_team_stats()."""
    return {
        "avg_goals_scored": 1.8,
        "avg_goals_conceded": 1.0,
        "avg_home_goals_scored": 2.1,
        "avg_home_goals_conceded": 0.9,
        "avg_away_goals_scored": 1.5,
        "avg_away_goals_conceded": 1.1,
        "avg_corners_won": 5.5,
        "avg_corners_conceded": 4.8,
        "avg_xg": 1.7,
        "avg_shots": 14.2,
        "avg_possession": 55.0,
        "form": 2.0,
        "matches_played": 15,
    }


@pytest.fixture
def sample_away_stats():
    """Sample away team stats."""
    return {
        "avg_goals_scored": 1.4,
        "avg_goals_conceded": 1.2,
        "avg_home_goals_scored": 1.6,
        "avg_home_goals_conceded": 1.1,
        "avg_away_goals_scored": 1.2,
        "avg_away_goals_conceded": 1.3,
        "avg_corners_won": 4.8,
        "avg_corners_conceded": 5.2,
        "avg_xg": 1.3,
        "avg_shots": 11.5,
        "avg_possession": 48.0,
        "form": 1.6,
        "matches_played": 15,
    }


@pytest.fixture
def sample_league_avgs():
    """Sample league averages."""
    return {
        "avg_goals_per_match": 2.7,
        "avg_home_goals": 1.5,
        "avg_away_goals": 1.2,
        "avg_corners_per_match": 10.3,
    }


@pytest.fixture
def sample_features(sample_team_stats, sample_away_stats, sample_league_avgs):
    """Sample feature vector for predictions."""
    from src.features.engineer import FeatureEngineer
    engineer = FeatureEngineer()
    return engineer.build_feature_vector(
        sample_team_stats, sample_away_stats, sample_league_avgs
    )


@pytest.fixture
def sample_matches_df():
    """Sample DataFrame of historical matches."""
    np.random.seed(42)
    n = 50
    base_date = datetime(2024, 8, 1)
    rows = []

    for i in range(n):
        rows.append({
            "match_id": i + 1,
            "date": base_date + timedelta(days=i * 3),
            "league": "EPL",
            "season": "2024-25",
            "home_team_id": (i % 10) + 1,
            "away_team_id": ((i + 5) % 10) + 1,
            "home_team": f"Team {(i % 10) + 1}",
            "away_team": f"Team {((i + 5) % 10) + 1}",
            "home_goals": np.random.poisson(1.5),
            "away_goals": np.random.poisson(1.2),
            "home_corners": np.random.poisson(5.5),
            "away_corners": np.random.poisson(4.8),
            "total_goals": None,  # Will be computed
            "total_corners": None,
        })

    df = pd.DataFrame(rows)
    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["total_corners"] = df["home_corners"] + df["away_corners"]
    return df
