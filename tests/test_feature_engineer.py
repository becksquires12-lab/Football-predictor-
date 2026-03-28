"""Tests for the feature engineering pipeline."""

import pytest
import numpy as np
from datetime import datetime, timedelta

from src.features.engineer import FeatureEngineer
from src.features.goals import estimate_poisson_lambdas, compute_goal_features
from src.features.corners import compute_corner_features, estimate_total_corners
from src.features.international import (
    adjust_features_for_international,
    estimate_international_lambdas,
    get_ranking_tier,
)


class TestFeatureEngineer:
    def test_build_feature_vector(self, sample_team_stats, sample_away_stats,
                                   sample_league_avgs):
        engineer = FeatureEngineer()
        features = engineer.build_feature_vector(
            sample_team_stats, sample_away_stats, sample_league_avgs
        )

        assert features is not None
        assert "home_attack_strength" in features
        assert "away_defense_strength" in features
        assert "home_expected_goals" in features
        assert "away_expected_goals" in features
        assert features["home_expected_goals"] > 0
        assert features["away_expected_goals"] > 0

    def test_build_feature_vector_with_none(self, sample_team_stats, sample_league_avgs):
        engineer = FeatureEngineer()
        result = engineer.build_feature_vector(None, sample_team_stats, sample_league_avgs)
        assert result is None

    def test_compute_team_stats_from_df(self, sample_matches_df):
        engineer = FeatureEngineer()
        stats = engineer.compute_team_stats(
            sample_matches_df, 1, datetime(2025, 1, 1)
        )

        assert stats is not None
        assert "avg_goals_scored" in stats
        assert "avg_goals_conceded" in stats
        assert "avg_corners_won" in stats
        assert "form" in stats
        assert stats["avg_goals_scored"] >= 0
        assert stats["matches_played"] > 0

    def test_compute_team_stats_insufficient_data(self, sample_matches_df):
        engineer = FeatureEngineer()
        # Team ID 99 doesn't exist in sample data
        stats = engineer.compute_team_stats(
            sample_matches_df, 99, datetime(2025, 1, 1)
        )
        assert stats is None

    def test_league_averages(self, sample_matches_df):
        engineer = FeatureEngineer()
        avgs = engineer.compute_league_averages(sample_matches_df)

        assert "avg_goals_per_match" in avgs
        assert "avg_home_goals" in avgs
        assert "avg_away_goals" in avgs
        assert avgs["avg_goals_per_match"] > 0

    def test_head_to_head(self, sample_matches_df):
        engineer = FeatureEngineer()
        h2h = engineer.head_to_head(sample_matches_df, 1, 6)

        assert "avg_total_goals" in h2h
        assert "matches" in h2h


class TestGoalFeatures:
    def test_estimate_poisson_lambdas(self, sample_features):
        home_lambda, away_lambda = estimate_poisson_lambdas(sample_features)

        assert 0.3 <= home_lambda <= 4.5
        assert 0.2 <= away_lambda <= 4.0

    def test_compute_goal_features(self, sample_team_stats, sample_away_stats,
                                    sample_league_avgs):
        features = compute_goal_features(
            sample_team_stats, sample_away_stats, sample_league_avgs
        )

        assert "home_attack_strength" in features
        assert "away_defense_weakness" in features
        assert "home_scoring_rate" in features
        # With xG data available
        assert "home_xg_rate" in features
        assert "home_xg_overperformance" in features


class TestCornerFeatures:
    def test_compute_corner_features(self, sample_team_stats, sample_away_stats,
                                      sample_league_avgs):
        features = compute_corner_features(
            sample_team_stats, sample_away_stats, sample_league_avgs
        )

        assert features is not None
        assert "home_avg_corners_won" in features
        assert "expected_total_corners" in features
        assert "home_corner_attack" in features
        assert features["expected_total_corners"] > 0

    def test_corner_features_without_data(self, sample_league_avgs):
        no_corners = {
            "avg_corners_won": None,
            "avg_corners_conceded": None,
        }
        result = compute_corner_features(no_corners, no_corners, sample_league_avgs)
        assert result is None

    def test_estimate_total_corners(self, sample_team_stats, sample_away_stats,
                                     sample_league_avgs):
        total = estimate_total_corners(
            sample_team_stats, sample_away_stats, sample_league_avgs
        )
        assert 5.0 < total < 18.0


class TestInternationalFeatures:
    def test_ranking_tiers(self):
        elite = get_ranking_tier(5)
        assert elite["attack_mult"] > 1.0

        weak = get_ranking_tier(150)
        assert weak["attack_mult"] < 1.0

    def test_estimate_international_lambdas(self):
        # Elite vs weak team
        h, a = estimate_international_lambdas(5, 150)
        assert h > a  # Elite home team should have higher lambda

    def test_adjust_features_for_international(self, sample_features):
        adjusted = adjust_features_for_international(
            sample_features, home_ranking=10, away_ranking=30
        )

        assert "is_international" in adjusted
        assert adjusted["is_international"] is True
        assert adjusted["home_expected_goals"] > 0
        assert adjusted["away_expected_goals"] > 0
