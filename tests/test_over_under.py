"""Tests for the Over/Under probability calculator."""

import pytest
from src.predictions.over_under import OverUnderPredictor


class TestOverUnderPredictor:
    def test_predict_returns_goals_and_corners(self, sample_features,
                                                sample_team_stats,
                                                sample_away_stats,
                                                sample_league_avgs):
        predictor = OverUnderPredictor()
        result = predictor.predict(
            sample_features, sample_team_stats, sample_away_stats, sample_league_avgs
        )

        assert "goals" in result
        assert "corners" in result

    def test_goals_prediction_structure(self, sample_features):
        predictor = OverUnderPredictor()
        result = predictor.quick_predict(sample_features)

        goals = result["goals"]
        assert "predicted_home" in goals
        assert "predicted_away" in goals
        assert "predicted_total" in goals
        assert "over_under" in goals
        assert "btts" in goals

    def test_quick_predict_without_corners(self, sample_features):
        predictor = OverUnderPredictor()
        result = predictor.quick_predict(sample_features)

        assert result["corners"]["predicted_total"] is None
        assert "note" in result["corners"]

    def test_goals_over_25_reasonable(self, sample_features):
        """Over 2.5 goals should be in a reasonable range for typical teams."""
        predictor = OverUnderPredictor()
        result = predictor.quick_predict(sample_features)

        ou_25 = result["goals"]["over_under"]["2.5"]
        # For average teams, Over 2.5 is usually between 30-70%
        assert 0.1 < ou_25["over"] < 0.9
