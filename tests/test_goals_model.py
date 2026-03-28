"""Tests for the Poisson goals prediction model."""

import pytest
from src.predictions.goals_model import GoalsModel


class TestGoalsModel:
    def test_predict_returns_expected_keys(self, sample_features):
        model = GoalsModel()
        result = model.predict(sample_features)

        assert "home_lambda" in result
        assert "away_lambda" in result
        assert "expected_total_goals" in result
        assert "predicted_home_goals" in result
        assert "predicted_away_goals" in result
        assert "over_under" in result
        assert "btts_yes" in result

    def test_lambdas_are_positive(self, sample_features):
        model = GoalsModel()
        result = model.predict(sample_features)

        assert result["home_lambda"] > 0
        assert result["away_lambda"] > 0

    def test_over_under_probabilities_sum_to_one(self, sample_features):
        model = GoalsModel()
        result = model.predict(sample_features)

        for threshold, probs in result["over_under"].items():
            total = probs["over"] + probs["under"]
            assert abs(total - 1.0) < 0.01, (
                f"O/U {threshold}: over + under = {total}"
            )

    def test_btts_probabilities_sum_to_one(self, sample_features):
        model = GoalsModel()
        result = model.predict(sample_features)

        total = result["btts_yes"] + result["btts_no"]
        assert abs(total - 1.0) < 0.01

    def test_higher_threshold_means_lower_over_prob(self, sample_features):
        model = GoalsModel()
        result = model.predict(sample_features)

        ou = result["over_under"]
        thresholds = sorted(ou.keys(), key=float)

        for i in range(len(thresholds) - 1):
            assert ou[thresholds[i]]["over"] >= ou[thresholds[i + 1]]["over"], (
                f"Over {thresholds[i]} should be >= Over {thresholds[i+1]}"
            )

    def test_predicted_score_is_non_negative(self, sample_features):
        model = GoalsModel()
        result = model.predict(sample_features)

        assert result["predicted_home_goals"] >= 0
        assert result["predicted_away_goals"] >= 0

    def test_scoreline_probs_are_valid(self, sample_features):
        model = GoalsModel()
        result = model.predict(sample_features)

        total_prob = sum(result["scoreline_probs"].values())
        # Should be close to 1.0 (some small prob scorelines are filtered out)
        assert total_prob > 0.9
        assert total_prob <= 1.01

        for score, prob in result["scoreline_probs"].items():
            assert prob > 0
            assert prob <= 1.0
            assert score[0] >= 0
            assert score[1] >= 0

    def test_strong_home_team_higher_lambda(self):
        """A strong home team should have higher home_lambda."""
        model = GoalsModel()

        strong_features = {
            "home_attack_strength": 1.5,
            "home_defense_strength": 0.8,
            "away_attack_strength": 0.8,
            "away_defense_strength": 1.2,
            "home_avg_goals_scored": 2.5,
            "home_avg_goals_conceded": 0.7,
            "away_avg_goals_scored": 1.0,
            "away_avg_goals_conceded": 1.5,
            "home_form": 2.5,
            "away_form": 1.0,
            "league_avg_goals": 2.7,
            "home_expected_goals": 2.7,
            "away_expected_goals": 0.96,
        }

        result = model.predict(strong_features)
        assert result["home_lambda"] > result["away_lambda"]
