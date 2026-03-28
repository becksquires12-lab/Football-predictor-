"""Tests for the corners prediction model."""

import pytest
import numpy as np
import pandas as pd

from src.predictions.corners_model import CornersModel


class TestCornersModel:
    def test_predict_untrained_uses_fallback(self):
        model = CornersModel()
        features = {
            "home_avg_corners_won": 5.5,
            "home_avg_corners_conceded": 4.8,
            "away_avg_corners_won": 4.5,
            "away_avg_corners_conceded": 5.2,
            "home_corner_attack": 1.1,
            "away_corner_attack": 0.9,
            "expected_total_corners": 10.0,
        }
        result = model.predict(features)

        assert "predicted_total_corners" in result
        assert "over_under" in result
        assert result["predicted_total_corners"] > 0

    def test_predict_with_none_features(self):
        model = CornersModel()
        result = model.predict(None)

        assert result["predicted_total_corners"] == 10.0  # Default fallback

    def test_over_under_probabilities_sum_to_one(self):
        model = CornersModel()
        features = {
            "expected_total_corners": 10.5,
        }
        result = model.predict(features)

        for threshold, probs in result["over_under"].items():
            total = probs["over"] + probs["under"]
            assert abs(total - 1.0) < 0.01

    def test_higher_threshold_means_lower_over_prob(self):
        model = CornersModel()
        features = {"expected_total_corners": 10.0}
        result = model.predict(features)

        ou = result["over_under"]
        thresholds = sorted(ou.keys(), key=float)

        for i in range(len(thresholds) - 1):
            assert ou[thresholds[i]]["over"] >= ou[thresholds[i + 1]]["over"]

    def test_distribution_is_valid(self):
        model = CornersModel()
        features = {"expected_total_corners": 10.0}
        result = model.predict(features)

        dist = result["distribution"]
        assert len(dist) > 0
        total_prob = sum(dist.values())
        assert total_prob > 0.9  # Most probability mass should be captured

        for count, prob in dist.items():
            assert count >= 0
            assert 0 < prob <= 1.0

    def test_train_and_predict(self):
        """Test training with synthetic data and then predicting."""
        model = CornersModel(n_estimators=50, max_depth=3)

        # Generate synthetic training data
        np.random.seed(42)
        n = 100
        X = pd.DataFrame({
            "home_avg_corners_won": np.random.normal(5.5, 1.0, n),
            "home_avg_corners_conceded": np.random.normal(4.8, 1.0, n),
            "away_avg_corners_won": np.random.normal(5.0, 1.0, n),
            "away_avg_corners_conceded": np.random.normal(5.0, 1.0, n),
            "home_corner_attack": np.random.normal(1.0, 0.2, n),
            "away_corner_attack": np.random.normal(1.0, 0.2, n),
            "expected_total_corners": np.random.normal(10.3, 1.5, n),
        })
        y = np.random.poisson(10.3, n).astype(float)

        model.train(X, y)
        assert model.trained

        features = {
            "home_avg_corners_won": 6.0,
            "home_avg_corners_conceded": 4.5,
            "away_avg_corners_won": 5.0,
            "away_avg_corners_conceded": 5.0,
            "home_corner_attack": 1.2,
            "away_corner_attack": 1.0,
            "expected_total_corners": 10.5,
        }
        result = model.predict(features)

        assert 5.0 < result["predicted_total_corners"] < 18.0
