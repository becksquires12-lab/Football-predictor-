"""Gradient Boosted Regressor + Negative Binomial model for corners."""

import numpy as np
from scipy.stats import nbinom
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from pathlib import Path

from src.features.corners import compute_corner_features, estimate_total_corners
from src.utils.constants import (
    MAX_CORNERS, CORNER_THRESHOLDS,
    DEFAULT_N_ESTIMATORS, DEFAULT_MAX_DEPTH, DEFAULT_LEARNING_RATE,
)
from src.utils.logger import get_logger

logger = get_logger("corners_model")


class CornersModel:
    """Predict match corners using Gradient Boosting + Negative Binomial.

    Corners are overdispersed (variance > mean), so a Poisson fit is poor.
    We use GBR to predict the mean, then fit a Negative Binomial distribution
    for probability calculations.
    """

    FEATURE_COLS = [
        "home_avg_corners_won", "home_avg_corners_conceded",
        "away_avg_corners_won", "away_avg_corners_conceded",
        "home_corner_attack", "away_corner_attack",
        "expected_total_corners",
    ]

    OPTIONAL_COLS = [
        "home_possession", "away_possession",
        "home_avg_shots", "away_avg_shots",
        "home_corners_per_shot", "away_corners_per_shot",
    ]

    def __init__(self, n_estimators=DEFAULT_N_ESTIMATORS,
                 max_depth=DEFAULT_MAX_DEPTH,
                 learning_rate=DEFAULT_LEARNING_RATE):
        self.model = GradientBoostingRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
        )
        self.trained = False
        self._feature_names = None
        self._dispersion = 1.5  # Default overdispersion parameter

    def train(self, X, y):
        """Train the corners model.

        Args:
            X: DataFrame or array of corner features.
            y: Array of actual total corners.
        """
        self._feature_names = list(X.columns) if hasattr(X, "columns") else None

        # Cross-validation score
        scores = cross_val_score(self.model, X, y, cv=5, scoring="neg_mean_squared_error")
        rmse = np.sqrt(-scores.mean())
        logger.info(f"Corners model CV RMSE: {rmse:.2f}")

        self.model.fit(X, y)
        self.trained = True

        # Estimate dispersion from residuals
        predictions = self.model.predict(X)
        residuals = y - predictions
        variance = np.var(residuals)
        mean_pred = np.mean(predictions)
        if mean_pred > 0:
            self._dispersion = max(variance / mean_pred, 1.01)

        logger.info(f"Corners model trained. Dispersion: {self._dispersion:.2f}")

    def predict(self, features):
        """Generate corner predictions.

        Args:
            features: Dict of corner features (from compute_corner_features)

        Returns:
            Dict with predicted corners and over/under probabilities.
        """
        if self.trained and features is not None:
            feature_vec = self._features_to_array(features)
            predicted_total = float(self.model.predict([feature_vec])[0])
        elif features is not None and "expected_total_corners" in features:
            predicted_total = features["expected_total_corners"]
        else:
            predicted_total = 10.0  # League average fallback

        predicted_total = np.clip(predicted_total, 3.0, 20.0)

        result = {
            "predicted_total_corners": round(predicted_total, 1),
        }

        # Negative Binomial distribution for probability spread
        ou_probs = self._over_under_probs(predicted_total, CORNER_THRESHOLDS)
        result["over_under"] = ou_probs

        # Distribution breakdown
        result["distribution"] = self._corner_distribution(predicted_total)

        return result

    def _features_to_array(self, features):
        """Convert feature dict to array matching training feature order."""
        if self._feature_names:
            return [features.get(f, 0.0) for f in self._feature_names]

        # Use default feature order
        vec = [features.get(f, 0.0) for f in self.FEATURE_COLS]
        for col in self.OPTIONAL_COLS:
            if col in features:
                vec.append(features[col])
        return vec

    def _over_under_probs(self, predicted_mean, thresholds):
        """Calculate Over/Under probabilities using Negative Binomial.

        The Negative Binomial is parameterized by:
        - n (number of successes) = mean^2 / (variance - mean)
        - p (probability) = mean / variance
        """
        n, p = self._nb_params(predicted_mean)

        result = {}
        for threshold in thresholds:
            # P(X > threshold) = 1 - P(X <= floor(threshold))
            p_under = float(nbinom.cdf(int(threshold), n, p))
            p_over = 1.0 - p_under

            key = str(threshold)
            result[key] = {
                "over": round(p_over, 4),
                "under": round(p_under, 4),
            }

        return result

    def _corner_distribution(self, predicted_mean, max_val=MAX_CORNERS):
        """Get probability distribution over corner counts."""
        n, p = self._nb_params(predicted_mean)
        dist = {}
        for k in range(max_val + 1):
            prob = float(nbinom.pmf(k, n, p))
            if prob > 0.005:
                dist[k] = round(prob, 4)
        return dist

    def _nb_params(self, mean):
        """Convert mean and dispersion to Negative Binomial n, p parameters."""
        variance = mean * self._dispersion
        if variance <= mean:
            variance = mean + 0.1  # Ensure overdispersion

        n = mean ** 2 / (variance - mean)
        p = mean / variance

        # Clamp to valid ranges
        n = max(n, 0.5)
        p = np.clip(p, 0.01, 0.99)
        return n, p

    def save(self, path="models/corners_model.joblib"):
        """Save model to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dump({
            "model": self.model,
            "trained": self.trained,
            "feature_names": self._feature_names,
            "dispersion": self._dispersion,
        }, path)
        logger.info(f"Corners model saved to {path}")

    def load(self, path="models/corners_model.joblib"):
        """Load model from disk."""
        data = load(path)
        self.model = data["model"]
        self.trained = data["trained"]
        self._feature_names = data.get("feature_names")
        self._dispersion = data.get("dispersion", 1.5)
        logger.info(f"Corners model loaded from {path}")
