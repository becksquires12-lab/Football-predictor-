"""Poisson regression model for predicting match goals."""

import numpy as np
from scipy.stats import poisson
from joblib import dump, load
from pathlib import Path

from src.features.goals import estimate_poisson_lambdas
from src.utils.constants import MAX_GOALS, GOAL_THRESHOLDS
from src.utils.logger import get_logger

logger = get_logger("goals_model")


class GoalsModel:
    """Predict match goals using Poisson distribution.

    The model estimates separate Poisson lambdas for home and away
    team goals, then derives:
    - Exact scoreline probabilities
    - Over/Under probabilities for each threshold
    - Most likely scoreline
    """

    def __init__(self):
        self.trained = False
        self._league_params = {}  # Per-league calibration adjustments

    def predict(self, features):
        """Generate goal predictions from a feature vector.

        Args:
            features: Dict from FeatureEngineer.build_feature_vector()

        Returns:
            Dict with lambdas, scoreline probs, over/under probs.
        """
        home_lambda, away_lambda = estimate_poisson_lambdas(features)

        # Apply league-specific calibration if available
        league = features.get("league")
        if league and league in self._league_params:
            cal = self._league_params[league]
            home_lambda *= cal.get("home_scale", 1.0)
            away_lambda *= cal.get("away_scale", 1.0)

        result = {
            "home_lambda": round(home_lambda, 3),
            "away_lambda": round(away_lambda, 3),
            "expected_total_goals": round(home_lambda + away_lambda, 2),
        }

        # Scoreline probability matrix
        scoreline_probs = self._scoreline_matrix(home_lambda, away_lambda)
        result["scoreline_probs"] = scoreline_probs

        # Most likely score
        most_likely = self._most_likely_score(scoreline_probs)
        result["predicted_home_goals"] = most_likely[0]
        result["predicted_away_goals"] = most_likely[1]
        result["predicted_score_prob"] = most_likely[2]

        # Over/Under probabilities
        total_lambda = home_lambda + away_lambda
        result["over_under"] = self._over_under_probs(
            home_lambda, away_lambda, GOAL_THRESHOLDS
        )

        # Both teams to score probability
        p_home_zero = poisson.pmf(0, home_lambda)
        p_away_zero = poisson.pmf(0, away_lambda)
        result["btts_yes"] = round(
            (1 - p_home_zero) * (1 - p_away_zero), 4
        )
        result["btts_no"] = round(1 - result["btts_yes"], 4)

        return result

    def _scoreline_matrix(self, home_lambda, away_lambda, max_goals=MAX_GOALS):
        """Compute probability matrix for all scorelines up to max_goals.

        Returns:
            Dict mapping (home_goals, away_goals) -> probability
        """
        probs = {}
        home_pmf = [poisson.pmf(i, home_lambda) for i in range(max_goals + 1)]
        away_pmf = [poisson.pmf(i, away_lambda) for i in range(max_goals + 1)]

        for h in range(max_goals + 1):
            for a in range(max_goals + 1):
                p = home_pmf[h] * away_pmf[a]
                if p > 0.001:  # Only store meaningful probabilities
                    probs[(h, a)] = round(p, 4)

        return probs

    def _most_likely_score(self, scoreline_probs):
        """Find the most probable scoreline."""
        if not scoreline_probs:
            return (1, 1, 0.0)

        best = max(scoreline_probs.items(), key=lambda x: x[1])
        return (best[0][0], best[0][1], round(best[1], 4))

    def _over_under_probs(self, home_lambda, away_lambda, thresholds):
        """Calculate Over/Under probabilities for given thresholds.

        Uses convolution of two Poisson distributions to get the
        exact total goals distribution.
        """
        # Compute P(total = k) for k = 0..2*MAX_GOALS by convolving
        max_total = 2 * MAX_GOALS
        total_probs = np.zeros(max_total + 1)

        for h in range(MAX_GOALS + 1):
            p_h = poisson.pmf(h, home_lambda)
            for a in range(MAX_GOALS + 1):
                total = h + a
                if total <= max_total:
                    total_probs[total] += p_h * poisson.pmf(a, away_lambda)

        result = {}
        for threshold in thresholds:
            # "Over X" means strictly more than X
            over_idx = int(threshold) + 1  # e.g. Over 2.5 = 3+ goals
            p_over = float(np.sum(total_probs[over_idx:]))
            p_under = 1.0 - p_over

            key = str(threshold)
            result[key] = {
                "over": round(p_over, 4),
                "under": round(p_under, 4),
            }

        return result

    def calibrate(self, league, actual_matches):
        """Calibrate model for a specific league using historical results.

        Compares predicted vs actual goal distributions and computes
        scaling factors.
        """
        if not actual_matches:
            return

        predicted_home = []
        predicted_away = []
        actual_home = []
        actual_away = []

        for match in actual_matches:
            features = match.get("features")
            if features is None:
                continue

            h_lambda, a_lambda = estimate_poisson_lambdas(features)
            predicted_home.append(h_lambda)
            predicted_away.append(a_lambda)
            actual_home.append(match["home_goals"])
            actual_away.append(match["away_goals"])

        if not predicted_home:
            return

        # Scale factor = actual_mean / predicted_mean
        pred_h_mean = np.mean(predicted_home)
        pred_a_mean = np.mean(predicted_away)

        home_scale = np.mean(actual_home) / pred_h_mean if pred_h_mean > 0 else 1.0
        away_scale = np.mean(actual_away) / pred_a_mean if pred_a_mean > 0 else 1.0

        self._league_params[league] = {
            "home_scale": np.clip(home_scale, 0.7, 1.4),
            "away_scale": np.clip(away_scale, 0.7, 1.4),
        }
        self.trained = True

        logger.info(
            f"Calibrated {league}: home_scale={home_scale:.3f}, "
            f"away_scale={away_scale:.3f}"
        )

    def save(self, path="models/goals_model.joblib"):
        """Save model parameters to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        dump({"league_params": self._league_params, "trained": self.trained}, path)
        logger.info(f"Goals model saved to {path}")

    def load(self, path="models/goals_model.joblib"):
        """Load model parameters from disk."""
        data = load(path)
        self._league_params = data.get("league_params", {})
        self.trained = data.get("trained", False)
        logger.info(f"Goals model loaded from {path}")
