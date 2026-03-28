"""Over/Under probability calculator — the main user-facing output."""

from src.predictions.goals_model import GoalsModel
from src.predictions.corners_model import CornersModel
from src.features.corners import compute_corner_features
from src.utils.constants import GOAL_THRESHOLDS, CORNER_THRESHOLDS
from src.utils.logger import get_logger

logger = get_logger("over_under")


class OverUnderPredictor:
    """Combine goals and corners models to produce Over/Under predictions."""

    def __init__(self, goals_model=None, corners_model=None):
        self.goals_model = goals_model or GoalsModel()
        self.corners_model = corners_model or CornersModel()

    def predict(self, features, home_stats=None, away_stats=None, league_avgs=None):
        """Generate full Over/Under predictions for a match.

        Args:
            features: Feature vector from FeatureEngineer.build_feature_vector()
            home_stats: Raw team stats for corner features
            away_stats: Raw team stats for corner features
            league_avgs: League averages for normalization

        Returns:
            Dict with goals and corners predictions.
        """
        result = {"goals": {}, "corners": {}}

        # Goals predictions
        goals_pred = self.goals_model.predict(features)
        result["goals"] = {
            "predicted_home": goals_pred["predicted_home_goals"],
            "predicted_away": goals_pred["predicted_away_goals"],
            "predicted_total": goals_pred["expected_total_goals"],
            "over_under": goals_pred["over_under"],
            "btts": {
                "yes": goals_pred["btts_yes"],
                "no": goals_pred["btts_no"],
            },
            "home_lambda": goals_pred["home_lambda"],
            "away_lambda": goals_pred["away_lambda"],
        }

        # Corner predictions
        corner_features = None
        if home_stats and away_stats and league_avgs:
            corner_features = compute_corner_features(
                home_stats, away_stats, league_avgs
            )

        corners_pred = self.corners_model.predict(corner_features)
        result["corners"] = {
            "predicted_total": corners_pred["predicted_total_corners"],
            "over_under": corners_pred["over_under"],
        }

        return result

    def quick_predict(self, features):
        """Simplified prediction using only the feature vector (no corner model).

        Useful when corner data isn't available.
        """
        goals_pred = self.goals_model.predict(features)

        return {
            "goals": {
                "predicted_home": goals_pred["predicted_home_goals"],
                "predicted_away": goals_pred["predicted_away_goals"],
                "predicted_total": goals_pred["expected_total_goals"],
                "over_under": goals_pred["over_under"],
                "btts": {
                    "yes": goals_pred["btts_yes"],
                    "no": goals_pred["btts_no"],
                },
            },
            "corners": {
                "predicted_total": None,
                "over_under": {},
                "note": "Corner data insufficient for prediction",
            },
        }

    def load_models(self, goals_path="models/goals_model.joblib",
                    corners_path="models/corners_model.joblib"):
        """Load both models from disk."""
        try:
            self.goals_model.load(goals_path)
        except FileNotFoundError:
            logger.warning("Goals model not found, using uncalibrated defaults")

        try:
            self.corners_model.load(corners_path)
        except FileNotFoundError:
            logger.warning("Corners model not found, using statistical estimates")
