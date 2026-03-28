"""Backtesting and accuracy evaluation for prediction models."""

import numpy as np
import pandas as pd

from src.features.engineer import FeatureEngineer
from src.features.corners import compute_corner_features
from src.predictions.goals_model import GoalsModel
from src.predictions.corners_model import CornersModel
from src.utils.constants import GOAL_THRESHOLDS, CORNER_THRESHOLDS
from src.utils.logger import get_logger

logger = get_logger("evaluator")


class ModelEvaluator:
    """Evaluate prediction models against historical results."""

    def __init__(self, goals_model=None, corners_model=None):
        self.engineer = FeatureEngineer()
        self.goals_model = goals_model or GoalsModel()
        self.corners_model = corners_model or CornersModel()

    def evaluate_goals(self, league, season=None):
        """Evaluate goals model accuracy for a league.

        Returns:
            Dict with Brier scores, calibration stats, and accuracy metrics.
        """
        df = self.engineer.load_matches_df(league=league)
        if season:
            df = df[df["season"] == season]

        if len(df) < 20:
            return {"error": f"Insufficient data: {len(df)} matches"}

        league_avgs = self.engineer.compute_league_averages(df)
        results = []

        for _, row in df.iterrows():
            home_stats = self.engineer.compute_team_stats(
                df, row["home_team_id"], row["date"]
            )
            away_stats = self.engineer.compute_team_stats(
                df, row["away_team_id"], row["date"]
            )

            if home_stats is None or away_stats is None:
                continue

            features = self.engineer.build_feature_vector(
                home_stats, away_stats, league_avgs
            )
            if features is None:
                continue

            prediction = self.goals_model.predict(features)
            actual_total = row["total_goals"]

            result = {
                "predicted_total": prediction["expected_total_goals"],
                "actual_total": actual_total,
                "predicted_home": prediction["predicted_home_goals"],
                "predicted_away": prediction["predicted_away_goals"],
                "actual_home": row["home_goals"],
                "actual_away": row["away_goals"],
            }

            # Over/Under accuracy per threshold
            for t in GOAL_THRESHOLDS:
                key = str(t)
                pred_over = prediction["over_under"][key]["over"]
                actual_over = 1 if actual_total > t else 0
                result[f"ou_{key}_pred"] = pred_over
                result[f"ou_{key}_actual"] = actual_over

            results.append(result)

        if not results:
            return {"error": "No valid predictions generated"}

        return self._compute_metrics(results, GOAL_THRESHOLDS, "goals")

    def evaluate_corners(self, league, season=None):
        """Evaluate corners model accuracy for a league."""
        df = self.engineer.load_matches_df(league=league)
        if season:
            df = df[df["season"] == season]

        corner_df = df.dropna(subset=["home_corners", "away_corners"])
        if len(corner_df) < 20:
            return {"error": f"Insufficient corner data: {len(corner_df)} matches"}

        league_avgs = self.engineer.compute_league_averages(corner_df)
        results = []

        for _, row in corner_df.iterrows():
            home_stats = self.engineer.compute_team_stats(
                corner_df, row["home_team_id"], row["date"]
            )
            away_stats = self.engineer.compute_team_stats(
                corner_df, row["away_team_id"], row["date"]
            )

            if home_stats is None or away_stats is None:
                continue

            corner_features = compute_corner_features(
                home_stats, away_stats, league_avgs
            )
            prediction = self.corners_model.predict(corner_features)
            actual_total = row["total_corners"]

            result = {
                "predicted_total": prediction["predicted_total_corners"],
                "actual_total": actual_total,
            }

            for t in CORNER_THRESHOLDS:
                key = str(t)
                if key in prediction["over_under"]:
                    pred_over = prediction["over_under"][key]["over"]
                    actual_over = 1 if actual_total > t else 0
                    result[f"ou_{key}_pred"] = pred_over
                    result[f"ou_{key}_actual"] = actual_over

            results.append(result)

        if not results:
            return {"error": "No valid corner predictions generated"}

        return self._compute_metrics(results, CORNER_THRESHOLDS, "corners")

    def _compute_metrics(self, results, thresholds, model_type):
        """Compute evaluation metrics from prediction results."""
        df = pd.DataFrame(results)

        metrics = {
            "model": model_type,
            "n_matches": len(df),
            "mae": float(np.mean(np.abs(
                df["predicted_total"] - df["actual_total"]
            ))),
            "rmse": float(np.sqrt(np.mean(
                (df["predicted_total"] - df["actual_total"]) ** 2
            ))),
        }

        # Brier score and accuracy per O/U threshold
        ou_metrics = {}
        for t in thresholds:
            key = str(t)
            pred_col = f"ou_{key}_pred"
            actual_col = f"ou_{key}_actual"

            if pred_col not in df.columns:
                continue

            valid = df[[pred_col, actual_col]].dropna()
            if len(valid) == 0:
                continue

            brier = float(np.mean((valid[pred_col] - valid[actual_col]) ** 2))

            # Accuracy when we pick the side with >50% probability
            predicted_over = (valid[pred_col] > 0.5).astype(int)
            accuracy = float(np.mean(predicted_over == valid[actual_col]))

            ou_metrics[key] = {
                "brier_score": round(brier, 4),
                "accuracy": round(accuracy, 4),
                "actual_over_rate": round(float(valid[actual_col].mean()), 4),
            }

        metrics["over_under"] = ou_metrics
        return metrics

    def print_report(self, metrics):
        """Print a human-readable evaluation report."""
        print(f"\n{'='*50}")
        print(f"  {metrics['model'].upper()} MODEL EVALUATION")
        print(f"{'='*50}")
        print(f"  Matches evaluated: {metrics['n_matches']}")
        print(f"  MAE (total):  {metrics['mae']:.2f}")
        print(f"  RMSE (total): {metrics['rmse']:.2f}")
        print()

        if "over_under" in metrics:
            print(f"  {'Threshold':<12} {'Brier':<10} {'Accuracy':<10} {'Actual O%'}")
            print(f"  {'-'*44}")
            for threshold, data in metrics["over_under"].items():
                print(
                    f"  O/U {threshold:<7} "
                    f"{data['brier_score']:<10.4f} "
                    f"{data['accuracy']:<10.1%} "
                    f"{data['actual_over_rate']:.1%}"
                )
        print()
