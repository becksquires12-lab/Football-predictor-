"""Model training, cross-validation, and persistence."""

import numpy as np
import pandas as pd

from src.features.engineer import FeatureEngineer
from src.features.corners import compute_corner_features
from src.features.goals import estimate_poisson_lambdas
from src.predictions.goals_model import GoalsModel
from src.predictions.corners_model import CornersModel
from src.utils.constants import LEAGUE_CODES
from src.utils.logger import get_logger

logger = get_logger("trainer")


class ModelTrainer:
    """Orchestrate training for goals and corners models."""

    def __init__(self):
        self.engineer = FeatureEngineer()
        self.goals_model = GoalsModel()
        self.corners_model = CornersModel()

    def train_all(self, leagues=None):
        """Train both models across specified leagues.

        Args:
            leagues: List of league keys. Defaults to all top 5.
        """
        if leagues is None:
            leagues = list(LEAGUE_CODES.keys())

        logger.info(f"Training models for leagues: {leagues}")

        # Train goals model (calibration per league)
        for league in leagues:
            self._calibrate_goals(league)

        # Train corners model (single model across all leagues)
        self._train_corners(leagues)

        # Save models
        self.goals_model.save()
        self.corners_model.save()

        logger.info("Training complete. Models saved.")

    def _calibrate_goals(self, league):
        """Calibrate the goals model for a specific league."""
        df = self.engineer.load_matches_df(league=league)
        if len(df) < 20:
            logger.warning(f"Insufficient data for {league} ({len(df)} matches)")
            return

        league_avgs = self.engineer.compute_league_averages(df)
        calibration_data = []

        for idx, row in df.iterrows():
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
            if features:
                features["league"] = league
                calibration_data.append({
                    "features": features,
                    "home_goals": row["home_goals"],
                    "away_goals": row["away_goals"],
                })

        if calibration_data:
            self.goals_model.calibrate(league, calibration_data)
            logger.info(
                f"Goals model calibrated for {league} "
                f"({len(calibration_data)} matches)"
            )

    def _train_corners(self, leagues):
        """Train the corners model using data from all leagues."""
        all_features = []
        all_targets = []

        for league in leagues:
            df = self.engineer.load_matches_df(league=league)
            # Filter to matches with corner data
            corner_df = df.dropna(subset=["home_corners", "away_corners"])
            if len(corner_df) < 10:
                logger.warning(f"Insufficient corner data for {league}")
                continue

            league_avgs = self.engineer.compute_league_averages(corner_df)

            for idx, row in corner_df.iterrows():
                home_stats = self.engineer.compute_team_stats(
                    corner_df, row["home_team_id"], row["date"]
                )
                away_stats = self.engineer.compute_team_stats(
                    corner_df, row["away_team_id"], row["date"]
                )

                if home_stats is None or away_stats is None:
                    continue

                corner_feats = compute_corner_features(
                    home_stats, away_stats, league_avgs
                )
                if corner_feats is None:
                    continue

                # Build feature vector from the core corner features
                feature_vec = {
                    k: corner_feats[k]
                    for k in self.corners_model.FEATURE_COLS
                    if k in corner_feats
                }
                # Add optional features that exist
                for col in self.corners_model.OPTIONAL_COLS:
                    if col in corner_feats:
                        feature_vec[col] = corner_feats[col]

                all_features.append(feature_vec)
                all_targets.append(row["total_corners"])

        if len(all_features) < 20:
            logger.warning(
                f"Insufficient corner training data ({len(all_features)} samples)"
            )
            return

        X = pd.DataFrame(all_features).fillna(0)
        y = np.array(all_targets)

        self.corners_model.train(X, y)
        logger.info(f"Corners model trained on {len(all_features)} matches")
