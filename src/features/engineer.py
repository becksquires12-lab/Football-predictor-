"""Main feature engineering pipeline for match predictions."""

import pandas as pd
import numpy as np

from src.storage.database import get_session
from src.storage.models import Match, Team, MatchStats
from src.utils.constants import DEFAULT_ROLLING_WINDOW, MIN_MATCHES_REQUIRED
from src.utils.logger import get_logger

logger = get_logger("features")


class FeatureEngineer:
    """Compute features from historical match data for prediction models."""

    def __init__(self, rolling_window=DEFAULT_ROLLING_WINDOW):
        self.window = rolling_window

    def load_matches_df(self, league=None, is_international=False):
        """Load finished matches from the database into a DataFrame."""
        with get_session() as session:
            query = session.query(Match).filter(Match.status == "FINISHED")

            if league:
                query = query.filter(Match.league == league)
            if is_international:
                query = query.filter(Match.is_international == True)

            matches = query.order_by(Match.date).all()

            rows = []
            for m in matches:
                row = {
                    "match_id": m.id,
                    "date": m.date,
                    "league": m.league,
                    "season": m.season,
                    "home_team_id": m.home_team_id,
                    "away_team_id": m.away_team_id,
                    "home_team": m.home_team.name if m.home_team else None,
                    "away_team": m.away_team.name if m.away_team else None,
                    "home_goals": m.home_goals,
                    "away_goals": m.away_goals,
                    "home_corners": m.home_corners,
                    "away_corners": m.away_corners,
                    "total_goals": m.total_goals,
                    "total_corners": m.total_corners,
                }
                if m.stats:
                    row.update({
                        "home_xg": m.stats.home_xg,
                        "away_xg": m.stats.away_xg,
                        "home_shots": m.stats.home_shots,
                        "away_shots": m.stats.away_shots,
                        "home_possession": m.stats.home_possession,
                        "away_possession": m.stats.away_possession,
                    })
                rows.append(row)

            return pd.DataFrame(rows)

    def compute_team_stats(self, df, team_id, before_date):
        """Compute rolling stats for a team up to a given date.

        Returns dict with averages for goals scored/conceded, corners, shots, etc.
        """
        # Get matches where this team played (home or away)
        home_mask = (df["home_team_id"] == team_id) & (df["date"] < before_date)
        away_mask = (df["away_team_id"] == team_id) & (df["date"] < before_date)

        home_matches = df[home_mask].tail(self.window)
        away_matches = df[away_mask].tail(self.window)

        total_matches = len(home_matches) + len(away_matches)
        if total_matches < MIN_MATCHES_REQUIRED:
            return None

        stats = {}

        # Goals scored and conceded
        goals_scored = pd.concat([
            home_matches["home_goals"],
            away_matches["away_goals"],
        ])
        goals_conceded = pd.concat([
            home_matches["away_goals"],
            away_matches["home_goals"],
        ])
        stats["avg_goals_scored"] = goals_scored.mean()
        stats["avg_goals_conceded"] = goals_conceded.mean()

        # Home-specific stats
        if len(home_matches) > 0:
            stats["avg_home_goals_scored"] = home_matches["home_goals"].mean()
            stats["avg_home_goals_conceded"] = home_matches["away_goals"].mean()
        else:
            stats["avg_home_goals_scored"] = stats["avg_goals_scored"]
            stats["avg_home_goals_conceded"] = stats["avg_goals_conceded"]

        # Away-specific stats
        if len(away_matches) > 0:
            stats["avg_away_goals_scored"] = away_matches["away_goals"].mean()
            stats["avg_away_goals_conceded"] = away_matches["home_goals"].mean()
        else:
            stats["avg_away_goals_scored"] = stats["avg_goals_scored"]
            stats["avg_away_goals_conceded"] = stats["avg_goals_conceded"]

        # Corners
        home_corners_won = home_matches["home_corners"].dropna()
        away_corners_won = away_matches["away_corners"].dropna()
        home_corners_conceded = home_matches["away_corners"].dropna()
        away_corners_conceded = away_matches["home_corners"].dropna()

        all_corners_won = pd.concat([home_corners_won, away_corners_won])
        all_corners_conceded = pd.concat([home_corners_conceded, away_corners_conceded])

        stats["avg_corners_won"] = all_corners_won.mean() if len(all_corners_won) > 0 else None
        stats["avg_corners_conceded"] = all_corners_conceded.mean() if len(all_corners_conceded) > 0 else None

        # xG if available
        home_xg = home_matches.get("home_xg", pd.Series(dtype=float)).dropna()
        away_xg = away_matches.get("away_xg", pd.Series(dtype=float)).dropna()
        all_xg = pd.concat([home_xg, away_xg])
        stats["avg_xg"] = all_xg.mean() if len(all_xg) > 0 else None

        # Shots if available
        home_shots = home_matches.get("home_shots", pd.Series(dtype=float)).dropna()
        away_shots = away_matches.get("away_shots", pd.Series(dtype=float)).dropna()
        all_shots = pd.concat([home_shots, away_shots])
        stats["avg_shots"] = all_shots.mean() if len(all_shots) > 0 else None

        # Possession
        home_poss = home_matches.get("home_possession", pd.Series(dtype=float)).dropna()
        away_poss = away_matches.get("away_possession", pd.Series(dtype=float)).dropna()
        all_poss = pd.concat([home_poss, away_poss])
        stats["avg_possession"] = all_poss.mean() if len(all_poss) > 0 else None

        # Form (last 5 results: W=3, D=1, L=0)
        recent_home = home_matches.tail(5)
        recent_away = away_matches.tail(5)
        form_points = []
        for _, row in recent_home.iterrows():
            if row["home_goals"] > row["away_goals"]:
                form_points.append(3)
            elif row["home_goals"] == row["away_goals"]:
                form_points.append(1)
            else:
                form_points.append(0)
        for _, row in recent_away.iterrows():
            if row["away_goals"] > row["home_goals"]:
                form_points.append(3)
            elif row["away_goals"] == row["home_goals"]:
                form_points.append(1)
            else:
                form_points.append(0)

        stats["form"] = np.mean(form_points[-5:]) if form_points else 1.0
        stats["matches_played"] = total_matches

        return stats

    def compute_league_averages(self, df):
        """Compute league-wide averages for normalization."""
        return {
            "avg_goals_per_match": df["total_goals"].mean(),
            "avg_home_goals": df["home_goals"].mean(),
            "avg_away_goals": df["away_goals"].mean(),
            "avg_corners_per_match": df["total_corners"].dropna().mean(),
        }

    def build_feature_vector(self, home_stats, away_stats, league_avgs):
        """Build a feature vector for a match prediction.

        Returns:
            Dict of features ready for model input.
        """
        if home_stats is None or away_stats is None:
            return None

        avg_goals = league_avgs.get("avg_goals_per_match", 2.7)
        avg_home = league_avgs.get("avg_home_goals", 1.5)
        avg_away = league_avgs.get("avg_away_goals", 1.2)

        # Attack and defense strength (relative to league average)
        home_attack = home_stats["avg_home_goals_scored"] / avg_home if avg_home > 0 else 1.0
        home_defense = home_stats["avg_home_goals_conceded"] / avg_away if avg_away > 0 else 1.0
        away_attack = away_stats["avg_away_goals_scored"] / avg_away if avg_away > 0 else 1.0
        away_defense = away_stats["avg_away_goals_conceded"] / avg_home if avg_home > 0 else 1.0

        features = {
            # Goal prediction features
            "home_attack_strength": home_attack,
            "home_defense_strength": home_defense,
            "away_attack_strength": away_attack,
            "away_defense_strength": away_defense,
            "home_avg_goals_scored": home_stats["avg_goals_scored"],
            "home_avg_goals_conceded": home_stats["avg_goals_conceded"],
            "away_avg_goals_scored": away_stats["avg_goals_scored"],
            "away_avg_goals_conceded": away_stats["avg_goals_conceded"],
            "home_form": home_stats["form"],
            "away_form": away_stats["form"],
            "league_avg_goals": avg_goals,
            # Expected lambda for Poisson
            "home_expected_goals": home_attack * away_defense * avg_home,
            "away_expected_goals": away_attack * home_defense * avg_away,
        }

        # Corner features (if available)
        avg_corners = league_avgs.get("avg_corners_per_match")
        if (home_stats.get("avg_corners_won") is not None and
                away_stats.get("avg_corners_won") is not None and
                avg_corners is not None and avg_corners > 0):
            features.update({
                "home_avg_corners_won": home_stats["avg_corners_won"],
                "home_avg_corners_conceded": home_stats["avg_corners_conceded"],
                "away_avg_corners_won": away_stats["avg_corners_won"],
                "away_avg_corners_conceded": away_stats["avg_corners_conceded"],
                "league_avg_corners": avg_corners,
            })

        # Optional advanced features
        if home_stats.get("avg_xg") is not None:
            features["home_avg_xg"] = home_stats["avg_xg"]
        if away_stats.get("avg_xg") is not None:
            features["away_avg_xg"] = away_stats["avg_xg"]
        if home_stats.get("avg_shots") is not None:
            features["home_avg_shots"] = home_stats["avg_shots"]
        if away_stats.get("avg_shots") is not None:
            features["away_avg_shots"] = away_stats["avg_shots"]
        if home_stats.get("avg_possession") is not None:
            features["home_avg_possession"] = home_stats["avg_possession"]
        if away_stats.get("avg_possession") is not None:
            features["away_avg_possession"] = away_stats["avg_possession"]

        return features

    def head_to_head(self, df, team1_id, team2_id, last_n=5):
        """Get head-to-head record between two teams."""
        h2h = df[
            ((df["home_team_id"] == team1_id) & (df["away_team_id"] == team2_id)) |
            ((df["home_team_id"] == team2_id) & (df["away_team_id"] == team1_id))
        ].tail(last_n)

        if len(h2h) == 0:
            return {"avg_total_goals": None, "avg_total_corners": None, "matches": 0}

        return {
            "avg_total_goals": h2h["total_goals"].mean(),
            "avg_total_corners": h2h["total_corners"].dropna().mean() if h2h["total_corners"].notna().any() else None,
            "matches": len(h2h),
        }
