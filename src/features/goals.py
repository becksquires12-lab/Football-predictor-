"""Goal-specific feature extraction and Poisson lambda estimation."""

import numpy as np


def estimate_poisson_lambdas(features):
    """Estimate home and away Poisson lambda parameters from features.

    Uses attack/defense strength model:
        home_lambda = home_attack * away_defense * league_avg_home_goals
        away_lambda = away_attack * home_defense * league_avg_away_goals

    These are already computed in the feature vector, but we apply
    xG-based adjustments here if available.

    Returns:
        Tuple of (home_lambda, away_lambda)
    """
    home_lambda = features["home_expected_goals"]
    away_lambda = features["away_expected_goals"]

    # Adjust with xG data if available (blend 70% stats, 30% xG)
    if "home_avg_xg" in features and "away_avg_xg" in features:
        xg_home = features["home_avg_xg"]
        xg_away = features["away_avg_xg"]

        # xG-adjusted lambdas (scale xG by opponent defense strength)
        xg_home_lambda = xg_home * features["away_defense_strength"]
        xg_away_lambda = xg_away * features["home_defense_strength"]

        home_lambda = 0.7 * home_lambda + 0.3 * xg_home_lambda
        away_lambda = 0.7 * away_lambda + 0.3 * xg_away_lambda

    # Form adjustment: scale by form relative to average (1.0 = average form)
    form_adjustment_home = 0.9 + 0.1 * (features["home_form"] / 1.5)
    form_adjustment_away = 0.9 + 0.1 * (features["away_form"] / 1.5)

    home_lambda *= form_adjustment_home
    away_lambda *= form_adjustment_away

    # Ensure reasonable bounds
    home_lambda = np.clip(home_lambda, 0.3, 4.5)
    away_lambda = np.clip(away_lambda, 0.2, 4.0)

    return home_lambda, away_lambda


def compute_goal_features(home_stats, away_stats, league_avgs):
    """Extract goal-specific features for model training.

    Returns:
        Dict of goal-related features.
    """
    avg_home = league_avgs.get("avg_home_goals", 1.5)
    avg_away = league_avgs.get("avg_away_goals", 1.2)

    features = {
        "home_attack_strength": (
            home_stats["avg_home_goals_scored"] / avg_home if avg_home > 0 else 1.0
        ),
        "home_defense_weakness": (
            home_stats["avg_home_goals_conceded"] / avg_away if avg_away > 0 else 1.0
        ),
        "away_attack_strength": (
            away_stats["avg_away_goals_scored"] / avg_away if avg_away > 0 else 1.0
        ),
        "away_defense_weakness": (
            away_stats["avg_away_goals_conceded"] / avg_home if avg_home > 0 else 1.0
        ),
        "home_scoring_rate": home_stats["avg_goals_scored"],
        "away_scoring_rate": away_stats["avg_goals_scored"],
        "home_conceding_rate": home_stats["avg_goals_conceded"],
        "away_conceding_rate": away_stats["avg_goals_conceded"],
    }

    # Add xG features if available
    if home_stats.get("avg_xg") is not None:
        features["home_xg_rate"] = home_stats["avg_xg"]
        features["home_xg_overperformance"] = (
            home_stats["avg_goals_scored"] - home_stats["avg_xg"]
        )
    if away_stats.get("avg_xg") is not None:
        features["away_xg_rate"] = away_stats["avg_xg"]
        features["away_xg_overperformance"] = (
            away_stats["avg_goals_scored"] - away_stats["avg_xg"]
        )

    return features
