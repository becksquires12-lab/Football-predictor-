"""Feature adjustments for international matches.

International matches lack regular league form, so we use different
signals: FIFA ranking, recent national team results, and competition
stage weighting.
"""

import numpy as np


# Approximate FIFA ranking tiers and their goal-scoring tendencies
# Based on historical data analysis of international matches
RANKING_TIERS = {
    "elite": {"rank_range": (1, 10), "attack_mult": 1.3, "defense_mult": 0.8},
    "strong": {"rank_range": (11, 30), "attack_mult": 1.1, "defense_mult": 0.95},
    "average": {"rank_range": (31, 60), "attack_mult": 1.0, "defense_mult": 1.0},
    "below_avg": {"rank_range": (61, 100), "attack_mult": 0.9, "defense_mult": 1.1},
    "weak": {"rank_range": (101, 211), "attack_mult": 0.75, "defense_mult": 1.3},
}

# International match averages (historically lower than club football)
INTL_AVERAGES = {
    "avg_goals_per_match": 2.5,
    "avg_home_goals": 1.4,
    "avg_away_goals": 1.1,
    "avg_corners_per_match": 9.5,
}

# Competition stage multipliers for corners (knockout = more attacking = more corners)
STAGE_MULTIPLIERS = {
    "group": 1.0,
    "round_of_16": 1.05,
    "quarter_final": 1.08,
    "semi_final": 1.1,
    "final": 1.15,
    "friendly": 0.9,
    "qualifier": 0.95,
}


def get_ranking_tier(ranking):
    """Determine the tier for a given FIFA ranking."""
    if ranking is None:
        return RANKING_TIERS["average"]
    for tier_data in RANKING_TIERS.values():
        low, high = tier_data["rank_range"]
        if low <= ranking <= high:
            return tier_data
    return RANKING_TIERS["weak"]


def adjust_features_for_international(features, home_ranking=None,
                                       away_ranking=None, stage="group"):
    """Adjust a feature vector for international match context.

    Club-level features are blended with ranking-based estimates since
    national team form is sparse (few matches per year).

    Args:
        features: Base feature dict from FeatureEngineer
        home_ranking: FIFA ranking of home team (1-211)
        away_ranking: FIFA ranking of away team (1-211)
        stage: Competition stage (group, quarter_final, etc.)

    Returns:
        Adjusted feature dict.
    """
    adjusted = features.copy()
    home_tier = get_ranking_tier(home_ranking)
    away_tier = get_ranking_tier(away_ranking)

    # Adjust expected goals using ranking multipliers
    base_home = INTL_AVERAGES["avg_home_goals"]
    base_away = INTL_AVERAGES["avg_away_goals"]

    ranking_home_lambda = base_home * home_tier["attack_mult"] * away_tier["defense_mult"]
    ranking_away_lambda = base_away * away_tier["attack_mult"] * home_tier["defense_mult"]

    # If we have club-level features, blend them (40% club, 60% ranking)
    # because international form is very different from club form
    if "home_expected_goals" in features:
        adjusted["home_expected_goals"] = (
            0.4 * features["home_expected_goals"] + 0.6 * ranking_home_lambda
        )
        adjusted["away_expected_goals"] = (
            0.4 * features["away_expected_goals"] + 0.6 * ranking_away_lambda
        )
    else:
        adjusted["home_expected_goals"] = ranking_home_lambda
        adjusted["away_expected_goals"] = ranking_away_lambda

    # Stage adjustment for corners
    stage_mult = STAGE_MULTIPLIERS.get(stage, 1.0)
    if "expected_total_corners" in adjusted:
        adjusted["expected_total_corners"] *= stage_mult
    adjusted["stage_multiplier"] = stage_mult

    # Use international averages
    adjusted["league_avg_goals"] = INTL_AVERAGES["avg_goals_per_match"]
    adjusted["is_international"] = True

    return adjusted


def estimate_international_lambdas(home_ranking, away_ranking, stage="group"):
    """Estimate Poisson lambdas for an international match using only rankings.

    Useful when no historical match data is available for these teams.
    """
    home_tier = get_ranking_tier(home_ranking)
    away_tier = get_ranking_tier(away_ranking)

    home_lambda = (
        INTL_AVERAGES["avg_home_goals"]
        * home_tier["attack_mult"]
        * away_tier["defense_mult"]
    )
    away_lambda = (
        INTL_AVERAGES["avg_away_goals"]
        * away_tier["attack_mult"]
        * home_tier["defense_mult"]
    )

    # Clamp to reasonable range
    home_lambda = np.clip(home_lambda, 0.3, 4.0)
    away_lambda = np.clip(away_lambda, 0.2, 3.5)

    return home_lambda, away_lambda
