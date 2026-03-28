"""Corner-specific feature extraction."""

import numpy as np


def compute_corner_features(home_stats, away_stats, league_avgs):
    """Extract corner-related features for the corners model.

    Corner counts correlate with attacking style, possession, shots,
    and the opponent's defensive approach. Teams that press high and
    cross frequently tend to win more corners.

    Returns:
        Dict of corner features, or None if corner data is insufficient.
    """
    avg_corners = league_avgs.get("avg_corners_per_match")

    # Need at least corner averages to build features
    if (home_stats.get("avg_corners_won") is None or
            away_stats.get("avg_corners_won") is None):
        return None

    half_avg = avg_corners / 2.0 if avg_corners and avg_corners > 0 else 5.0

    features = {
        # Direct corner stats
        "home_avg_corners_won": home_stats["avg_corners_won"],
        "home_avg_corners_conceded": home_stats["avg_corners_conceded"],
        "away_avg_corners_won": away_stats["avg_corners_won"],
        "away_avg_corners_conceded": away_stats["avg_corners_conceded"],

        # Corner strength relative to league average
        "home_corner_attack": home_stats["avg_corners_won"] / half_avg,
        "away_corner_attack": away_stats["avg_corners_won"] / half_avg,

        # Expected total corners for this matchup
        "expected_total_corners": (
            home_stats["avg_corners_won"] +
            home_stats["avg_corners_conceded"] +
            away_stats["avg_corners_won"] +
            away_stats["avg_corners_conceded"]
        ) / 2.0,
    }

    # Possession-based corner indicators
    # Teams with high possession tend to win more corners
    if home_stats.get("avg_possession") is not None:
        features["home_possession"] = home_stats["avg_possession"]
    if away_stats.get("avg_possession") is not None:
        features["away_possession"] = away_stats["avg_possession"]

    # Shot volume correlates with corners (blocked shots -> corners)
    if home_stats.get("avg_shots") is not None:
        features["home_avg_shots"] = home_stats["avg_shots"]
    if away_stats.get("avg_shots") is not None:
        features["away_avg_shots"] = away_stats["avg_shots"]

    # Corner ratio: corners per shot (indicates crossing/wing play style)
    if (home_stats.get("avg_shots") and home_stats["avg_shots"] > 0):
        features["home_corners_per_shot"] = (
            home_stats["avg_corners_won"] / home_stats["avg_shots"]
        )
    if (away_stats.get("avg_shots") and away_stats["avg_shots"] > 0):
        features["away_corners_per_shot"] = (
            away_stats["avg_corners_won"] / away_stats["avg_shots"]
        )

    return features


def estimate_total_corners(home_stats, away_stats, league_avgs):
    """Quick estimate of expected total corners without the full model.

    Uses a weighted average of team corner rates and league average.

    Returns:
        Float estimate of total corners.
    """
    avg_corners = league_avgs.get("avg_corners_per_match", 10.0)

    if (home_stats.get("avg_corners_won") is None or
            away_stats.get("avg_corners_won") is None):
        return avg_corners

    # Simple estimate: average of (team corners won + opponent corners conceded)
    home_expected = (
        home_stats["avg_corners_won"] + away_stats["avg_corners_conceded"]
    ) / 2.0
    away_expected = (
        away_stats["avg_corners_won"] + home_stats["avg_corners_conceded"]
    ) / 2.0

    team_estimate = home_expected + away_expected

    # Blend with league average (80% team-based, 20% league average)
    return 0.8 * team_estimate + 0.2 * avg_corners
