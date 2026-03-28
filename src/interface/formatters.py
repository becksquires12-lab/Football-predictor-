"""Pretty-print prediction tables for terminal output."""

from tabulate import tabulate


def format_prediction(prediction, home_team, away_team, league=None):
    """Format a full prediction as a readable terminal output.

    Args:
        prediction: Dict from OverUnderPredictor.predict()
        home_team: Home team name
        away_team: Away team name
        league: Optional league name for display
    """
    lines = []
    header = f"  {home_team} vs {away_team}"
    if league:
        header += f"  --  {league}"
    lines.append("")
    lines.append("=" * max(50, len(header) + 4))
    lines.append(header)
    lines.append("=" * max(50, len(header) + 4))

    # Goals section
    goals = prediction.get("goals", {})
    if goals:
        lines.append("")
        lines.append("  GOALS")
        lines.append("  " + "-" * 46)

        pred_h = goals.get("predicted_home", "?")
        pred_a = goals.get("predicted_away", "?")
        pred_total = goals.get("predicted_total", "?")
        lines.append(f"  Predicted Score:  {pred_h} - {pred_a}  (Total: {pred_total})")

        # BTTS
        btts = goals.get("btts", {})
        if btts:
            lines.append(
                f"  Both Teams Score: Yes {_pct(btts.get('yes'))}  |  "
                f"No {_pct(btts.get('no'))}"
            )

        # Over/Under table
        ou = goals.get("over_under", {})
        if ou:
            lines.append("")
            ou_rows = []
            for threshold in sorted(ou.keys(), key=float):
                data = ou[threshold]
                ou_rows.append([
                    f"O/U {threshold}",
                    _pct(data["over"]),
                    _bar(data["over"]),
                    _pct(data["under"]),
                ])
            lines.append(tabulate(
                ou_rows,
                headers=["", "Over", "", "Under"],
                tablefmt="simple",
                colalign=("left", "right", "left", "right"),
            ))

    # Corners section
    corners = prediction.get("corners", {})
    if corners and corners.get("predicted_total") is not None:
        lines.append("")
        lines.append("  CORNERS")
        lines.append("  " + "-" * 46)
        lines.append(
            f"  Predicted Total: {corners['predicted_total']}"
        )

        ou = corners.get("over_under", {})
        if ou:
            lines.append("")
            ou_rows = []
            for threshold in sorted(ou.keys(), key=float):
                data = ou[threshold]
                ou_rows.append([
                    f"O/U {threshold}",
                    _pct(data["over"]),
                    _bar(data["over"]),
                    _pct(data["under"]),
                ])
            lines.append(tabulate(
                ou_rows,
                headers=["", "Over", "", "Under"],
                tablefmt="simple",
                colalign=("left", "right", "left", "right"),
            ))
    elif corners.get("note"):
        lines.append("")
        lines.append(f"  CORNERS: {corners['note']}")

    lines.append("")
    return "\n".join(lines)


def format_evaluation(metrics):
    """Format model evaluation metrics as a readable report."""
    lines = []
    lines.append(f"\n{'=' * 50}")
    lines.append(f"  {metrics['model'].upper()} MODEL EVALUATION")
    lines.append(f"{'=' * 50}")
    lines.append(f"  Matches evaluated: {metrics['n_matches']}")
    lines.append(f"  MAE:  {metrics['mae']:.2f}")
    lines.append(f"  RMSE: {metrics['rmse']:.2f}")

    ou = metrics.get("over_under", {})
    if ou:
        lines.append("")
        rows = []
        for threshold in sorted(ou.keys(), key=float):
            data = ou[threshold]
            rows.append([
                f"O/U {threshold}",
                f"{data['brier_score']:.4f}",
                f"{data['accuracy']:.1%}",
                f"{data['actual_over_rate']:.1%}",
            ])
        lines.append(tabulate(
            rows,
            headers=["Threshold", "Brier", "Accuracy", "Actual O%"],
            tablefmt="simple",
        ))

    lines.append("")
    return "\n".join(lines)


def _pct(value):
    """Format a probability as a percentage string."""
    if value is None:
        return "  --  "
    return f"{value * 100:5.1f}%"


def _bar(value, width=15):
    """Create a simple bar visualization."""
    if value is None:
        return ""
    filled = int(value * width)
    return "█" * filled + "░" * (width - filled)
