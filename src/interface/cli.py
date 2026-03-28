"""Click-based CLI for the football predictor."""

import sys
import click

from src.utils.config_loader import get_config
from src.utils.logger import setup_logger, get_logger
from src.utils.constants import LEAGUE_CODES, INTERNATIONAL_COMPS
from src.storage.database import init_db


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
def cli(debug):
    """Football Predictor - Goals & Corners Over/Under Predictions"""
    level = "DEBUG" if debug else "INFO"
    setup_logger(level=level)
    config = get_config()
    init_db(config.get("database", {}).get("path", "data/football_predictor.db"))


@cli.command()
@click.argument("home_team")
@click.argument("away_team")
@click.option("--league", "-l", default="EPL", help="League key (e.g. EPL, LA_LIGA)")
@click.option("--international", "-i", is_flag=True, help="International match mode")
def predict(home_team, away_team, league, international):
    """Predict goals and corners for a match.

    Example: football-predictor predict "Arsenal" "Chelsea" --league EPL
    """
    from src.features.engineer import FeatureEngineer
    from src.features.international import adjust_features_for_international
    from src.predictions.over_under import OverUnderPredictor
    from src.interface.formatters import format_prediction
    from src.storage.database import get_session
    from src.storage.models import Team

    logger = get_logger("cli")
    engineer = FeatureEngineer()

    # Load match data
    if international:
        df = engineer.load_matches_df(is_international=True)
    else:
        df = engineer.load_matches_df(league=league)

    if len(df) == 0:
        click.echo(
            f"No match data found for {league}. "
            f"Run 'update-data --league {league}' first."
        )
        sys.exit(1)

    # Find team IDs
    with get_session() as session:
        home = session.query(Team).filter(
            Team.name.ilike(f"%{home_team}%")
        ).first()
        away = session.query(Team).filter(
            Team.name.ilike(f"%{away_team}%")
        ).first()

    if not home:
        click.echo(f"Team not found: '{home_team}'. Run 'list-teams' to see available teams.")
        sys.exit(1)
    if not away:
        click.echo(f"Team not found: '{away_team}'. Run 'list-teams' to see available teams.")
        sys.exit(1)

    # Compute features
    from datetime import datetime
    league_avgs = engineer.compute_league_averages(df)
    home_stats = engineer.compute_team_stats(df, home.id, datetime.now())
    away_stats = engineer.compute_team_stats(df, away.id, datetime.now())

    if home_stats is None:
        click.echo(f"Insufficient data for {home_team}. Need at least 5 matches.")
        sys.exit(1)
    if away_stats is None:
        click.echo(f"Insufficient data for {away_team}. Need at least 5 matches.")
        sys.exit(1)

    features = engineer.build_feature_vector(home_stats, away_stats, league_avgs)
    if international:
        features = adjust_features_for_international(features)

    # Generate predictions
    predictor = OverUnderPredictor()
    predictor.load_models()

    prediction = predictor.predict(features, home_stats, away_stats, league_avgs)

    # Display league name
    league_name = league
    if league in LEAGUE_CODES:
        league_name = LEAGUE_CODES[league]["name"]
    elif league in INTERNATIONAL_COMPS:
        league_name = INTERNATIONAL_COMPS[league]["name"]

    output = format_prediction(prediction, home.name, away.name, league_name)
    click.echo(output)

    # Show head-to-head if available
    h2h = engineer.head_to_head(df, home.id, away.id)
    if h2h["matches"] > 0:
        click.echo(f"  Head-to-Head (last {h2h['matches']} meetings):")
        if h2h["avg_total_goals"] is not None:
            click.echo(f"    Avg goals: {h2h['avg_total_goals']:.1f}")
        if h2h["avg_total_corners"] is not None:
            click.echo(f"    Avg corners: {h2h['avg_total_corners']:.1f}")
        click.echo()


@cli.command("update-data")
@click.option("--league", "-l", default=None, help="Specific league to update (default: all)")
@click.option("--season", "-s", default=None, help="Season year (e.g. 2024)")
def update_data(league, season):
    """Fetch latest match data from APIs and scrapers."""
    from src.data.api_football_data import FootballDataClient
    from src.storage.database import get_session
    from src.storage.models import Team, Match

    logger = get_logger("cli")
    config = get_config()

    api_key = config.get("api_keys", {}).get("football_data")
    if not api_key or api_key == "your_key_here":
        click.echo(
            "No football-data.org API key configured. "
            "Set FOOTBALL_DATA_API_KEY env var or update config/settings.yaml"
        )
        sys.exit(1)

    leagues = [league] if league else list(LEAGUE_CODES.keys())
    if season is None:
        # Default to current season
        from datetime import datetime
        year = datetime.now().year
        season = year if datetime.now().month >= 8 else year - 1

    client = FootballDataClient(api_key)

    for lg in leagues:
        click.echo(f"Fetching data for {lg} ({season})...")
        try:
            matches = client.get_matches(lg, season)
            _store_matches(matches)
            click.echo(f"  Stored {len(matches)} matches for {lg}")
        except Exception as e:
            click.echo(f"  Error fetching {lg}: {e}")

    click.echo("Data update complete.")


@cli.command()
@click.option("--model", "-m", type=click.Choice(["goals", "corners", "all"]), default="all")
def train(model):
    """Train prediction models on stored data."""
    from src.predictions.trainer import ModelTrainer

    trainer = ModelTrainer()

    if model in ("goals", "all"):
        click.echo("Training goals model...")
    if model in ("corners", "all"):
        click.echo("Training corners model...")

    trainer.train_all()
    click.echo("Training complete.")


@cli.command()
@click.option("--league", "-l", default="EPL", help="League to evaluate")
@click.option("--season", "-s", default=None, help="Specific season to evaluate")
def evaluate(league, season):
    """Evaluate model accuracy against historical data."""
    from src.predictions.evaluator import ModelEvaluator
    from src.interface.formatters import format_evaluation

    evaluator = ModelEvaluator()

    click.echo(f"Evaluating models for {league}...")

    goals_metrics = evaluator.evaluate_goals(league, season)
    if "error" not in goals_metrics:
        click.echo(format_evaluation(goals_metrics))
    else:
        click.echo(f"  Goals: {goals_metrics['error']}")

    corners_metrics = evaluator.evaluate_corners(league, season)
    if "error" not in corners_metrics:
        click.echo(format_evaluation(corners_metrics))
    else:
        click.echo(f"  Corners: {corners_metrics['error']}")


@cli.command("list-leagues")
def list_leagues():
    """Show available leagues and competitions."""
    click.echo("\nLeagues:")
    for key, info in LEAGUE_CODES.items():
        click.echo(f"  {key:<15} {info['name']} ({info['country']})")

    click.echo("\nInternational:")
    for key, info in INTERNATIONAL_COMPS.items():
        click.echo(f"  {key:<15} {info['name']}")
    click.echo()


@cli.command("list-teams")
@click.option("--league", "-l", default=None, help="Filter by league")
def list_teams(league):
    """Show teams in the database."""
    from src.storage.database import get_session
    from src.storage.models import Team

    with get_session() as session:
        query = session.query(Team)
        if league:
            query = query.filter(Team.league == league)
        teams = query.order_by(Team.name).all()

    if not teams:
        click.echo("No teams found. Run 'update-data' first.")
        return

    click.echo(f"\nTeams ({len(teams)}):")
    for team in teams:
        click.echo(f"  {team.name:<30} {team.league or ''}")
    click.echo()


def _store_matches(matches):
    """Store fetched matches in the database."""
    from src.storage.database import get_session
    from src.storage.models import Team, Match

    with get_session() as session:
        for m in matches:
            # Ensure teams exist
            home_team = _get_or_create_team(
                session, m["home_team"], m.get("league"),
                football_data_id=m.get("home_team_id"),
            )
            away_team = _get_or_create_team(
                session, m["away_team"], m.get("league"),
                football_data_id=m.get("away_team_id"),
            )

            # Check if match already exists
            existing = session.query(Match).filter(
                Match.home_team_id == home_team.id,
                Match.away_team_id == away_team.id,
                Match.date == m["date"],
            ).first()

            if existing:
                # Update with latest data
                existing.home_goals = m.get("home_goals") or existing.home_goals
                existing.away_goals = m.get("away_goals") or existing.away_goals
                existing.home_corners = m.get("home_corners") or existing.home_corners
                existing.away_corners = m.get("away_corners") or existing.away_corners
                existing.status = m.get("status", existing.status)
            else:
                match = Match(
                    date=m["date"],
                    league=m.get("league", ""),
                    season=m.get("season", ""),
                    matchday=m.get("matchday"),
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    home_goals=m.get("home_goals"),
                    away_goals=m.get("away_goals"),
                    home_corners=m.get("home_corners"),
                    away_corners=m.get("away_corners"),
                    status=m.get("status", "SCHEDULED"),
                    is_international=m.get("is_international", False),
                )
                session.add(match)


def _get_or_create_team(session, name, league, **kwargs):
    """Get an existing team or create a new one."""
    from src.storage.models import Team

    if not name:
        return None

    team = session.query(Team).filter(Team.name == name).first()
    if team:
        return team

    team = Team(name=name, league=league)
    for key, value in kwargs.items():
        if value is not None and hasattr(team, key):
            setattr(team, key, value)
    session.add(team)
    session.flush()
    return team


def main():
    cli()


if __name__ == "__main__":
    main()
