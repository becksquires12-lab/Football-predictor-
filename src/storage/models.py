"""SQLAlchemy ORM models for football match data."""

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Team(Base):
    __tablename__ = "teams"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    country = Column(String)
    league = Column(String)
    football_data_id = Column(Integer, unique=True)
    api_football_id = Column(Integer, unique=True)
    fbref_name = Column(String)

    __table_args__ = (UniqueConstraint("name", "country", name="uq_team_name_country"),)

    def __repr__(self):
        return f"<Team(name='{self.name}', league='{self.league}')>"


class Match(Base):
    __tablename__ = "matches"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(DateTime, nullable=False)
    league = Column(String, nullable=False)
    season = Column(String, nullable=False)  # e.g. "2024-25"
    matchday = Column(Integer)
    is_international = Column(Boolean, default=False)

    home_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)
    away_team_id = Column(Integer, ForeignKey("teams.id"), nullable=False)

    home_goals = Column(Integer)
    away_goals = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)

    status = Column(String, default="SCHEDULED")  # SCHEDULED, FINISHED, POSTPONED

    home_team = relationship("Team", foreign_keys=[home_team_id])
    away_team = relationship("Team", foreign_keys=[away_team_id])

    stats = relationship("MatchStats", back_populates="match", uselist=False)

    __table_args__ = (
        UniqueConstraint(
            "date", "home_team_id", "away_team_id", name="uq_match_fixture"
        ),
    )

    @property
    def total_goals(self):
        if self.home_goals is not None and self.away_goals is not None:
            return self.home_goals + self.away_goals
        return None

    @property
    def total_corners(self):
        if self.home_corners is not None and self.away_corners is not None:
            return self.home_corners + self.away_corners
        return None

    def __repr__(self):
        return (
            f"<Match({self.home_team_id} vs {self.away_team_id}, "
            f"{self.date.strftime('%Y-%m-%d')})>"
        )


class MatchStats(Base):
    __tablename__ = "match_stats"

    id = Column(Integer, primary_key=True, autoincrement=True)
    match_id = Column(Integer, ForeignKey("matches.id"), unique=True, nullable=False)

    # Expected goals
    home_xg = Column(Float)
    away_xg = Column(Float)

    # Shots
    home_shots = Column(Integer)
    away_shots = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)

    # Possession (percentage, 0-100)
    home_possession = Column(Float)
    away_possession = Column(Float)

    # Fouls
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)

    # Cards
    home_yellow_cards = Column(Integer)
    away_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    away_red_cards = Column(Integer)

    match = relationship("Match", back_populates="stats")

    def __repr__(self):
        return f"<MatchStats(match_id={self.match_id})>"
