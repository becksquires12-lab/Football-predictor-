"""Database schema creation and migration helpers."""

from src.storage.database import get_engine
from src.storage.models import Base


def create_all_tables():
    """Create all tables defined in the ORM models."""
    engine = get_engine()
    Base.metadata.create_all(engine)


def drop_all_tables():
    """Drop all tables. Use with caution."""
    engine = get_engine()
    Base.metadata.drop_all(engine)


def reset_database():
    """Drop and recreate all tables."""
    drop_all_tables()
    create_all_tables()
