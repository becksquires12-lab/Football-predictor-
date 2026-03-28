"""SQLite database connection manager."""

from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from src.storage.models import Base


_engine = None
_SessionFactory = None


def init_db(db_path="data/football_predictor.db"):
    """Initialize the database engine and create tables if needed."""
    global _engine, _SessionFactory

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    _engine = create_engine(f"sqlite:///{path}", echo=False)
    _SessionFactory = sessionmaker(bind=_engine)

    Base.metadata.create_all(_engine)
    return _engine


def get_engine():
    """Get the current database engine, initializing if needed."""
    if _engine is None:
        init_db()
    return _engine


@contextmanager
def get_session():
    """Provide a transactional session scope."""
    if _SessionFactory is None:
        init_db()

    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
