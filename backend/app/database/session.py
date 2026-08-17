"""Database engine, session factory, and declarative base.

This module wires SQLAlchemy to the SQLite database defined in
``core.config``. It exposes:

* ``engine``        - the SQLAlchemy engine bound to SQLite.
* ``SessionLocal``  - a factory that produces database sessions.
* ``Base``          - the declarative base every ORM model inherits from.
* ``get_db``        - a FastAPI dependency that yields a session per request.
* ``init_db``       - creates all tables (called once on startup).
"""

from __future__ import annotations

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings

# ``check_same_thread=False`` is required for SQLite when the connection is
# shared across threads, which happens under FastAPI/Uvicorn.
engine = create_engine(
    settings.database_url,
    connect_args={"check_same_thread": False},
    echo=False,
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    """Declarative base class shared by all ORM models."""


def get_db() -> Generator[Session, None, None]:
    """Yield a database session and guarantee it is closed afterwards.

    Used as a FastAPI dependency so each request gets its own session.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Create database tables for all imported models.

    Models must be imported before this runs so they are registered on
    ``Base.metadata``.
    """
    # Local import avoids circular imports at module load time.
    from app.models import dataset  # noqa: F401

    Base.metadata.create_all(bind=engine)
