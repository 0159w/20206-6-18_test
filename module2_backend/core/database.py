"""Database session management for FastAPI."""

from sqlmodel import create_engine, Session

from module2_backend.core.config import DATABASE_URL

engine = create_engine(DATABASE_URL, echo=False)


def get_session():
    """Dependency: yields a DB session and closes after request."""
    with Session(engine) as session:
        yield session
