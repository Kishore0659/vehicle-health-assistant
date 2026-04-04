"""
models.py
---------
SQLite database setup using SQLAlchemy ORM.
Stores user vehicle profiles and query history.
"""

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime, timezone
import os

# ── Database path ────────────────────────────────────────────────────────────
DB_DIR  = os.path.join(os.path.dirname(__file__), "..", "db")
os.makedirs(DB_DIR, exist_ok=True)
DB_PATH = os.path.join(DB_DIR, "vehicle_health.db")

DATABASE_URL = f"sqlite:///{DB_PATH}"

engine       = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base         = declarative_base()


# ─────────────────────────────────────────────────────────────────────────────
# ORM Models
# ─────────────────────────────────────────────────────────────────────────────

class VehicleProfile(Base):
    """Stores a user's vehicle details."""
    __tablename__ = "vehicle_profiles"

    id           = Column(Integer, primary_key=True, index=True)
    owner_name   = Column(String(100), nullable=False)
    vehicle_type = Column(String(20),  nullable=False)   # bike / car / suv
    vehicle_model= Column(String(100), nullable=True)
    km_driven    = Column(Integer,     nullable=True)
    created_at   = Column(DateTime,    default=lambda: datetime.now(timezone.utc))


class QueryHistory(Base):
    """Logs every diagnosis request and its result."""
    __tablename__ = "query_history"

    id           = Column(Integer, primary_key=True, index=True)
    vehicle_type = Column(String(20), nullable=False)
    query        = Column(Text,       nullable=False)
    severity     = Column(String(20), nullable=True)
    summary      = Column(Text,       nullable=True)
    created_at   = Column(DateTime,   default=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────────────────────────────────────
# Create all tables on import
# ─────────────────────────────────────────────────────────────────────────────

def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    """Dependency injector for FastAPI routes."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
