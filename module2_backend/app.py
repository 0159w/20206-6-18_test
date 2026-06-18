"""FastAPI application entry point — 矿井安全检查系统 API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from sqlmodel import Session

from module2_backend.core.config import UPLOAD_DIR
from module2_backend.core.database import engine
from module2_backend.models.database import create_all_tables, Team, Area
from module2_backend.routers.inspections import router as inspection_router


def _seed_initial_data():
    """Populate Team and Area tables with default records if empty."""
    with Session(engine) as session:
        existing_teams = session.query(Team).count()
        if existing_teams == 0:
            session.add(Team(id=1, name="施工一队"))
            session.add(Team(id=2, name="施工二队"))
            session.add(Team(id=3, name="施工三队"))
        existing_areas = session.query(Area).count()
        if existing_areas == 0:
            session.add(Area(id=1, name="采区A"))
            session.add(Area(id=2, name="采区B"))
            session.add(Area(id=3, name="采区C"))
        session.commit()
        print("✓ Seed data inserted: teams, areas")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables and seed data on startup."""
    # Create tables
    create_all_tables()

    # Seed initial data
    _seed_initial_data()

    yield


app = FastAPI(
    title="矿井安全检查系统 API",
    description="后端大模型工程师编程题 — YOLOv8明火检测 + FastAPI后端",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve uploaded photos — ensure directory exists before mounting
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Register routers
app.include_router(inspection_router)


@app.get("/")
def root():
    return {
        "service": "矿井安全检查系统",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
