"""FastAPI application entry point — 矿井安全检查系统 API."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from module2_backend.core.config import UPLOAD_DIR
from module2_backend.models.database import create_all_tables
from module2_backend.routers.inspections import router as inspection_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create tables on startup."""
    create_all_tables()
    yield


app = FastAPI(
    title="矿井安全检查系统 API",
    description="后端大模型工程师编程题 — YOLOv8明火检测 + FastAPI后端",
    version="1.0.0",
    lifespan=lifespan,
)

# Serve uploaded photos
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
