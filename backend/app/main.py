"""FastAPI application entry point.

Creates and configures the FastAPI app, initializes the database, mounts
routers, and exposes a health-check endpoint.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.datasets import router as datasets_router
from app.core.config import settings
from app.database.session import init_db


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup/shutdown tasks. Ensures tables exist before serving."""
    init_db()
    yield


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

# Allow the Streamlit frontend (a separate origin) to call the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount versioned API routes, e.g. /api/v1/datasets/upload
app.include_router(datasets_router, prefix=settings.API_V1_PREFIX)


@app.get("/", tags=["health"], summary="Health check")
def health_check() -> dict[str, str]:
    """Simple liveness probe used to confirm the backend is running."""
    return {"status": "ok", "app": settings.APP_NAME, "version": settings.APP_VERSION}
