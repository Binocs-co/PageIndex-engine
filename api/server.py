import os

from fastapi import FastAPI

from api.routers import api_router

_ENV = os.getenv("ENV", "local")
_ENV_URL = os.getenv("ENV_URL", "")
_SWAGGER_ENVS = {"local", "development"}


def create_app() -> FastAPI:
    app = FastAPI(
        title="PageIndex API",
        version="1.0.0",
        description="HTTP API for indexing markdown documents using the PageIndex PDF pipeline.",
        docs_url="/docs" if _ENV in _SWAGGER_ENVS else None,
        redoc_url="/redoc" if _ENV in _SWAGGER_ENVS else None,
        openapi_url="/openapi.json" if _ENV in _SWAGGER_ENVS else None,
        root_path=_ENV_URL,
    )
    app.include_router(api_router, prefix="/api/v1")
    return app


app = create_app()
