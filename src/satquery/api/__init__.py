"""HTTP surface: the FastAPI application and its versioned routers."""

from satquery.api.app import app, create_app

__all__ = ["app", "create_app"]
