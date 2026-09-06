"""Versioned API routers, all mounted under the ``/v1`` prefix."""

from satquery.api.routers import analyze, artifacts, health, registry, validate

__all__ = ["analyze", "artifacts", "health", "registry", "validate"]
