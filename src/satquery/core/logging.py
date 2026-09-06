"""structlog configuration — JSON on the wire, key/value in a dev console."""

from __future__ import annotations

import logging
import sys

import structlog

from satquery.core.config import Settings, get_settings


def configure_logging(settings: Settings | None = None) -> None:
    """Configure ``structlog`` and the stdlib root logger.

    Idempotent: calling it more than once simply re-applies the same configuration.
    """
    settings = settings or get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level, force=True)

    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if settings.log_json
        else structlog.dev.ConsoleRenderer(colors=True)
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    if not structlog.is_configured():
        configure_logging()
    return structlog.stdlib.get_logger(name)
