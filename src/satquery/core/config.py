"""Application settings, loaded from the environment and ``.env``."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from satquery.schemas.version import SCHEMA_VERSION


class Settings(BaseSettings):
    """Runtime configuration. Every field is overridable via a ``SATQUERY_*`` env var."""

    model_config = SettingsConfigDict(
        env_prefix="SATQUERY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "SatQuery AI"
    environment: str = Field(default="dev", description="'dev' | 'staging' | 'prod'.")
    version: str = "0.1.0"
    schema_version: str = SCHEMA_VERSION

    api_prefix: str = "/v1"
    cors_allow_origins: list[str] = Field(default_factory=lambda: ["*"])

    log_level: str = "INFO"
    log_json: bool = Field(default=True, description="False renders human-readable console logs.")

    artifact_root: Path = Field(
        default=Path("data/artifacts"),
        description="Root of the content-addressed artifact store served at /v1/artifacts.",
    )

    trace_db: Path = Field(
        default=Path("data/traces.sqlite3"),
        description="SQLite database backing GET /v1/trace/{id}.",
    )

    max_upload_mb: int = Field(default=512, gt=0)
    max_images: int = Field(default=2, gt=0)

    citation_policy: str = Field(
        default="flag",
        description="'flag' keeps unsupported numbers visible; 'strip' removes their sentence.",
    )
    allow_generic_fallback: bool = Field(
        default=True,
        description="False turns an unclassifiable query into 422 QUERY_UNCLASSIFIABLE.",
    )

    # -- VLM serving (Master.md §8 Phase 4) --------------------------------
    #
    # Every one of these is a *declaration of intent*. What the vlm_* tools are
    # actually allowed to do is decided by models.loader.available_backend(),
    # which probes the machine offline: pointing vlm_backend at "hf" on a box
    # with no weights downloaded leaves the tools unavailable and the answers
    # templated, rather than failing a request.

    vlm_backend: str = Field(
        default="auto",
        description="'auto' | 'hf' (transformers/ROCm) | 'llamacpp' (GGUF) | 'none'.",
    )
    vlm_disabled: bool = Field(
        default=False,
        description="True forces every vlm_* tool unavailable — the deterministic-only baseline.",
    )
    vlm_model_id: str = Field(default="Qwen/Qwen3-VL-8B-Instruct")
    vlm_model_path: Path | None = Field(
        default=None,
        description="Local weights directory. Overrides the Hugging Face cache lookup.",
    )
    vlm_gguf_path: Path | None = Field(
        default=None, description="GGUF quantisation for the offline llama.cpp path."
    )
    vlm_server_url: str | None = Field(
        default=None,
        description="llama.cpp server base URL, e.g. http://127.0.0.1:8080.",
    )
    vlm_device: str = Field(
        default="rocm:0", description="Contract device string; 'rocm:N' maps to torch 'cuda:N'."
    )
    vlm_dtype: str = Field(default="bfloat16", description="'bfloat16' is the only one that fits.")
    vlm_vram_budget_gb: float = Field(
        default=22.0, gt=0.0, description="Hard ceiling on the 24 GB card (Master.md §8 Phase 4)."
    )
    vlm_max_new_tokens: int = Field(default=384, gt=0)
    vlm_idle_unload_s: float = Field(
        default=900.0,
        description="Unload the weights after this long idle; 0 keeps them resident.",
    )
    vlm_request_timeout_s: float = Field(default=180.0, gt=0.0)
    vlm_attn_implementation: str | None = Field(
        default=None,
        description="'sdpa' | 'eager' | 'flash_attention_2'. None lets transformers choose.",
    )
    vlm_prompt_version: str = Field(
        default="grounded_v1", description="Key into models.prompts.templates.TEMPLATES."
    )

    @property
    def is_dev(self) -> bool:
        """True when running in the local development environment."""
        return self.environment == "dev"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
