"""
Application settings and configuration management using Pydantic.

This module provides validated configuration settings loaded from environment
variables with proper type checking and default values.
"""

import os
from pathlib import Path
from functools import lru_cache
from typing import Optional, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


class LangfuseSettings(BaseModel):
    """Langfuse observability configuration."""
    
    enabled: bool = Field(default=False, description="Enable Langfuse tracing")
    public_key: Optional[str] = Field(default=None, description="Langfuse public key")
    secret_key: Optional[str] = Field(default=None, description="Langfuse secret key")
    host: str = Field(default="https://cloud.langfuse.com", description="Langfuse host URL")
    
    @model_validator(mode='after')
    def validate_keys_if_enabled(self) -> 'LangfuseSettings':
        """Validate that keys are provided if Langfuse is enabled."""
        if self.enabled and (not self.public_key or not self.secret_key):
            raise ValueError("Langfuse public_key and secret_key are required when enabled")
        return self


class RateLimitSettings(BaseModel):
    """Rate limiting configuration for API calls."""
    
    enabled: bool = Field(default=True, description="Enable rate limiting")
    gemini_rpm: int = Field(default=60, ge=1, description="Gemini API requests per minute")
    gemini_tpm: int = Field(default=1_000_000, ge=1, description="Gemini API tokens per minute")
    bigquery_qpm: int = Field(default=100, ge=1, description="BigQuery queries per minute")
    embedding_rpm: int = Field(default=100, ge=1, description="Embedding API requests per minute")


class CacheSettings(BaseModel):
    """Caching configuration."""
    
    enabled: bool = Field(default=True, description="Enable caching")
    embedding_cache_dir: Path = Field(
        default=Path(".cache/embeddings"),
        description="Directory for embedding cache files"
    )
    embedding_cache_ttl_hours: int = Field(
        default=168,  # 1 week
        ge=1,
        description="Embedding cache TTL in hours"
    )
    
    @field_validator('embedding_cache_dir', mode='before')
    @classmethod
    def resolve_cache_path(cls, v: str | Path) -> Path:
        """Resolve cache path relative to project root."""
        path = Path(v) if isinstance(v, str) else v
        if not path.is_absolute():
            # Make relative to project root
            project_root = Path(__file__).parent.parent.parent
            path = project_root / path
        return path


class AgentSettings(BaseModel):
    """Multi-agent architecture configuration."""
    
    # Model configuration
    gemini_model: str = Field(default="gemini-2.5-pro", description="Primary Gemini model")
    
    # Agent-specific models (can override primary)
    investigator_model: Optional[str] = Field(
        default=None,
        description="Model for Investigator agent (defaults to gemini_model)"
    )
    diagnostician_model: Optional[str] = Field(
        default=None,
        description="Model for Diagnostician agent (defaults to gemini_model)"
    )
    fix_proposer_model: Optional[str] = Field(
        default=None,
        description="Model for Fix Proposer agent (defaults to gemini_model)"
    )
    
    # Token budgets per agent (to prevent output truncation)
    investigator_max_tokens: int = Field(default=4096, ge=256, description="Max output tokens for Investigator")
    diagnostician_max_tokens: int = Field(default=2048, ge=256, description="Max output tokens for Diagnostician")
    fix_proposer_max_tokens: int = Field(default=4096, ge=256, description="Max output tokens for Fix Proposer")
    
    # Timeouts
    agent_timeout_seconds: int = Field(default=120, ge=10, description="Agent execution timeout")
    tool_timeout_seconds: int = Field(default=30, ge=5, description="Individual tool call timeout")
    
    def get_investigator_model(self) -> str:
        """Get the model for Investigator agent."""
        return self.investigator_model or self.gemini_model
    
    def get_diagnostician_model(self) -> str:
        """Get the model for Diagnostician agent."""
        return self.diagnostician_model or self.gemini_model
    
    def get_fix_proposer_model(self) -> str:
        """Get the model for Fix Proposer agent."""
        return self.fix_proposer_model or self.gemini_model


class Settings(BaseSettings):
    """
    Application configuration settings with validation.
    
    Settings are loaded from environment variables with the following precedence:
    1. Environment variables
    2. config.env file
    3. .env file
    4. Default values
    """
    
    model_config = SettingsConfigDict(
        env_file=('config.env', '.env'),
        env_file_encoding='utf-8',
        extra='ignore',
        case_sensitive=False,
    )
    
    # === API Keys ===
    google_api_key: str = Field(
        default="",
        description="Google API key for Gemini"
    )
    
    # === BigQuery Configuration ===
    bigquery_project_id: str = Field(default="", description="BigQuery project ID")
    bigquery_dataset: str = Field(default="elementary", description="BigQuery dataset name")
    bigquery_credentials_path: str = Field(default="", description="Path to BigQuery service account JSON")
    
    # === dbt Project Configuration ===
    dbt_project_path: Path = Field(
        default=Path("/Users/duc.tran/airbnb-dbt-project/dbt"),
        description="Path to dbt project root"
    )
    dbt_profiles_dir: Path = Field(
        default=Path("~/.dbt"),
        description="Path to dbt profiles directory"
    )
    dbt_project_name: str = Field(default="airbnb-analytics", description="dbt project display name")
    dbt_adapter: str = Field(default="BigQuery", description="dbt adapter type")
    
    # === Knowledge Base ===
    knowledge_base_path: Path = Field(
        default=Path("/Users/duc.tran/dbt-copilot/knowledge_base"),
        description="Path to knowledge base directory"
    )
    
    # === Application Settings ===
    polling_interval_seconds: int = Field(default=30, ge=1, description="Polling interval in seconds")
    use_mock_data: bool = Field(default=True, description="Use mock data for demo")
    debug_mode: bool = Field(default=False, description="Enable debug logging")
    
    # === Nested Settings ===
    # Langfuse settings (loaded from LANGFUSE_* env vars)
    langfuse_enabled: bool = Field(default=False, alias="LANGFUSE_ENABLED")
    langfuse_public_key: Optional[str] = Field(default=None, alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: Optional[str] = Field(default=None, alias="LANGFUSE_SECRET_KEY")
    langfuse_base_url: str = Field(default="https://cloud.langfuse.com", alias="LANGFUSE_BASE_URL")
    
    # Rate limit settings
    rate_limit_enabled: bool = Field(default=True, alias="RATE_LIMIT_ENABLED")
    rate_limit_gemini_rpm: int = Field(default=60, alias="RATE_LIMIT_GEMINI_RPM")
    rate_limit_gemini_tpm: int = Field(default=1_000_000, alias="RATE_LIMIT_GEMINI_TPM")
    rate_limit_bigquery_qpm: int = Field(default=100, alias="RATE_LIMIT_BIGQUERY_QPM")
    rate_limit_embedding_rpm: int = Field(default=100, alias="RATE_LIMIT_EMBEDDING_RPM")
    
    # Cache settings
    cache_enabled: bool = Field(default=True, alias="CACHE_ENABLED")
    cache_embedding_dir: str = Field(default=".cache/embeddings", alias="CACHE_EMBEDDING_DIR")
    cache_embedding_ttl_hours: int = Field(default=168, alias="CACHE_EMBEDDING_TTL_HOURS")
    
    # Agent settings
    gemini_model: str = Field(default="gemini-2.5-pro", alias="GEMINI_MODEL")
    agent_investigator_model: Optional[str] = Field(default=None, alias="AGENT_INVESTIGATOR_MODEL")
    agent_diagnostician_model: Optional[str] = Field(default=None, alias="AGENT_DIAGNOSTICIAN_MODEL")
    agent_fix_proposer_model: Optional[str] = Field(default=None, alias="AGENT_FIX_PROPOSER_MODEL")
    agent_timeout_seconds: int = Field(default=120, alias="AGENT_TIMEOUT_SECONDS")
    
    @field_validator('dbt_profiles_dir', mode='before')
    @classmethod
    def expand_home_dir(cls, v: str | Path) -> Path:
        """Expand ~ in paths."""
        return Path(v).expanduser() if isinstance(v, (str, Path)) else v
    
    @field_validator('dbt_project_path', 'knowledge_base_path', mode='before')
    @classmethod
    def resolve_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v
    
    @property
    def langfuse(self) -> LangfuseSettings:
        """Get Langfuse settings as a nested object."""
        return LangfuseSettings(
            enabled=self.langfuse_enabled,
            public_key=self.langfuse_public_key,
            secret_key=self.langfuse_secret_key,
            host=self.langfuse_base_url,
        )
    
    @property
    def rate_limits(self) -> RateLimitSettings:
        """Get rate limit settings as a nested object."""
        return RateLimitSettings(
            enabled=self.rate_limit_enabled,
            gemini_rpm=self.rate_limit_gemini_rpm,
            gemini_tpm=self.rate_limit_gemini_tpm,
            bigquery_qpm=self.rate_limit_bigquery_qpm,
            embedding_rpm=self.rate_limit_embedding_rpm,
        )
    
    @property
    def cache(self) -> CacheSettings:
        """Get cache settings as a nested object."""
        return CacheSettings(
            enabled=self.cache_enabled,
            embedding_cache_dir=Path(self.cache_embedding_dir),
            embedding_cache_ttl_hours=self.cache_embedding_ttl_hours,
        )
    
    @property
    def agent(self) -> AgentSettings:
        """Get agent settings as a nested object."""
        return AgentSettings(
            gemini_model=self.gemini_model,
            investigator_model=self.agent_investigator_model,
            diagnostician_model=self.agent_diagnostician_model,
            fix_proposer_model=self.agent_fix_proposer_model,
            agent_timeout_seconds=self.agent_timeout_seconds,
        )
    
    @property
    def manifest_path(self) -> Path:
        """Path to dbt manifest.json."""
        return self.dbt_project_path / "target" / "manifest.json"
    
    @property
    def models_path(self) -> Path:
        """Path to dbt models directory."""
        return self.dbt_project_path / "models"
    
    @property
    def tests_path(self) -> Path:
        """Path to dbt tests directory."""
        return self.dbt_project_path / "tests"
    
    def validate_required_settings(self) -> list[str]:
        """
        Validate that required settings are configured.
        
        Returns:
            List of validation error messages (empty if all valid)
        """
        errors = []
        
        if not self.google_api_key:
            errors.append("GOOGLE_API_KEY is required")
        
        if not self.use_mock_data:
            if not self.bigquery_project_id:
                errors.append("BIGQUERY_PROJECT_ID is required when not using mock data")
            if not self.bigquery_credentials_path:
                errors.append("BIGQUERY_CREDENTIALS_PATH is required when not using mock data")
        
        if not self.dbt_project_path.exists():
            errors.append(f"DBT_PROJECT_PATH does not exist: {self.dbt_project_path}")
        
        return errors


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Settings are loaded once and cached for the application lifetime.
    Use clear_settings_cache() to reload settings.
    
    Returns:
        Validated Settings instance
    """
    # Try to load from config.env first
    env_file = Path(__file__).parent.parent.parent / "config.env"
    if env_file.exists():
        load_dotenv(env_file)
    else:
        load_dotenv()
    
    return Settings()


def clear_settings_cache() -> None:
    """Clear the settings cache to force reload on next access."""
    get_settings.cache_clear()

