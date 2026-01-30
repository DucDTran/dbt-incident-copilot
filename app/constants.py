"""
Constants and configuration values for dbt Co-Work.

This module centralizes all magic numbers, strings, and configuration
constants used throughout the application.
"""

from enum import Enum
from typing import Final


# ===========================
# Application Constants
# ===========================

APP_NAME: Final[str] = "dbt-copilot"
APP_DISPLAY_NAME: Final[str] = "dbt Co-Work"
APP_VERSION: Final[str] = "1.0.0"


# ===========================
# Agent Names
# ===========================

class AgentName(str, Enum):
    """Names for the multi-agent system."""
    INVESTIGATOR = "investigator"
    DIAGNOSTICIAN = "diagnostician"
    FIX_PROPOSER = "fix_proposer"


# ===========================
# Test Status Values
# ===========================

class TestStatus(str, Enum):
    """Possible test result statuses."""
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    ERROR = "error"


class TestSeverity(str, Enum):
    """Test severity levels."""
    ERROR = "ERROR"
    WARN = "WARN"


# ===========================
# Fix Types
# ===========================

class FixType(str, Enum):
    """Types of fixes that can be proposed."""
    SCHEMA = "schema"
    SQL = "sql"


class RiskLevel(str, Enum):
    """Risk levels for proposed fixes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


# ===========================
# Diagnosis Categories
# ===========================

class DiagnosisCategory(str, Enum):
    """Categories for test failure diagnoses."""
    SCHEMA_MISMATCH = "schema_mismatch"
    DATA_QUALITY = "data_quality"
    BUSINESS_RULE_CHANGE = "business_rule_change"
    UPSTREAM_ISSUE = "upstream_issue"
    TEST_MISCONFIGURATION = "test_misconfiguration"


# ===========================
# SQL Security
# ===========================

# SQL keywords that indicate write operations (blocked)
SQL_WRITE_KEYWORDS: Final[tuple[str, ...]] = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "CREATE",
    "ALTER",
    "TRUNCATE",
    "GRANT",
    "REVOKE",
    "EXEC",
    "EXECUTE",
    "CALL",
)

# Default SQL query limits
SQL_DEFAULT_ROW_LIMIT: Final[int] = 100
SQL_MAX_ROW_LIMIT: Final[int] = 1000
SQL_QUERY_TIMEOUT_SECONDS: Final[int] = 30


# ===========================
# Embedding Configuration
# ===========================

EMBEDDING_MODEL: Final[str] = "text-embedding-004"
EMBEDDING_MAX_TEXT_LENGTH: Final[int] = 8000
EMBEDDING_MIN_SIMILARITY_SCORE: Final[float] = 0.3


# ===========================
# UI Constants
# ===========================

# Color palette
class Colors:
    """UI color constants."""
    PRIMARY: Final[str] = "#FF683B"
    SUCCESS: Final[str] = "#22c55e"
    WARNING: Final[str] = "#f59e0b"
    ERROR: Final[str] = "#ef4444"
    INFO: Final[str] = "#3b82f6"
    BACKGROUND_DARK: Final[str] = "#0d0d1a"
    BACKGROUND_LIGHT: Final[str] = "#ffffff"


# Status indicators
STATUS_EMOJI: Final[dict[str, str]] = {
    "pass": "✅",
    "fail": "❌",
    "warn": "⚠️",
    "error": "🚨",
    "thinking": "🔄",
    "success": "✓",
}


# ===========================
# File Patterns
# ===========================

DBT_SQL_PATTERN: Final[str] = "**/*.sql"
DBT_YAML_PATTERN: Final[str] = "**/*.yml"
KNOWLEDGE_BASE_PATTERN: Final[str] = "**/*.md"


# ===========================
# Timeout Defaults (seconds)
# ===========================

class Timeouts:
    """Timeout values in seconds."""
    AGENT_DEFAULT: Final[int] = 120
    TOOL_DEFAULT: Final[int] = 30
    DBT_COMPILE: Final[int] = 60
    SQL_QUERY: Final[int] = 30
    RATE_LIMIT_WAIT: Final[int] = 30


# ===========================
# Rate Limit Defaults
# ===========================

class RateLimitDefaults:
    """Default rate limit values."""
    GEMINI_RPM: Final[int] = 60
    GEMINI_TPM: Final[int] = 1_000_000
    BIGQUERY_QPM: Final[int] = 100
    EMBEDDING_RPM: Final[int] = 100


# ===========================
# Cache Defaults
# ===========================

class CacheDefaults:
    """Default cache configuration values."""
    EMBEDDING_TTL_HOURS: Final[int] = 168  # 1 week
    MEMORY_CACHE_MAX_ENTRIES: Final[int] = 1000
    DISK_CACHE_MAX_ENTRIES: Final[int] = 10000
