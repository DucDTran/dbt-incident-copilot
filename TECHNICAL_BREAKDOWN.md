# 🔬 dbt Co-Work Technical Breakdown

> A comprehensive technical documentation of every file in the dbt-copilot codebase

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [Configuration Files](#configuration-files)
3. [Core Application](#core-application)
4. [Configuration Module](#configuration-module)
5. [Agent Module](#agent-module)
6. [Agent Tools](#agent-tools)
7. [Prompts Module](#prompts-module)
8. [Database Module](#database-module)
9. [UI Module](#ui-module)
10. [System Architecture](#system-architecture)
11. [Data Flow](#data-flow)

---

## Overview

**dbt Co-Work** is an Agentic AI Platform for Analytics Engineering Incident Resolution. It uses Google ADK (Agent Development Kit) with Gemini models to autonomously investigate dbt test failures, diagnose root causes, and propose fixes.

### Tech Stack

| Component | Technology |
|-----------|------------|
| **AI Engine** | Google ADK + Gemini 2.0/2.5 |
| **Web UI** | Streamlit |
| **Database** | BigQuery (with mock mode) |
| **Configuration** | Pydantic Settings |
| **Observability** | Langfuse (optional) |

---

## Configuration Files

### `config.example.env`

**Purpose**: Template configuration file with all environment variables needed to run the application.

**Key Configuration Sections**:

```dotenv
# ============================================================
# Gemini API Configuration
# ============================================================
GOOGLE_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-pro

# Agent-specific models (optional - falls back to GEMINI_MODEL)
AGENT_INVESTIGATOR_MODEL=gemini-2.5-flash
AGENT_DIAGNOSTICIAN_MODEL=gemini-2.5-pro
AGENT_FIX_PROPOSER_MODEL=gemini-2.5-pro

# ============================================================
# BigQuery Configuration
# ============================================================
BIGQUERY_PROJECT_ID=your_project_id
BIGQUERY_CREDENTIALS_PATH=./credentials/bigquery-sa.json
BIGQUERY_DATASET=elementary

# ============================================================
# dbt Project Configuration
# ============================================================
DBT_PROJECT_PATH=/path/to/your/dbt/project
KNOWLEDGE_BASE_PATH=./knowledge_base

# ============================================================
# Application Settings
# ============================================================
USE_MOCK_DATA=true  # Set to false for real BigQuery

# ============================================================
# Langfuse Observability (Optional)
# ============================================================
LANGFUSE_ENABLED=false
LANGFUSE_PUBLIC_KEY=pk-lf-xxx
LANGFUSE_SECRET_KEY=sk-lf-xxx
```

**Usage**: Copy to `config.env` and fill in your values.

---

### `requirements.txt`

**Purpose**: Python package dependencies.

**Key Dependencies**:

| Package | Version | Purpose |
|---------|---------|---------|
| `google-adk>=0.3.0` | Latest | Google Agent Development Kit - multi-agent orchestration |
| `google-genai>=1.0.0` | Latest | Gemini API client for LLM calls |
| `streamlit>=1.40.0` | Latest | Web UI framework |
| `google-cloud-bigquery>=3.25.0` | Latest | BigQuery integration for Elementary |
| `dbt-core>=1.8.0` | Latest | dbt CLI integration |
| `langfuse>=2.0.0` | Latest | Observability and tracing |
| `pydantic-settings>=2.0.0` | Latest | Configuration validation |
| `PyYAML>=6.0` | Latest | YAML parsing for schema files |
| `pandas>=2.0.0` | Latest | Data manipulation |

---

### `run.sh`

**Purpose**: One-line setup and launch script.

**Implementation**:

```bash
#!/bin/bash
set -e

# Navigate to project directory
cd "$(dirname "$0")"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/upgrade dependencies
pip install -r requirements.txt --quiet

# Export environment variables from config.env
if [ -f "config.env" ]; then
    export $(grep -v '^#' config.env | xargs)
fi

# Launch Streamlit application
streamlit run app/main.py --server.port 8501
```

**Usage**: `./run.sh` from project root.

---

## Core Application

### `app/main.py`

**Purpose**: Main Streamlit application entry point. Handles page navigation and application lifecycle.

**Key Components**:

```python
import streamlit as st
from app.ui.mission_control import render_mission_control
from app.ui.resolution_studio import render_resolution_studio
from app.ui.snoozed import render_snoozed_page
from app.config import get_settings

def main():
    """Main application entry point."""
    st.set_page_config(
        page_title="dbt Co-Work",
        page_icon="🚀",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    # Define pages using Streamlit's native navigation
    pages = [
        st.Page(dashboard_page, title="Dashboard", icon=":material/dashboard:"),
        st.Page(labs_page, title="Labs", icon=":material/science:"),
        st.Page(snoozed_page, title="Snoozed", icon=":material/snooze:"),
    ]
    
    # Render navigation
    selected_page = st.navigation(pages)
    selected_page.run()

def dashboard_page():
    """Render the Dashboard (Mission Control) page."""
    render_mission_control()

def labs_page():
    """Render the Labs (Resolution Studio) page."""
    render_resolution_studio()

def snoozed_page():
    """Render the Snoozed incidents page."""
    render_snoozed_page()

if __name__ == "__main__":
    main()
```

**Connections**:
- Imports UI renderers from `app/ui/`
- Uses `get_settings()` from `app/config/`
- Session state management for navigation

---

## Configuration Module

### `app/config/__init__.py`

**Purpose**: Module initialization and exports.

```python
from app.config.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
```

---

### `app/config/settings.py`

**Purpose**: Pydantic-based configuration management with validation and environment variable loading.

**Key Classes**:

```python
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from functools import lru_cache

class LangfuseSettings(BaseModel):
    """Langfuse observability configuration."""
    enabled: bool = Field(default=False, description="Enable Langfuse tracing")
    public_key: Optional[str] = Field(default=None)
    secret_key: Optional[str] = Field(default=None)
    host: str = Field(default="https://cloud.langfuse.com")

class RateLimitSettings(BaseModel):
    """Rate limiting configuration for API calls."""
    gemini_rpm: int = Field(default=60, description="Gemini requests per minute")
    gemini_tpm: int = Field(default=1_000_000, description="Gemini tokens per minute")
    bigquery_qpm: int = Field(default=100, description="BigQuery queries per minute")

class AgentSettings(BaseModel):
    """Multi-agent architecture configuration."""
    gemini_model: str = Field(default="gemini-2.5-pro")
    investigator_model: Optional[str] = Field(default=None)
    diagnostician_model: Optional[str] = Field(default=None)
    fix_proposer_model: Optional[str] = Field(default=None)
    
    def get_investigator_model(self) -> str:
        """Get model for Investigator agent (falls back to default)."""
        return self.investigator_model or self.gemini_model
    
    def get_diagnostician_model(self) -> str:
        """Get model for Diagnostician agent (falls back to default)."""
        return self.diagnostician_model or self.gemini_model
    
    def get_fix_proposer_model(self) -> str:
        """Get model for Fix Proposer agent (falls back to default)."""
        return self.fix_proposer_model or self.gemini_model

class Settings(BaseSettings):
    """Main application configuration loaded from environment."""
    
    model_config = SettingsConfigDict(
        env_file=('config.env', '.env'),
        env_file_encoding='utf-8',
        extra='ignore',
        env_nested_delimiter='__',
    )
    
    # Required settings
    google_api_key: str = Field(..., description="Gemini API key")
    dbt_project_path: Path = Field(..., description="Path to dbt project")
    
    # BigQuery settings
    bigquery_project_id: str = Field(default="")
    bigquery_credentials_path: Optional[Path] = Field(default=None)
    bigquery_dataset: str = Field(default="elementary")
    
    # Application settings
    use_mock_data: bool = Field(default=True)
    knowledge_base_path: Path = Field(default=Path("./knowledge_base"))
    
    # Nested settings
    langfuse: LangfuseSettings = Field(default_factory=LangfuseSettings)
    rate_limits: RateLimitSettings = Field(default_factory=RateLimitSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance (singleton pattern)."""
    return Settings()
```

**Key Features**:
- Environment variable loading from `config.env`
- Type validation via Pydantic
- Nested configuration objects
- Cached singleton pattern via `@lru_cache()`

---

## Agent Module

### `app/agent/__init__.py`

**Purpose**: Module exports for agent implementations.

```python
"""
Agent module for dbt Co-Work.

Contains the ADK agent implementations:
- CopilotAgent: Single-agent implementation (legacy)
- MultiAgentCopilot: Multi-agent implementation (recommended)
"""

from app.agent.copilot_agent import CopilotAgent, run_investigation
from app.agent.multi_agent_copilot import (
    MultiAgentCopilot,
    create_multi_agent_copilot,
    run_multi_agent_investigation,
    Investigation,
    InvestigationStep,
    InvestigationContext,
    Diagnosis,
)

__all__ = [
    # Single-agent (legacy)
    "CopilotAgent",
    "run_investigation",
    # Multi-agent (recommended)
    "MultiAgentCopilot",
    "create_multi_agent_copilot",
    "run_multi_agent_investigation",
    # Data classes
    "Investigation",
    "InvestigationStep",
    "InvestigationContext",
    "Diagnosis",
]
```

---

### `app/agent/copilot_agent.py`

**Purpose**: Legacy single-agent implementation using Google ADK. Kept as fallback but superseded by multi-agent.

**Key Data Classes**:

```python
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional

@dataclass
class InvestigationStep:
    """A single step in the agent's investigation process."""
    timestamp: datetime
    action: str                    # 'tool_call', 'thinking', 'diagnosis'
    tool_name: Optional[str]       # Name of tool called (if applicable)
    input_summary: str             # Summary of input/reasoning
    output_summary: str            # Summary of output/result
    status: str                    # 'success', 'error', 'thinking'
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class Investigation:
    """Complete investigation result for a test failure."""
    test_result: Dict[str, Any]              # Original test failure data
    steps: List[InvestigationStep]           # All investigation steps
    diagnosis: Optional[str] = None          # Final diagnosis text
    fix_options: List[Dict[str, Any]] = field(default_factory=list)
    raw_context: Dict[str, Any] = field(default_factory=dict)
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
```

**ADK Tool Functions** (wrapped for agent use):

```python
def adk_get_model_lineage(model_name: str) -> str:
    """Get upstream and downstream dependencies for a dbt model.
    
    Args:
        model_name: Name of the dbt model (e.g., 'dim_listing')
    
    Returns:
        JSON string with upstream, downstream models and associated tests
    """
    result = get_model_lineage(model_name)
    return json.dumps(result, indent=2)

def adk_read_model_sql(model_name: str) -> str:
    """Read the SQL transformation code for a dbt model.
    
    Args:
        model_name: Name of the dbt model
    
    Returns:
        JSON string with file path and SQL content
    """
    file_result = find_file_by_model_name(model_name)
    if file_result["status"] == "success":
        read_result = tool_read_repo(file_result["data"]["path"])
        return json.dumps(read_result, indent=2)
    return json.dumps(file_result)

def adk_read_schema_definition(model_name: str) -> str:
    """Read schema.yml definition including columns and tests.
    
    Args:
        model_name: Name of the dbt model
    
    Returns:
        JSON string with schema file content and parsed model definition
    """

def adk_search_knowledge_base(query: str, context_column: str = "", 
                              context_model: str = "") -> str:
    """Search business rules knowledge base for relevant documentation.
    
    Uses semantic search with Gemini embeddings to find relevant
    business rules, policies, and playbooks.
    
    Args:
        query: Natural language search query
        context_column: Optional column name for context
        context_model: Optional model name for context
    
    Returns:
        JSON string with matching documents and relevance scores
    """
```

**Main Agent Class**:

```python
from google.adk import LlmAgent, Runner
from google.adk.sessions import InMemorySessionService

class CopilotAgent:
    """Single-agent implementation for test failure investigation."""
    
    def __init__(self):
        self.settings = get_settings()
        self._agent = self._create_agent()
        self._runner = Runner(
            agent=self._agent,
            app_name="dbt-copilot",
            session_service=InMemorySessionService(),
        )
    
    def _create_agent(self) -> LlmAgent:
        """Create the ADK agent with tools."""
        return LlmAgent(
            name="dbt_copilot",
            model=self.settings.agent.gemini_model,
            instruction=COPILOT_SYSTEM_INSTRUCTION,
            tools=[
                adk_get_model_lineage,
                adk_read_model_sql,
                adk_read_schema_definition,
                adk_search_knowledge_base,
                adk_execute_sql,
                adk_propose_fix,
            ],
        )
    
    async def investigate(self, test_result: Dict[str, Any], 
                         stream_steps: bool = True) -> AsyncGenerator:
        """Run investigation on a test failure.
        
        Args:
            test_result: Test failure data from Elementary
            stream_steps: Whether to yield steps as they happen
        
        Yields:
            InvestigationStep objects as investigation progresses
        
        Returns:
            Final Investigation object
        """
        session = await self._runner.session_service.create_session(
            app_name="dbt-copilot",
            user_id="default_user",
        )
        
        prompt = get_investigation_prompt(
            test_name=test_result.get("test_name"),
            model_name=test_result.get("model_name"),
            column_name=test_result.get("column_name"),
            error_message=test_result.get("error_message"),
            failed_rows=test_result.get("failed_rows", 0),
            test_id=test_result.get("test_id"),
        )
        
        async for event in self._runner.run_async(
            session_id=session.id,
            user_id="default_user",
            new_message=prompt,
        ):
            step = self._process_event(event)
            if step and stream_steps:
                yield step
```

---

### `app/agent/multi_agent_copilot.py`

**Purpose**: Multi-agent architecture that splits investigation into three specialized agents to prevent output token truncation and improve quality.

**Architecture Diagram**:

```
┌─────────────────┐    ┌──────────────────┐    ┌────────────────────┐
│  INVESTIGATOR   │───▶│  DIAGNOSTICIAN   │───▶│   FIX PROPOSER     │
│     Agent       │    │     Agent        │    │      Agent         │
│  (6 tools)      │    │  (no tools)      │    │   (1 tool)         │
└─────────────────┘    └──────────────────┘    └────────────────────┘
        │                      │                        │
        ▼                      ▼                        ▼
  Gathers context       Analyzes findings        Generates fixes
  using tools           identifies root cause    with rationale
```

**Key Data Classes**:

```python
@dataclass
class InvestigationContext:
    """Context gathered by the Investigator Agent."""
    test_details: Optional[str] = None       # Detailed test failure info
    schema_definition: Optional[str] = None  # YAML schema content
    sql_code: Optional[str] = None           # Model SQL content
    lineage: Optional[str] = None            # Upstream/downstream models
    business_rules: Optional[str] = None     # Relevant KB matches
    data_samples: Optional[str] = None       # Sample data from SQL queries

@dataclass
class Diagnosis:
    """Structured diagnosis from the Diagnostician Agent."""
    root_cause: str           # Primary cause of failure
    evidence: List[str]       # Supporting evidence from investigation
    impact_assessment: str    # Business/data impact
    category: str             # 'schema_mismatch', 'data_quality', etc.
    severity: str             # 'critical', 'high', 'medium', 'low'
    confidence: str           # 'high', 'medium', 'low'
```

**Main Class**:

```python
class MultiAgentCopilot:
    """Multi-agent implementation for robust investigation."""
    
    def __init__(self):
        self.settings = get_settings()
        self._investigator = self._create_investigator_agent()
        self._diagnostician = self._create_diagnostician_agent()
        self._fix_proposer = self._create_fix_proposer_agent()
        
        # Separate runners for each agent
        self._investigator_runner = Runner(
            agent=self._investigator,
            app_name="dbt-copilot-investigator",
            session_service=InMemorySessionService(),
        )
        # ... similar for other agents
    
    def _create_investigator_agent(self) -> LlmAgent:
        """Create Investigator Agent with context-gathering tools."""
        return LlmAgent(
            name="investigator",
            model=self.settings.agent.get_investigator_model(),
            instruction=INVESTIGATOR_SYSTEM_INSTRUCTION,
            tools=[
                adk_get_model_lineage,
                adk_read_model_sql,
                adk_read_schema_definition,
                adk_search_knowledge_base,
                adk_get_test_details,
                adk_execute_sql,
            ],
        )
    
    def _create_diagnostician_agent(self) -> LlmAgent:
        """Create Diagnostician Agent (no tools - pure reasoning)."""
        return LlmAgent(
            name="diagnostician",
            model=self.settings.agent.get_diagnostician_model(),
            instruction=DIAGNOSTICIAN_SYSTEM_INSTRUCTION,
            tools=[],  # No tools - analyzes context only
        )
    
    def _create_fix_proposer_agent(self) -> LlmAgent:
        """Create Fix Proposer Agent with fix generation tool."""
        return LlmAgent(
            name="fix_proposer",
            model=self.settings.agent.get_fix_proposer_model(),
            instruction=FIX_PROPOSER_SYSTEM_INSTRUCTION,
            tools=[adk_propose_fix],  # Only one tool
        )
    
    async def investigate(self, test_result: Dict[str, Any]) -> AsyncGenerator:
        """Run multi-stage investigation pipeline.
        
        Pipeline:
        1. Investigator gathers context using tools
        2. Diagnostician analyzes context and produces diagnosis
        3. Fix Proposer generates fix options using diagnosis
        
        Yields:
            InvestigationStep objects as each stage progresses
        """
        # Stage 1: Investigation
        async for step in self._run_investigator(test_result):
            yield step
        
        # Stage 2: Diagnosis
        async for step in self._run_diagnostician(context):
            yield step
        
        # Stage 3: Fix Proposal
        async for step in self._run_fix_proposer(diagnosis, context):
            yield step
    
    async def _run_fix_proposer(self, diagnosis: Diagnosis, 
                                context: InvestigationContext) -> AsyncGenerator:
        """Run Fix Proposer agent to generate fix options.
        
        Important: Handles ADK response wrapper unwrapping.
        ADK wraps tool responses in {"result": "..."} format.
        """
        async for event in self._fix_proposer_runner.run_async(...):
            if hasattr(event, 'function_responses'):
                for response in event.function_responses:
                    fix_data = json.loads(response.response)
                    
                    # CRITICAL: Unwrap ADK result wrapper
                    if "result" in fix_data and len(fix_data) == 1:
                        inner = fix_data["result"]
                        if isinstance(inner, str):
                            fix_data = json.loads(inner)
                        elif isinstance(inner, dict):
                            fix_data = inner
                    
                    # Process fix options...
```

---

## Agent Tools

### `app/agent/tools/__init__.py`

**Purpose**: Tool exports and public API.

```python
"""Tool implementations for the dbt Co-Work agent."""

from .manifest_tool import tool_read_manifest, get_model_lineage
from .repo_tool import tool_read_repo, tool_write_repo
from .elementary_tool import tool_query_elementary, get_failed_tests
from .knowledge_base_tool import tool_consult_knowledge_base
from .fix_tool import apply_fix, run_dbt_test
from .dbt_tool import run_dbt_compile, run_dry_run, check_dbt_installed
from .sql_tool import tool_execute_sql, adk_execute_sql
from .agentic_fix_tool import (
    adk_generate_schema_fix,
    adk_generate_sql_fix,
    adk_apply_fix,
    adk_undo_fix,
    adk_propose_fix,
    load_fixed_incidents,
    save_fixed_incident,
    remove_fixed_incident,
    is_incident_fixed,
)

__all__ = [
    "tool_read_manifest",
    "get_model_lineage",
    "tool_read_repo",
    "tool_write_repo",
    "tool_query_elementary",
    "get_failed_tests",
    "tool_consult_knowledge_base",
    "apply_fix",
    "run_dbt_test",
    "run_dbt_compile",
    "run_dry_run",
    "check_dbt_installed",
    "tool_execute_sql",
    "adk_execute_sql",
    "adk_generate_schema_fix",
    "adk_generate_sql_fix",
    "adk_apply_fix",
    "adk_undo_fix",
    "adk_propose_fix",
    "load_fixed_incidents",
    "save_fixed_incident",
    "remove_fixed_incident",
    "is_incident_fixed",
]
```

---

### `app/agent/tools/manifest_tool.py`

**Purpose**: Parse dbt `manifest.json` for model metadata and lineage analysis.

**Key Functions**:

```python
import json
from pathlib import Path
from app.config import get_settings

def tool_read_manifest() -> Dict[str, Any]:
    """Parse dbt manifest.json and extract models, tests, sources.
    
    Reads from: {dbt_project_path}/target/manifest.json
    
    Returns:
        Dict with:
        - models: List of model definitions
        - tests: List of test definitions  
        - sources: List of source definitions
        - metadata: Manifest metadata (dbt version, generated_at)
    """
    settings = get_settings()
    manifest_path = settings.dbt_project_path / "target" / "manifest.json"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    models = []
    tests = []
    sources = []
    
    for unique_id, node in manifest.get("nodes", {}).items():
        if node.get("resource_type") == "model":
            models.append({
                "unique_id": unique_id,
                "name": node.get("name"),
                "schema": node.get("schema"),
                "database": node.get("database"),
                "depends_on": node.get("depends_on", {}).get("nodes", []),
                "columns": node.get("columns", {}),
            })
        elif node.get("resource_type") == "test":
            tests.append({
                "unique_id": unique_id,
                "name": node.get("name"),
                "test_metadata": node.get("test_metadata", {}),
                "depends_on": node.get("depends_on", {}).get("nodes", []),
            })
    
    return {
        "status": "success",
        "data": {
            "models": models,
            "tests": tests,
            "sources": sources,
            "metadata": manifest.get("metadata", {}),
        }
    }

def get_model_lineage(model_name: str) -> Dict[str, Any]:
    """Get upstream/downstream dependencies for a specific model.
    
    Args:
        model_name: Name of the dbt model (e.g., 'dim_listing')
    
    Returns:
        Dict with:
        - upstream: List of models this model depends on
        - downstream: List of models that depend on this model
        - tests: List of tests associated with this model
    """
    manifest = tool_read_manifest()
    if manifest["status"] != "success":
        return manifest
    
    models = manifest["data"]["models"]
    tests = manifest["data"]["tests"]
    
    # Find target model
    target_model = None
    for model in models:
        if model["name"] == model_name:
            target_model = model
            break
    
    if not target_model:
        return {"status": "error", "message": f"Model '{model_name}' not found"}
    
    # Get upstream (what this model depends on)
    upstream = []
    for dep in target_model.get("depends_on", []):
        if "model." in dep:
            upstream.append(dep.split(".")[-1])
    
    # Get downstream (what depends on this model)
    downstream = []
    for model in models:
        if target_model["unique_id"] in model.get("depends_on", []):
            downstream.append(model["name"])
    
    # Get associated tests
    associated_tests = []
    for test in tests:
        if target_model["unique_id"] in test.get("depends_on", []):
            associated_tests.append(test["name"])
    
    return {
        "status": "success",
        "data": {
            "model": model_name,
            "upstream": upstream,
            "downstream": downstream,
            "tests": associated_tests,
        }
    }
```

---

### `app/agent/tools/repo_tool.py`

**Purpose**: File system operations for reading/writing dbt project files.

**Key Functions**:

```python
import shutil
from pathlib import Path
from datetime import datetime
from app.config import get_settings

def tool_read_repo(file_path: str) -> Dict[str, Any]:
    """Read file from dbt project repository.
    
    Args:
        file_path: Absolute or relative path to file
    
    Returns:
        Dict with path, content, and line count
    """
    settings = get_settings()
    
    # Handle relative paths
    if not file_path.startswith("/"):
        full_path = settings.dbt_project_path / file_path
    else:
        full_path = Path(file_path)
    
    # Security: Ensure path is within project
    if not str(full_path).startswith(str(settings.dbt_project_path)):
        return {"status": "error", "message": "Path outside project directory"}
    
    if not full_path.exists():
        return {"status": "error", "message": f"File not found: {file_path}"}
    
    content = full_path.read_text()
    return {
        "status": "success",
        "data": {
            "path": str(full_path),
            "relative_path": str(full_path.relative_to(settings.dbt_project_path)),
            "content": content,
            "lines": len(content.splitlines()),
        }
    }

def tool_write_repo(file_path: str, content: str, 
                    create_backup: bool = True) -> Dict[str, Any]:
    """Write file with automatic backup.
    
    Args:
        file_path: Path to file to write
        content: New file content
        create_backup: Whether to create .bak file
    
    Returns:
        Dict with status and backup path
    """
    full_path = Path(file_path)
    backup_path = None
    
    if create_backup and full_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{file_path}.bak.{timestamp}"
        shutil.copy2(full_path, backup_path)
    
    full_path.write_text(content)
    
    return {
        "status": "success",
        "data": {
            "path": str(full_path),
            "backup_path": backup_path,
        }
    }

def find_file_by_model_name(model_name: str) -> Dict[str, Any]:
    """Find SQL file for a model name.
    
    Searches in {dbt_project_path}/models/ for {model_name}.sql
    """
    settings = get_settings()
    models_path = settings.dbt_project_path / "models"
    
    for sql_file in models_path.rglob(f"{model_name}.sql"):
        return {
            "status": "success",
            "data": {
                "path": str(sql_file),
                "relative_path": str(sql_file.relative_to(settings.dbt_project_path)),
            }
        }
    
    return {"status": "error", "message": f"SQL file not found for model: {model_name}"}

def find_schema_file(model_name: str) -> Dict[str, Any]:
    """Find schema.yml containing model definition.
    
    Searches for .yml files containing the model name.
    """
    settings = get_settings()
    models_path = settings.dbt_project_path / "models"
    
    for yml_file in models_path.rglob("*.yml"):
        content = yml_file.read_text()
        if f"name: {model_name}" in content:
            return {
                "status": "success",
                "data": {
                    "path": str(yml_file),
                    "relative_path": str(yml_file.relative_to(settings.dbt_project_path)),
                }
            }
    
    return {"status": "error", "message": f"Schema file not found for model: {model_name}"}

def get_file_diff(original: str, modified: str) -> Dict[str, Any]:
    """Generate unified diff between two file versions.
    
    Returns:
        Dict with diff text, additions count, deletions count
    """
    import difflib
    
    diff = list(difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="original",
        tofile="modified",
    ))
    
    additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
    
    return {
        "status": "success",
        "data": {
            "diff": "".join(diff),
            "additions": additions,
            "deletions": deletions,
        }
    }
```

---

### `app/agent/tools/elementary_tool.py`

**Purpose**: Query Elementary test results from BigQuery or mock data.

**Key Functions**:

```python
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from app.config import get_settings
from app.db.mock_elementary import MockElementaryDB

@dataclass
class TestResult:
    """Represents a single test result."""
    test_id: str
    test_name: str
    model_name: str
    column_name: Optional[str]
    status: str                    # 'pass', 'fail', 'warn', 'error'
    error_message: Optional[str]
    failed_rows: int
    failed_row_samples: List[Dict[str, Any]]
    test_type: str                 # 'accepted_values', 'not_null', etc.
    severity: str                  # 'ERROR', 'WARN'
    executed_at: datetime

def tool_query_elementary(status: str = None, model_name: str = None, 
                         limit: int = 50) -> Dict[str, Any]:
    """Query test results from Elementary.
    
    Uses mock data when USE_MOCK_DATA=true, otherwise queries BigQuery.
    
    Args:
        status: Filter by status ('pass', 'fail', 'warn', 'error')
        model_name: Filter by model name
        limit: Maximum results to return
    
    Returns:
        Dict with list of test results
    """
    settings = get_settings()
    
    if settings.use_mock_data:
        return _query_mock_results(status, model_name, limit)
    else:
        return _query_bigquery_results(status, model_name, limit)

def _query_mock_results(status, model_name, limit) -> Dict[str, Any]:
    """Query mock Elementary database."""
    mock_db = MockElementaryDB()
    
    if status == "fail":
        results = mock_db.get_failed_results()
    else:
        results = mock_db.get_all_results()
    
    if model_name:
        results = [r for r in results if r.get("model_name") == model_name]
    
    return {
        "status": "success",
        "data": {
            "results": results[:limit],
            "total": len(results),
        }
    }

def _query_bigquery_results(status, model_name, limit) -> Dict[str, Any]:
    """Query real Elementary results from BigQuery."""
    from google.cloud import bigquery
    
    settings = get_settings()
    client = bigquery.Client(
        project=settings.bigquery_project_id,
        credentials_path=settings.bigquery_credentials_path,
    )
    
    query = f"""
    SELECT 
        test_unique_id as test_id,
        test_name,
        model_unique_id as model_name,
        column_name,
        status,
        error_message,
        failed_rows,
        test_type,
        severity,
        detected_at as executed_at
    FROM `{settings.bigquery_project_id}.{settings.bigquery_dataset}.test_results`
    WHERE 1=1
    """
    
    if status:
        query += f" AND status = '{status}'"
    if model_name:
        query += f" AND model_unique_id LIKE '%{model_name}%'"
    
    query += f" ORDER BY detected_at DESC LIMIT {limit}"
    
    results = client.query(query).result()
    # ... convert to list of dicts

def get_failed_tests() -> Dict[str, Any]:
    """Convenience function to get all failed tests."""
    return tool_query_elementary(status="fail")

def get_failed_rows_for_test(test_id: str) -> Dict[str, Any]:
    """Get sample of failed rows for a specific test."""
    settings = get_settings()
    
    if settings.use_mock_data:
        mock_db = MockElementaryDB()
        result = mock_db.get_result_by_id(test_id)
        if result:
            return {
                "status": "success",
                "data": result.get("failed_row_samples", [])
            }
    # ... BigQuery implementation
```

---

### `app/agent/tools/knowledge_base_tool.py`

**Purpose**: Semantic search over business documentation using Gemini embeddings.

**Key Classes and Functions**:

```python
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from google import genai
from google.genai import types
from app.config import get_settings

class GeminiKnowledgeBase:
    """Semantic search over markdown documentation."""
    
    EMBEDDING_MODEL = "text-embedding-004"
    CHUNK_SIZE = 1000          # Characters per chunk
    CHUNK_OVERLAP = 200        # Overlap between chunks
    
    def __init__(self, kb_path: Path):
        self.kb_path = kb_path
        self.settings = get_settings()
        self.client = genai.Client(api_key=self.settings.google_api_key)
        self.documents: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self._load_documents()
    
    def _load_documents(self):
        """Load and chunk all markdown files."""
        for md_file in self.kb_path.glob("*.md"):
            content = md_file.read_text()
            chunks = self._chunk_text(content)
            
            for i, chunk in enumerate(chunks):
                self.documents.append({
                    "file": md_file.name,
                    "chunk_index": i,
                    "content": chunk,
                    "title": self._extract_title(content),
                })
        
        # Generate embeddings for all chunks
        self._generate_embeddings()
    
    def _chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks."""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.CHUNK_SIZE
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - self.CHUNK_OVERLAP
        return chunks
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding vector for text."""
        response = self.client.models.embed_content(
            model=self.EMBEDDING_MODEL,
            contents=text,
            config=types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")
        )
        return response.embeddings[0].values
    
    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        a = np.array(a)
        b = np.array(b)
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Semantic search over knowledge base.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
        
        Returns:
            List of matching documents with relevance scores
        """
        query_embedding = self._get_embedding(query)
        
        # Calculate similarity to all documents
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self._cosine_similarity(query_embedding, doc_embedding)
            similarities.append((sim, i))
        
        # Sort by similarity and return top_k
        similarities.sort(reverse=True)
        
        results = []
        for sim, idx in similarities[:top_k]:
            doc = self.documents[idx].copy()
            doc["relevance_score"] = float(sim)
            results.append(doc)
        
        return results

def tool_consult_knowledge_base(query: str) -> Dict[str, Any]:
    """High-level knowledge base query tool.
    
    Args:
        query: Natural language query
    
    Returns:
        Dict with matching documents
    """
    settings = get_settings()
    kb = GeminiKnowledgeBase(settings.knowledge_base_path)
    
    results = kb.search(query, top_k=5)
    
    return {
        "status": "success",
        "data": {
            "query": query,
            "results": results,
            "total": len(results),
        }
    }

def search_for_business_rule(column: str, model: str, test_type: str) -> Dict[str, Any]:
    """Search for specific business rule related to a test failure.
    
    Constructs a targeted query based on column, model, and test type.
    """
    query = f"business rule policy for {column} column in {model} model {test_type} test"
    return tool_consult_knowledge_base(query)
```

---

### `app/agent/tools/sql_tool.py`

**Purpose**: Execute read-only SQL queries against BigQuery for investigation.

**Security Implementation**:

```python
import re
from google.cloud import bigquery
from app.config import get_settings

# Write operations that should be blocked
WRITE_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", 
    "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL",
    "MERGE", "REPLACE", "UPSERT"
]

def _is_read_only_query(sql: str) -> bool:
    """Validate that a SQL query is read-only.
    
    Checks:
    1. Query starts with SELECT
    2. No write keywords present
    """
    sql_upper = sql.upper().strip()
    
    # Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        return False
    
    # Check for write keywords
    for keyword in WRITE_KEYWORDS:
        pattern = r'\b' + keyword + r'\b'
        if re.search(pattern, sql_upper):
            return False
    
    return True

def _ensure_limit(sql: str, limit: int = 100) -> str:
    """Ensure query has a LIMIT clause.
    
    If no LIMIT exists, appends one. Does not modify existing LIMIT.
    """
    if "LIMIT" not in sql.upper():
        sql = sql.rstrip().rstrip(';')
        sql = f"{sql} LIMIT {limit}"
    return sql

def tool_execute_sql(sql: str, limit: int = 100) -> Dict[str, Any]:
    """Execute SQL query against data warehouse.
    
    Security measures:
    - Only SELECT queries allowed
    - Automatic LIMIT enforcement
    - Query timeout (30 seconds)
    - Max bytes billed (10MB)
    
    Args:
        sql: SQL query to execute
        limit: Maximum rows to return
    
    Returns:
        Dict with query results as list of dicts
    """
    settings = get_settings()
    
    # Validate read-only
    if not _is_read_only_query(sql):
        return {
            "status": "error",
            "message": "Only read-only SELECT queries are allowed"
        }
    
    # Ensure LIMIT
    sql = _ensure_limit(sql, min(limit, 1000))  # Max 1000 rows
    
    try:
        client = bigquery.Client(project=settings.bigquery_project_id)
        
        job_config = bigquery.QueryJobConfig(
            maximum_bytes_billed=10 * 1024 * 1024,  # 10 MB
        )
        
        query_job = client.query(sql, job_config=job_config, timeout=30)
        results = query_job.result(timeout=30)
        
        rows = [dict(row) for row in results]
        
        return {
            "status": "success",
            "data": {
                "rows": rows,
                "row_count": len(rows),
                "query": sql,
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

def adk_execute_sql(sql: str, limit: int = 100) -> str:
    """ADK-wrapped SQL execution for agent use.
    
    Returns JSON string for ADK tool response.
    """
    result = tool_execute_sql(sql, limit)
    return json.dumps(result, indent=2, default=str)
```

---

### `app/agent/tools/dbt_tool.py`

**Purpose**: Run dbt CLI commands for validation and testing.

```python
import subprocess
import tempfile
import shutil
from pathlib import Path
from app.config import get_settings
from .repo_tool import tool_read_repo, tool_write_repo

def check_dbt_installed() -> bool:
    """Check if dbt CLI is available in PATH."""
    try:
        result = subprocess.run(
            ["dbt", "--version"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False

def run_dbt_compile(model_name: str, modified_content: str = None) -> Dict[str, Any]:
    """Run dbt compile on a model.
    
    If modified_content is provided, temporarily applies it before compiling.
    
    Args:
        model_name: Name of model to compile
        modified_content: Optional modified SQL to test
    
    Returns:
        Dict with compilation result and any errors
    """
    settings = get_settings()
    original_content = None
    model_path = None
    
    try:
        # Optionally apply modified content temporarily
        if modified_content:
            from .repo_tool import find_file_by_model_name
            file_result = find_file_by_model_name(model_name)
            if file_result["status"] == "success":
                model_path = file_result["data"]["path"]
                read_result = tool_read_repo(model_path)
                original_content = read_result["data"]["content"]
                Path(model_path).write_text(modified_content)
        
        # Run dbt compile
        cmd = ["dbt", "compile", "--select", model_name]
        result = subprocess.run(
            cmd,
            cwd=settings.dbt_project_path,
            capture_output=True,
            text=True,
            timeout=60
        )
        
        return {
            "status": "success" if result.returncode == 0 else "error",
            "data": {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        }
    finally:
        # Restore original content
        if original_content and model_path:
            Path(model_path).write_text(original_content)

def run_dry_run(fix_option: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Complete dry run simulation for a fix option.
    
    Steps:
    1. Temporarily apply code changes
    2. Run dbt compile to verify syntax
    3. Check for downstream impact
    4. Restore original files
    
    Args:
        fix_option: Fix option with code_changes dict
        model_name: Name of model being fixed
    
    Returns:
        Dict with dry run results and any errors
    """
    settings = get_settings()
    backups = {}
    
    try:
        # Apply changes temporarily
        for file_path, new_content in fix_option.get("code_changes", {}).items():
            if Path(file_path).exists():
                backups[file_path] = Path(file_path).read_text()
            Path(file_path).write_text(new_content)
        
        # Run dbt compile
        compile_result = run_dbt_compile(model_name)
        
        # Check downstream models
        from .manifest_tool import get_model_lineage
        lineage = get_model_lineage(model_name)
        downstream_models = lineage.get("data", {}).get("downstream", [])
        
        downstream_results = []
        for downstream in downstream_models[:5]:  # Limit to 5
            ds_result = run_dbt_compile(downstream)
            downstream_results.append({
                "model": downstream,
                "status": ds_result["status"],
            })
        
        return {
            "status": compile_result["status"],
            "data": {
                "compile_result": compile_result,
                "downstream_impact": downstream_results,
                "files_modified": list(fix_option.get("code_changes", {}).keys()),
            }
        }
    finally:
        # Restore all backups
        for file_path, content in backups.items():
            Path(file_path).write_text(content)
```

---

### `app/agent/tools/fix_tool.py`

**Purpose**: Apply fixes to the codebase and run verification tests.

```python
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any, List
from .repo_tool import tool_read_repo, tool_write_repo, get_file_diff

class FixType(Enum):
    """Types of fixes that can be applied."""
    UPDATE_ACCEPTED_VALUES = "update_accepted_values"
    ADD_FILTER = "add_filter"
    ADD_COALESCE = "add_coalesce"
    UPDATE_RANGE = "update_range"
    SNOOZE = "snooze"
    ADD_WHERE_CLAUSE = "add_where_clause"

@dataclass
class FixOption:
    """Represents a fix option."""
    id: str
    title: str
    description: str
    fix_type: FixType
    risk_level: str          # 'low', 'medium', 'high'
    code_changes: Dict[str, str]  # file_path -> new_content
    rationale: str
    pros: str = ""
    cons: str = ""
    when_appropriate: str = ""

def apply_fix(fix_option: Dict, incident: Dict = None, 
              dry_run: bool = False) -> Dict[str, Any]:
    """Apply a fix option to the codebase.
    
    Args:
        fix_option: Fix option dict with code_changes
        incident: Original incident data (for logging)
        dry_run: If True, only show diff without applying
    
    Returns:
        Dict with status, diffs, and backup paths
    """
    code_changes = fix_option.get("code_changes", {})
    
    if not code_changes:
        return {
            "status": "error",
            "message": "No code changes to apply"
        }
    
    results = []
    
    for file_path, new_content in code_changes.items():
        # Read original
        read_result = tool_read_repo(file_path)
        if read_result["status"] != "success":
            results.append({
                "file": file_path,
                "status": "error",
                "message": f"Could not read file: {read_result.get('message')}"
            })
            continue
        
        original_content = read_result["data"]["content"]
        
        # Generate diff
        diff_result = get_file_diff(original_content, new_content)
        
        if dry_run:
            results.append({
                "file": file_path,
                "status": "dry_run",
                "diff": diff_result["data"]["diff"],
                "additions": diff_result["data"]["additions"],
                "deletions": diff_result["data"]["deletions"],
            })
        else:
            # Actually apply the change
            write_result = tool_write_repo(file_path, new_content)
            results.append({
                "file": file_path,
                "status": "success",
                "diff": diff_result["data"]["diff"],
                "backup_path": write_result.get("data", {}).get("backup_path"),
            })
    
    return {
        "status": "success",
        "data": {
            "results": results,
            "dry_run": dry_run,
        }
    }

def run_dbt_test(model_name: str, test_name: str = None) -> Dict[str, Any]:
    """Run dbt test to verify a fix.
    
    Args:
        model_name: Model to test
        test_name: Optional specific test name
    
    Returns:
        Dict with test results
    """
    import subprocess
    from app.config import get_settings
    
    settings = get_settings()
    
    cmd = ["dbt", "test", "--select", model_name]
    if test_name:
        cmd = ["dbt", "test", "--select", f"{model_name},{test_name}"]
    
    result = subprocess.run(
        cmd,
        cwd=settings.dbt_project_path,
        capture_output=True,
        text=True,
        timeout=120
    )
    
    return {
        "status": "success" if result.returncode == 0 else "failed",
        "data": {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "return_code": result.returncode,
        }
    }
```

---

### `app/agent/tools/agentic_fix_tool.py`

**Purpose**: Agent-driven fix generation (not hardcoded templates). This is the main tool used by the Fix Proposer agent.

```python
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from app.config import get_settings
from .repo_tool import tool_read_repo, tool_write_repo, find_schema_file, find_file_by_model_name, get_file_diff

# ============================================================
# Fixed Incidents Persistence
# ============================================================

def _get_fixed_incidents_path() -> str:
    """Get path to persistent fixed incidents file."""
    copilot_dir = Path.home() / ".dbt-copilot"
    copilot_dir.mkdir(exist_ok=True)
    return str(copilot_dir / "fixed_incidents.json")

def load_fixed_incidents() -> Dict[str, Any]:
    """Load fixed incidents from persistent storage."""
    path = _get_fixed_incidents_path()
    if Path(path).exists():
        return json.loads(Path(path).read_text())
    return {}

def save_fixed_incident(test_id: str, model_name: str, column_name: str,
                        test_name: str, fix_description: str,
                        verified: bool = False, backup_paths: List[str] = None) -> Dict[str, Any]:
    """Save a fixed incident to persistent storage."""
    incidents = load_fixed_incidents()
    incidents[test_id] = {
        "model_name": model_name,
        "column_name": column_name,
        "test_name": test_name,
        "fix_description": fix_description,
        "fixed_at": datetime.now().isoformat(),
        "verified": verified,
        "backup_paths": backup_paths or [],
    }
    Path(_get_fixed_incidents_path()).write_text(json.dumps(incidents, indent=2))
    return {"status": "success", "data": incidents[test_id]}

def is_incident_fixed(test_id: str) -> bool:
    """Check if an incident has been marked as fixed."""
    return test_id in load_fixed_incidents()

# ============================================================
# Code Generation Functions
# ============================================================

def adk_generate_schema_fix(model_name: str, column_name: str,
                            fix_type: str, fix_details: str) -> str:
    """Generate schema.yml fix for a dbt model.
    
    Args:
        model_name: Name of the dbt model
        column_name: Name of the column to fix
        fix_type: One of:
            - 'add_accepted_values': Add values to accepted_values test
            - 'update_range': Update min/max in range test
            - 'change_severity': Change test severity to 'warn'
            - 'add_test': Add a new test
            - 'remove_test': Remove an existing test
        fix_details: JSON string with fix-specific parameters
    
    Returns:
        JSON string with generated diff and file paths
    """
    import yaml
    
    # Find schema file
    schema_result = find_schema_file(model_name)
    if schema_result["status"] != "success":
        return json.dumps({"status": "error", "message": "Schema file not found"})
    
    file_path = schema_result["data"]["path"]
    
    # Read current content
    read_result = tool_read_repo(file_path)
    original_content = read_result["data"]["content"]
    
    # Parse YAML and apply fix
    data = yaml.safe_load(original_content)
    details = json.loads(fix_details) if isinstance(fix_details, str) else fix_details
    
    # Find and modify the target column
    for model in data.get("models", []):
        if model.get("name") == model_name:
            for column in model.get("columns", []):
                if column.get("name") == column_name:
                    if fix_type == "add_accepted_values":
                        # Add values to accepted_values test
                        new_values = details.get("values", [])
                        tests = column.get("tests", [])
                        for test in tests:
                            if isinstance(test, dict) and "accepted_values" in test:
                                current = test["accepted_values"].get("values", [])
                                test["accepted_values"]["values"] = list(set(current + new_values))
                                break
                    # ... handle other fix types
    
    # Generate new YAML
    new_content = yaml.dump(data, default_flow_style=False, sort_keys=False)
    
    # Generate diff
    diff_result = get_file_diff(original_content, new_content)
    
    return json.dumps({
        "status": "success",
        "file_path": file_path,
        "diff": diff_result["data"]["diff"],
        "original_content": original_content,
        "new_content": new_content,
    }, indent=2)

def adk_generate_sql_fix(model_name: str, column_name: str,
                         fix_type: str, fix_code: str) -> str:
    """Generate SQL model fix.
    
    Args:
        model_name: Name of the dbt model
        column_name: Name of the column
        fix_type: One of:
            - 'add_where_clause': Add WHERE condition
            - 'add_coalesce': Wrap column with COALESCE
            - 'replace_column': Replace column expression
            - 'add_filter': Add filter (alias for add_where_clause)
        fix_code: The SQL code to apply
    
    Returns:
        JSON string with generated diff and file paths
    """
    # ... implementation similar to schema fix

def adk_propose_fix(test_name: str, model_name: str, column_name: str,
                    root_cause: str, fix_options: str) -> str:
    """Main tool for agents to propose fixes.
    
    This is called by the Fix Proposer agent after receiving the diagnosis.
    The agent provides structured fix options which are validated and enhanced.
    
    Args:
        test_name: Name of the failing test
        model_name: Name of the dbt model
        column_name: Name of the column (if applicable)
        root_cause: Diagnosed root cause from Diagnostician
        fix_options: JSON string with array of fix options, each with:
            - id: Unique identifier
            - title: Clear action title
            - description: What this fix does
            - fix_type: 'schema' or 'sql'
            - code_change: Object with action and details/code
            - rationale: Why this fix works
            - pros: Advantages
            - cons: Disadvantages
            - when_appropriate: When to use this fix
    
    Returns:
        JSON string with validated fix options ready for UI display
    """
    try:
        options = json.loads(fix_options) if isinstance(fix_options, str) else fix_options
    except json.JSONDecodeError as e:
        return json.dumps({"status": "error", "message": f"Invalid JSON: {e}"})
    
    validated_options = []
    
    for opt in options:
        if not opt.get("title") or not opt.get("fix_type"):
            continue
        
        validated_opt = {
            "id": opt.get("id") or str(uuid.uuid4()),
            "title": opt.get("title"),
            "description": opt.get("description", ""),
            "fix_type": opt.get("fix_type"),
            "rationale": opt.get("rationale", ""),
            "pros": opt.get("pros", ""),
            "cons": opt.get("cons", ""),
            "when_appropriate": opt.get("when_appropriate", ""),
            "root_cause": root_cause,
            "code_change": opt.get("code_change", {}),
        }
        
        # Pre-generate diffs if code_change provided
        code_change = opt.get("code_change", {})
        if code_change and opt.get("fix_type") == "schema":
            diff_result = adk_generate_schema_fix(
                model_name, column_name,
                code_change.get("action", ""),
                json.dumps(code_change.get("details", {}))
            )
            diff_data = json.loads(diff_result)
            if diff_data.get("status") == "success":
                validated_opt["code_changes"] = {
                    diff_data["file_path"]: diff_data["new_content"]
                }
                validated_opt["diff_preview"] = diff_data.get("diff", "")
        
        validated_options.append(validated_opt)
    
    return json.dumps({
        "status": "success",
        "message": f"Generated {len(validated_options)} fix option(s)",
        "data": {
            "test_name": test_name,
            "model_name": model_name,
            "column_name": column_name,
            "root_cause": root_cause,
            "options": validated_options,
        }
    }, indent=2)
```

---

## Prompts Module

### `app/prompts/__init__.py`

**Purpose**: Centralized prompt exports.

```python
"""Centralized prompt management for dbt Co-Work."""

from app.prompts.agent_prompts import (
    COPILOT_SYSTEM_INSTRUCTION,
    get_investigation_prompt,
)
from app.prompts.fix_prompts import get_fix_enhancement_prompt
from app.prompts.multi_agent_prompts import (
    INVESTIGATOR_SYSTEM_INSTRUCTION,
    DIAGNOSTICIAN_SYSTEM_INSTRUCTION,
    FIX_PROPOSER_SYSTEM_INSTRUCTION,
    get_investigator_prompt,
    get_diagnostician_prompt,
    get_fix_proposer_prompt,
)

__all__ = [
    "COPILOT_SYSTEM_INSTRUCTION",
    "get_investigation_prompt",
    "get_fix_enhancement_prompt",
    "INVESTIGATOR_SYSTEM_INSTRUCTION",
    "DIAGNOSTICIAN_SYSTEM_INSTRUCTION",
    "FIX_PROPOSER_SYSTEM_INSTRUCTION",
    "get_investigator_prompt",
    "get_diagnostician_prompt",
    "get_fix_proposer_prompt",
]
```

---

### `app/prompts/agent_prompts.py`

**Purpose**: Legacy single-agent system instruction and prompts.

```python
COPILOT_SYSTEM_INSTRUCTION = """
You are dbt Co-Work, an expert Analytics Engineering assistant specializing in 
investigating and resolving dbt test failures. You have deep knowledge of:

- dbt (data build tool) architecture, models, tests, and configurations
- SQL best practices and common data quality issues
- Data pipeline patterns and root cause analysis

## Your Tools
You have access to these investigation tools:
1. adk_get_model_lineage - Get upstream/downstream model dependencies
2. adk_read_model_sql - Read SQL transformation code
3. adk_read_schema_definition - Read schema.yml with tests and columns
4. adk_search_knowledge_base - Search business rules and policies
5. adk_execute_sql - Execute read-only SQL queries
6. adk_propose_fix - Propose fix options (call after diagnosis)

## Response Structure (CRITICAL)
Your response MUST contain these three sections:

### SECTION 1: Investigation Steps
Document each step of your investigation:
- Step 1: [Your reasoning and tool calls]
- Step 2: [Your reasoning and tool calls]
...

### SECTION 2: Final Diagnosis
Wrap your diagnosis in tags:
<final_diagnosis>
### Root Cause
[Clear explanation of what caused the failure]

### Evidence
[List of evidence supporting your diagnosis]

### Impact Assessment
[Business and data impact]
</final_diagnosis>

### SECTION 3: Fix Recommendations
After completing diagnosis, call the adk_propose_fix tool with 4-5 fix options.
Each option should have pros, cons, and when-to-use guidance.
"""

def get_investigation_prompt(test_name: str, model_name: str, column_name: str,
                             error_message: str, failed_rows: int, test_id: str) -> str:
    """Generate investigation task prompt for agent."""
    return f"""
## Test Failure Alert

A dbt test has failed and requires investigation:

| Field | Value |
|-------|-------|
| Test Name | {test_name} |
| Model | {model_name} |
| Column | {column_name or 'N/A'} |
| Error | {error_message} |
| Failed Rows | {failed_rows} |
| Test ID | {test_id} |

## Your Task
1. Investigate the root cause using available tools
2. Check model lineage for upstream issues
3. Review SQL code and schema definitions
4. Consult knowledge base for business rules
5. Provide diagnosis with evidence
6. Propose 4-5 fix options using adk_propose_fix tool

Begin your investigation now.
"""
```

---

### `app/prompts/multi_agent_prompts.py`

**Purpose**: Specialized prompts for multi-agent architecture.

```python
# ============================================================
# Investigator Agent
# ============================================================

INVESTIGATOR_SYSTEM_INSTRUCTION = """
You are the Investigator Agent in a multi-agent system for dbt test failure resolution.

Your Role: Gather ALL relevant context using your tools. Do NOT diagnose or suggest fixes.

Available Tools:
1. adk_get_model_lineage - Get upstream/downstream dependencies
2. adk_read_model_sql - Read SQL transformation code
3. adk_read_schema_definition - Read schema.yml definitions
4. adk_search_knowledge_base - Search business documentation
5. adk_get_test_details - Get detailed test failure info
6. adk_execute_sql - Execute investigative SQL queries

Your Output:
Provide a structured summary of all gathered context. Another agent will analyze this.
"""

def get_investigator_prompt(test_name, model_name, column_name, 
                           error_message, failed_rows, test_id) -> str:
    """Generate task prompt for Investigator Agent."""
    return f"""
Investigate this test failure and gather all relevant context:

Test: {test_name}
Model: {model_name}
Column: {column_name or 'N/A'}
Error: {error_message}
Failed Rows: {failed_rows}

Gather:
1. Model lineage (upstream and downstream)
2. SQL transformation code
3. Schema definition with test configuration
4. Relevant business rules from knowledge base
5. Sample data if helpful (use adk_execute_sql)

Compile all findings into a structured context summary.
"""

# ============================================================
# Diagnostician Agent
# ============================================================

DIAGNOSTICIAN_SYSTEM_INSTRUCTION = """
You are the Diagnostician Agent in a multi-agent system for dbt test failure resolution.

Your Role: Analyze the investigation context and produce a structured diagnosis.

You have NO tools - your job is pure analysis and reasoning.

Your Output Format:
<diagnosis>
ROOT_CAUSE: [One clear sentence explaining the root cause]
EVIDENCE:
- [Evidence point 1]
- [Evidence point 2]
IMPACT: [Business and data impact assessment]
CATEGORY: [schema_mismatch|data_quality|upstream_issue|config_error|business_logic]
SEVERITY: [critical|high|medium|low]
CONFIDENCE: [high|medium|low]
</diagnosis>
"""

def get_diagnostician_prompt(test_name, model_name, column_name,
                             error_message, investigation_context) -> str:
    """Generate task prompt for Diagnostician Agent."""
    return f"""
Analyze this investigation context and provide a diagnosis:

## Test Failure
- Test: {test_name}
- Model: {model_name}
- Column: {column_name or 'N/A'}
- Error: {error_message}

## Investigation Context
{investigation_context}

Provide your diagnosis in the specified format.
"""

# ============================================================
# Fix Proposer Agent
# ============================================================

FIX_PROPOSER_SYSTEM_INSTRUCTION = """
You are the Fix Proposer Agent in a multi-agent system for dbt test failure resolution.

Your Role: Generate 4-5 fix options based on the diagnosis.

You have ONE tool: adk_propose_fix

Your Task:
1. Review the diagnosis and context
2. Generate 4-5 fix options covering different approaches
3. Call adk_propose_fix with structured options

Each option must include:
- id: Unique identifier (fix_1, fix_2, etc.)
- title: Clear action title
- description: What this fix does
- fix_type: 'schema' or 'sql'
- code_change: {action: '...', details/code: '...'}
- rationale: Why this works
- pros: Advantages
- cons: Disadvantages
- when_appropriate: When to use this fix
"""

def get_fix_proposer_prompt(test_name, model_name, column_name,
                            diagnosis, schema_definition, sql_code) -> str:
    """Generate task prompt for Fix Proposer Agent."""
    return f"""
Generate fix options for this diagnosed test failure:

## Diagnosis
{diagnosis}

## Context
- Test: {test_name}
- Model: {model_name}
- Column: {column_name or 'N/A'}

## Schema Definition
{schema_definition}

## SQL Code
{sql_code}

Call adk_propose_fix with 4-5 fix options as a JSON array.
"""
```

---

### `app/prompts/fix_prompts.py`

**Purpose**: Fix enhancement prompts.

```python
def get_fix_enhancement_prompt(test_name: str, model_name: str, column_name: str,
                               error_message: str, diagnosis: str,
                               sql_excerpt: str, options_text: str) -> str:
    """Generate prompt to enhance fix options with specific code.
    
    Used to refine agent-generated options with actual code snippets.
    """
    return f"""Based on this test failure analysis, enhance these fix options with specific code.

## Test Failure
- Test: {test_name}
- Model: {model_name}
- Column: {column_name}
- Error: {error_message}

## Diagnosis
{diagnosis}

## SQL Code (excerpt)
```sql
{sql_excerpt}
```

## Available Fix Options
{options_text}

For each option, provide:
1. **Specific code change** - The exact SQL or YAML code
2. **Why this works** - Technical explanation
3. **Business impact** - Who/what is affected
4. **Pros** - Advantages
5. **Cons** - Tradeoffs
6. **When appropriate** - Best scenarios for this fix

Return as JSON array."""
```

---

## Database Module

### `app/db/__init__.py`

```python
from app.db.mock_elementary import MockElementaryDB

__all__ = ["MockElementaryDB"]
```

---

### `app/db/mock_elementary.py`

**Purpose**: Mock Elementary database for demo/testing without BigQuery.

```python
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

@dataclass
class MockTestResult:
    """Mock test result matching Elementary schema."""
    test_id: str
    test_name: str
    model_name: str
    column_name: Optional[str]
    status: str                          # 'pass', 'fail', 'warn', 'error'
    error_message: Optional[str]
    failed_rows: int
    failed_row_samples: List[Dict[str, Any]]
    test_type: str                       # 'accepted_values', 'not_null', etc.
    severity: str                        # 'ERROR', 'WARN'
    executed_at: datetime = field(default_factory=datetime.now)

class MockElementaryDB:
    """Mock database with pre-populated test failures."""
    
    def __init__(self):
        self._results: List[MockTestResult] = []
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Create realistic mock test failures."""
        now = datetime.now()
        
        # Failure 1: Unexpected sentiment values
        self._results.append(MockTestResult(
            test_id="test_accepted_values_fact_reviews_sentiment",
            test_name="accepted_values_fact_reviews_sentiment",
            model_name="fact_reviews",
            column_name="sentiment",
            status="fail",
            error_message="Got unexpected values: ['mixed', 'unknown']",
            failed_rows=156,
            failed_row_samples=[
                {"review_id": 12345, "sentiment": "mixed"},
                {"review_id": 12346, "sentiment": "unknown"},
            ],
            test_type="accepted_values",
            severity="ERROR",
            executed_at=now - timedelta(hours=2),
        ))
        
        # Failure 2: NULL host_id values
        self._results.append(MockTestResult(
            test_id="test_not_null_dim_listing_host_id",
            test_name="not_null_dim_listing_host_id",
            model_name="dim_listing",
            column_name="host_id",
            status="fail",
            error_message="12 NULL values found",
            failed_rows=12,
            failed_row_samples=[
                {"listing_id": 99901, "host_id": None},
                {"listing_id": 99902, "host_id": None},
            ],
            test_type="not_null",
            severity="ERROR",
            executed_at=now - timedelta(hours=1),
        ))
        
        # ... more mock failures for:
        # - New room type ('Studio')
        # - Price out of range
        # - Orphan reviews (referential integrity)
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all test results as dicts."""
        return [self._to_dict(r) for r in self._results]
    
    def get_failed_results(self) -> List[Dict[str, Any]]:
        """Get only failed test results."""
        return [self._to_dict(r) for r in self._results if r.status == "fail"]
    
    def get_result_by_id(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get specific test result by ID."""
        for r in self._results:
            if r.test_id == test_id:
                return self._to_dict(r)
        return None
    
    def _to_dict(self, result: MockTestResult) -> Dict[str, Any]:
        """Convert MockTestResult to dict."""
        return {
            "test_id": result.test_id,
            "test_name": result.test_name,
            "model_name": result.model_name,
            "column_name": result.column_name,
            "status": result.status,
            "error_message": result.error_message,
            "failed_rows": result.failed_rows,
            "failed_row_samples": result.failed_row_samples,
            "test_type": result.test_type,
            "severity": result.severity,
            "executed_at": result.executed_at.isoformat(),
        }
```

---

## UI Module

### `app/ui/__init__.py`

```python
from app.ui.components import icon, render_header, render_sidebar
from app.ui.mission_control import render_mission_control
from app.ui.resolution_studio import render_resolution_studio
from app.ui.snoozed import render_snoozed_page

__all__ = [
    "icon",
    "render_header",
    "render_sidebar",
    "render_mission_control",
    "render_resolution_studio",
    "render_snoozed_page",
]
```

---

### `app/ui/components.py`

**Purpose**: Shared UI components and styling.

```python
import streamlit as st

def icon(name: str, size: int = 20, color: str = "#FF683B") -> str:
    """Return HTML for a Material Icon.
    
    Args:
        name: Material icon name (e.g., 'bug_report', 'check_circle')
        size: Icon size in pixels
        color: CSS color
    
    Returns:
        HTML string to embed in st.markdown
    """
    return f'<span class="material-icons" style="font-size: {size}px; color: {color};">{name}</span>'

def render_header():
    """Render application header with logo."""
    st.markdown("""
        <div style="display: flex; align-items: center; margin-bottom: 1rem;">
            <span style="font-size: 2rem;">🚀</span>
            <h1 style="margin-left: 0.5rem;">dbt Co-Work</h1>
        </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render sidebar with navigation and settings."""
    with st.sidebar:
        st.markdown("### Settings")
        
        # Mock data toggle
        use_mock = st.checkbox("Use Mock Data", value=True)
        
        # Connection status
        st.markdown("---")
        st.markdown("### Status")
        st.success("✓ Connected")

def render_code_block(code: str, language: str = "sql"):
    """Render syntax-highlighted code block."""
    st.code(code, language=language)

def render_diff_view(diff_text: str):
    """Render git-style diff view with colors."""
    lines = diff_text.split('\n')
    html_lines = []
    
    for line in lines:
        if line.startswith('+') and not line.startswith('+++'):
            html_lines.append(f'<div style="background: #d4edda; color: #155724;">{line}</div>')
        elif line.startswith('-') and not line.startswith('---'):
            html_lines.append(f'<div style="background: #f8d7da; color: #721c24;">{line}</div>')
        else:
            html_lines.append(f'<div>{line}</div>')
    
    st.markdown(
        f'<pre style="font-family: monospace; font-size: 12px;">{"".join(html_lines)}</pre>',
        unsafe_allow_html=True
    )
```

---

### `app/ui/mission_control.py`

**Purpose**: Dashboard page showing active incidents.

```python
import streamlit as st
from typing import List, Dict, Any
from app.agent.tools import get_failed_tests
from app.agent.tools.agentic_fix_tool import is_incident_fixed
from app.ui.snoozed import is_snoozed
from app.ui.components import icon

def render_mission_control():
    """Render the Dashboard (home) view."""
    st.title("Mission Control")
    st.markdown("Monitor and manage data quality incidents")
    
    # Fetch test results
    result = get_failed_tests()
    if result["status"] != "success":
        st.error("Failed to fetch test results")
        return
    
    failures = result["data"]["results"]
    
    # Filter out snoozed and fixed incidents
    active_failures = [
        f for f in failures
        if not is_snoozed(f.get("test_id")) and not is_incident_fixed(f.get("test_id"))
    ]
    
    # Render statistics
    render_stats(active_failures)
    
    # Render incident cards
    st.markdown("### Active Incidents")
    
    # Sort by severity and time
    failures_sorted = sorted(
        active_failures,
        key=lambda x: (x.get("severity") != "ERROR", x.get("executed_at")),
    )
    
    for idx, failure in enumerate(failures_sorted):
        render_incident_card(failure, idx)

def render_stats(failures: List[Dict[str, Any]]):
    """Render statistics cards."""
    total = len(failures)
    errors = sum(1 for f in failures if f.get("severity") == "ERROR")
    warnings = sum(1 for f in failures if f.get("severity") == "WARN")
    models = len(set(f.get("model_name") for f in failures))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container(border=True):
            st.markdown(f'{icon("bug_report", 18)} Total Failures', unsafe_allow_html=True)
            st.metric("Total Failures", total, label_visibility="collapsed")
    
    with col2:
        with st.container(border=True):
            st.markdown(f'{icon("error", 18)} Critical', unsafe_allow_html=True)
            st.metric("Critical", errors, label_visibility="collapsed")
    
    # ... col3, col4 for warnings and models affected

def render_incident_card(failure: Dict[str, Any], idx: int):
    """Render single incident card."""
    severity = failure.get("severity", "ERROR")
    test_name = failure.get("test_name", "Unknown")
    model_name = failure.get("model_name", "Unknown")
    
    with st.container(border=True):
        col1, col2, col3 = st.columns([3, 1, 1])
        
        with col1:
            severity_icon = "error" if severity == "ERROR" else "warning"
            st.markdown(
                f'{icon(severity_icon)} **{test_name}** on `{model_name}`',
                unsafe_allow_html=True
            )
            st.caption(failure.get("error_message", "")[:100])
        
        with col2:
            st.metric("Failed Rows", failure.get("failed_rows", 0))
        
        with col3:
            def start_investigation(incident):
                st.session_state.selected_incident = incident
                st.session_state.page = "labs"
            
            st.button(
                "Investigate",
                key=f"investigate_{idx}",
                on_click=start_investigation,
                args=(failure,),
                type="primary"
            )
```

---

### `app/ui/resolution_studio.py`

**Purpose**: Labs page - investigation and fix recommendation UI. This is the largest UI file (~1500 lines).

```python
import streamlit as st
import asyncio
from typing import Dict, Any, List
from app.agent.multi_agent_copilot import (
    MultiAgentCopilot,
    Investigation,
    InvestigationStep,
    Diagnosis,
)
from app.agent.tools import tool_read_repo, get_model_lineage
from app.agent.tools.repo_tool import find_file_by_model_name, find_schema_file
from app.agent.tools.dbt_tool import run_dry_run
from app.ui.components import icon, render_diff_view

def render_resolution_studio():
    """Main Labs view - two-panel layout."""
    # Check for selected incident
    if "selected_incident" not in st.session_state:
        st.info("Select an incident from the Dashboard to investigate")
        return
    
    incident = st.session_state.selected_incident
    
    # Two-column layout
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        render_context_panel(incident)
    
    with col_right:
        render_investigation_panel(incident)

def render_context_panel(incident: Dict[str, Any]):
    """Render left panel with context information."""
    st.markdown("### Context")
    
    # Test Context expander
    with st.expander("📋 Test Context", expanded=True):
        st.markdown(f"**Test:** {incident.get('test_name')}")
        st.markdown(f"**Model:** {incident.get('model_name')}")
        st.markdown(f"**Column:** {incident.get('column_name', 'N/A')}")
        st.markdown(f"**Failed Rows:** {incident.get('failed_rows', 0)}")
    
    # Error Log expander
    with st.expander("🚨 Error Log", expanded=True):
        st.error(incident.get("error_message", "No error message"))
    
    # Model SQL expander
    model_name = incident.get("model_name")
    with st.expander("📄 Model SQL", expanded=False):
        file_result = find_file_by_model_name(model_name)
        if file_result["status"] == "success":
            read_result = tool_read_repo(file_result["data"]["path"])
            st.code(read_result["data"]["content"], language="sql")
    
    # Schema Definition expander
    with st.expander("📐 Schema Definition", expanded=False):
        schema_result = find_schema_file(model_name)
        if schema_result["status"] == "success":
            read_result = tool_read_repo(schema_result["data"]["path"])
            st.code(read_result["data"]["content"], language="yaml")

def render_investigation_panel(incident: Dict[str, Any]):
    """Render right panel with agent investigation."""
    st.markdown("### AI Investigation")
    
    # Failed Sample Rows
    if incident.get("failed_row_samples"):
        with st.expander("Sample Failed Rows", expanded=True):
            st.dataframe(incident["failed_row_samples"])
    
    # Investigation controls
    if "investigation_running" not in st.session_state:
        st.session_state.investigation_running = False
    
    if st.button("Start Investigation", type="primary", 
                 disabled=st.session_state.investigation_running):
        st.session_state.investigation_running = True
        run_investigation_with_live_display(incident)
    
    # Display investigation results if available
    if "investigation_result" in st.session_state:
        render_investigation_results()

def run_investigation_with_live_display(incident: Dict[str, Any]):
    """Run multi-agent investigation with streaming display."""
    
    # Create progress containers
    investigator_container = st.container()
    diagnostician_container = st.container()
    fix_proposer_container = st.container()
    
    async def run_async():
        copilot = MultiAgentCopilot()
        steps = []
        
        async for step in copilot.investigate(incident):
            steps.append(step)
            
            # Update appropriate container based on step source
            if "investigator" in step.action.lower():
                with investigator_container:
                    render_step(step)
            elif "diagnostician" in step.action.lower():
                with diagnostician_container:
                    render_step(step)
            elif "fix_proposer" in step.action.lower():
                with fix_proposer_container:
                    render_step(step)
        
        return steps
    
    # Run async investigation
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    steps = loop.run_until_complete(run_async())
    
    # Store results
    st.session_state.investigation_steps = steps
    st.session_state.investigation_running = False
    st.rerun()

def render_step(step: InvestigationStep):
    """Render a single investigation step."""
    status_icon = "✅" if step.status == "success" else "⏳" if step.status == "thinking" else "❌"
    
    with st.expander(f"{status_icon} {step.action}", expanded=False):
        if step.tool_name:
            st.caption(f"Tool: {step.tool_name}")
        st.markdown(step.input_summary)
        if step.output_summary:
            st.code(step.output_summary[:500], language="json")

def render_investigation_results():
    """Render completed investigation results."""
    # Diagnosis section
    if "diagnosis" in st.session_state:
        diagnosis = st.session_state.diagnosis
        st.markdown("### 🔬 Diagnosis")
        st.info(diagnosis.root_cause)
        
        with st.expander("Evidence"):
            for evidence in diagnosis.evidence:
                st.markdown(f"- {evidence}")
    
    # Fix options section
    if "fix_options" in st.session_state:
        st.markdown("### 🛠️ Fix Options")
        render_fix_options(st.session_state.fix_options)

def render_fix_options(options: List[Dict[str, Any]]):
    """Render fix option cards with selection."""
    for i, option in enumerate(options):
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"**{option.get('title')}**")
                st.markdown(option.get('description', ''))
                
                # Pros/Cons
                if option.get('pros'):
                    st.markdown(f"✅ **Pros:** {option.get('pros')}")
                if option.get('cons'):
                    st.markdown(f"⚠️ **Cons:** {option.get('cons')}")
                
                # Code preview
                if option.get('diff_preview'):
                    with st.expander("View Code Changes"):
                        render_diff_view(option['diff_preview'])
            
            with col2:
                if st.button("Dry Run", key=f"dry_run_{i}"):
                    result = run_dry_run(option, st.session_state.selected_incident["model_name"])
                    if result["status"] == "success":
                        st.success("Dry run passed!")
                    else:
                        st.error("Dry run failed")
                
                if st.button("Apply Fix", key=f"apply_{i}", type="primary"):
                    apply_selected_fix(option)

def apply_selected_fix(option: Dict[str, Any]):
    """Apply the selected fix option."""
    from app.agent.tools import apply_fix
    from app.agent.tools.agentic_fix_tool import save_fixed_incident
    
    result = apply_fix(option, st.session_state.selected_incident)
    
    if result["status"] == "success":
        # Save as fixed
        incident = st.session_state.selected_incident
        save_fixed_incident(
            test_id=incident["test_id"],
            model_name=incident["model_name"],
            column_name=incident.get("column_name", ""),
            test_name=incident["test_name"],
            fix_description=option["title"],
        )
        st.success("Fix applied successfully!")
        st.balloons()
    else:
        st.error(f"Failed to apply fix: {result.get('message')}")
```

---

### `app/ui/snoozed.py`

**Purpose**: Snoozed incidents management.

```python
import streamlit as st
from datetime import datetime, timedelta
from typing import List, Dict, Any

def _get_snoozed_items() -> List[Dict[str, Any]]:
    """Get snoozed items from session state."""
    if "snoozed_items" not in st.session_state:
        st.session_state.snoozed_items = []
    return st.session_state.snoozed_items

def add_snoozed_item(incident: Dict[str, Any], duration_hours: int = 24):
    """Add an incident to snoozed list."""
    snoozed = _get_snoozed_items()
    
    snoozed.append({
        "test_id": incident["test_id"],
        "test_name": incident["test_name"],
        "model_name": incident["model_name"],
        "snoozed_at": datetime.now().isoformat(),
        "snooze_until": (datetime.now() + timedelta(hours=duration_hours)).isoformat(),
        "reason": incident.get("snooze_reason", ""),
    })
    
    st.session_state.snoozed_items = snoozed

def is_snoozed(test_id: str) -> bool:
    """Check if a test is currently snoozed."""
    snoozed = _get_snoozed_items()
    now = datetime.now()
    
    for item in snoozed:
        if item["test_id"] == test_id:
            snooze_until = datetime.fromisoformat(item["snooze_until"])
            if now < snooze_until:
                return True
    return False

def unsnooze_item(test_id: str):
    """Remove an item from snoozed list."""
    snoozed = _get_snoozed_items()
    st.session_state.snoozed_items = [
        item for item in snoozed if item["test_id"] != test_id
    ]

def render_snoozed_page():
    """Render the Snoozed page."""
    st.title("Snoozed Incidents")
    
    snoozed = _get_snoozed_items()
    
    if not snoozed:
        st.info("No snoozed incidents")
        return
    
    for item in snoozed:
        with st.container(border=True):
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f"**{item['test_name']}** on `{item['model_name']}`")
                st.caption(f"Snoozed until: {item['snooze_until']}")
            
            with col2:
                remaining = datetime.fromisoformat(item['snooze_until']) - datetime.now()
                st.metric("Time Left", f"{remaining.total_seconds() / 3600:.1f}h")
            
            with col3:
                if st.button("Unsnooze", key=f"unsnooze_{item['test_id']}"):
                    unsnooze_item(item['test_id'])
                    st.rerun()
```

---

### `app/ui/fixed_tests.py`

**Purpose**: View resolved/fixed tests.

```python
import streamlit as st
from datetime import datetime
from typing import Dict, Any
from app.agent.tools.agentic_fix_tool import load_fixed_incidents, remove_fixed_incident

def render_fixed_tests_page():
    """Render the Resolved page showing fixed tests."""
    st.title("Resolved Incidents")
    
    fixed = load_fixed_incidents()
    
    if not fixed:
        st.info("No resolved incidents yet")
        return
    
    for test_id, data in fixed.items():
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"**{data['test_name']}** on `{data['model_name']}`")
                st.caption(f"Fixed: {data['fixed_at']}")
                st.markdown(f"Fix: {data['fix_description']}")
                
                if data.get('verified'):
                    st.success("✓ Verified with dbt test")
            
            with col2:
                if st.button("Reopen", key=f"reopen_{test_id}"):
                    remove_fixed_incident(test_id)
                    st.rerun()
```

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Streamlit UI                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │  Dashboard   │  │    Labs      │  │   Snoozed    │              │
│  │  (Mission    │  │  (Resolution │  │   & Fixed    │              │
│  │   Control)   │  │   Studio)    │  │              │              │
│  └──────────────┘  └──────────────┘  └──────────────┘              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Agent Copilot (ADK)                         │
│  ┌─────────────┐    ┌──────────────┐    ┌─────────────────┐        │
│  │Investigator │───▶│Diagnostician │───▶│  Fix Proposer   │        │
│  │ (6 tools)   │    │ (no tools)   │    │   (1 tool)      │        │
│  └─────────────┘    └──────────────┘    └─────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                          Tool Layer                                  │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐           │
│  │ Manifest  │ │   Repo    │ │Elementary │ │ Knowledge │           │
│  │   Tool    │ │   Tool    │ │   Tool    │ │Base Tool  │           │
│  └───────────┘ └───────────┘ └───────────┘ └───────────┘           │
│  ┌───────────┐ ┌───────────┐ ┌───────────┐                         │
│  │  SQL Tool │ │  dbt Tool │ │ Fix Tool  │                         │
│  └───────────┘ └───────────┘ └───────────┘                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
        ┌───────────────────────┼───────────────────────┐
        ▼                       ▼                       ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│  dbt Project  │    │   BigQuery    │    │  Knowledge    │
│  (manifest,   │    │  (Elementary  │    │    Base       │
│   SQL, YAML)  │    │   results)    │    │  (markdown)   │
└───────────────┘    └───────────────┘    └───────────────┘
```

---

## Data Flow

1. **Dashboard Load**
   - `render_mission_control()` calls `get_failed_tests()`
   - `get_failed_tests()` queries Elementary (mock or BigQuery)
   - Filter out snoozed and fixed incidents
   - Display incident cards

2. **User Clicks Investigate**
   - Set `st.session_state.selected_incident`
   - Navigate to Labs page
   - `render_resolution_studio()` displays context panels

3. **Start Investigation**
   - Create `MultiAgentCopilot` instance
   - **Stage 1 - Investigator**:
     - Calls tools: lineage, SQL, schema, KB, execute_sql
     - Compiles `InvestigationContext`
   - **Stage 2 - Diagnostician**:
     - Analyzes context (no tools)
     - Produces structured `Diagnosis`
   - **Stage 3 - Fix Proposer**:
     - Generates fix options via `adk_propose_fix`
     - Returns validated options with code changes

4. **User Selects Fix**
   - Click "Dry Run" → `run_dry_run()` validates
   - Click "Apply Fix" → `apply_fix()` writes files
   - `save_fixed_incident()` marks as resolved

5. **Post-Fix Verification**
   - Optionally run `run_dbt_test()` to verify
   - Update incident status

---

## File Summary Table

| File | Lines | Purpose |
|------|-------|---------|
| `config.example.env` | ~50 | Environment configuration template |
| `requirements.txt` | ~15 | Python dependencies |
| `run.sh` | ~20 | One-line setup and launch |
| `app/main.py` | ~100 | Streamlit entry point |
| `app/config/settings.py` | ~150 | Pydantic configuration |
| `app/agent/copilot_agent.py` | ~800 | Legacy single-agent |
| `app/agent/multi_agent_copilot.py` | ~600 | Multi-agent orchestrator |
| `app/agent/tools/manifest_tool.py` | ~150 | dbt manifest parsing |
| `app/agent/tools/repo_tool.py` | ~200 | File operations |
| `app/agent/tools/elementary_tool.py` | ~150 | Test result queries |
| `app/agent/tools/knowledge_base_tool.py` | ~250 | Semantic search |
| `app/agent/tools/sql_tool.py` | ~150 | SQL execution |
| `app/agent/tools/dbt_tool.py` | ~150 | dbt CLI operations |
| `app/agent/tools/fix_tool.py` | ~150 | Fix application |
| `app/agent/tools/agentic_fix_tool.py` | ~600 | Agent-driven fixes |
| `app/prompts/agent_prompts.py` | ~200 | Single-agent prompts |
| `app/prompts/multi_agent_prompts.py` | ~300 | Multi-agent prompts |
| `app/prompts/fix_prompts.py` | ~100 | Fix enhancement prompts |
| `app/db/mock_elementary.py` | ~200 | Mock test data |
| `app/ui/components.py` | ~100 | Shared UI components |
| `app/ui/mission_control.py` | ~150 | Dashboard page |
| `app/ui/resolution_studio.py` | ~1500 | Investigation UI |
| `app/ui/snoozed.py` | ~100 | Snooze management |
| `app/ui/fixed_tests.py` | ~80 | Resolved incidents |

---

*Generated on January 19, 2026*
