"""
Tool implementations for the dbt Co-Work agent.
"""

from .manifest_tool import tool_read_manifest, get_model_lineage
from .repo_tool import tool_read_repo, tool_write_repo
from .elementary_tool import tool_query_elementary, get_failed_tests
from .knowledge_base_tool import tool_consult_knowledge_base
from .fix_tool import apply_fix
from .fix_tool import run_dbt_test as simple_run_dbt_test  # Basic test runner
from .dbt_tool import run_dbt_compile, run_dry_run, check_dbt_installed, run_dbt_test, preview_impact
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
    "run_dbt_test",  # Full test runner with fix application from dbt_tool
    "simple_run_dbt_test",  # Basic test runner from fix_tool
    "run_dbt_compile",
    "run_dry_run",
    "preview_impact",
    "check_dbt_installed",
    # SQL execution tools
    "tool_execute_sql",
    "adk_execute_sql",
    # Agentic fix tools
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
