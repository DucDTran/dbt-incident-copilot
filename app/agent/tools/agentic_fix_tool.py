"""
Agentic Fix Tools for dbt Co-Work.

These tools allow the agent to propose and generate code fixes autonomously,
rather than using hardcoded templates.
"""

from typing import Dict, Any, List, Optional
import json
import os
import difflib
import uuid
from datetime import datetime

from app.config import get_settings
from google import genai
from google.genai import types

from .repo_tool import (
    tool_read_repo, 
    tool_write_repo, 
    find_schema_file, 
    find_variable_definition,
    list_model_files,
    find_file_by_model_name,
    get_file_diff
)


# ============================================================
# Fixed Incidents Persistence
# ============================================================

def _get_fixed_incidents_path() -> str:
    """Get the path to the fixed incidents JSON file."""
    settings = get_settings()
    copilot_dir = os.path.expanduser("~/.dbt-copilot")
    os.makedirs(copilot_dir, exist_ok=True)
    return os.path.join(copilot_dir, "fixed_incidents.json")


def load_fixed_incidents() -> Dict[str, Any]:
    """Load fixed incidents from persistent storage."""
    path = _get_fixed_incidents_path()
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_fixed_incident(
    test_id: str,
    model_name: str,
    column_name: str,
    test_name: str,
    fix_description: str,
    verified: bool = False,
    backup_paths: List[str] = None
) -> Dict[str, Any]:
    """
    Save a fixed incident to persistent storage.
    
    Args:
        test_id: Unique identifier for the test
        model_name: Name of the dbt model
        column_name: Name of the column
        test_name: Name of the test
        fix_description: Description of the fix applied
        verified: Whether the fix has been verified with dbt test
        backup_paths: Paths to backup files created
        
    Returns:
        Dict with status and saved incident data
    """
    incidents = load_fixed_incidents()
    
    incidents[test_id] = {
        "model_name": model_name,
        "column_name": column_name,
        "test_name": test_name,
        "fix_description": fix_description,
        "fixed_at": datetime.now().isoformat(),
        "verified": verified,
        "backup_paths": backup_paths or []
    }
    
    path = _get_fixed_incidents_path()
    try:
        with open(path, "w") as f:
            json.dump(incidents, f, indent=2)
        return {"status": "success", "data": incidents[test_id]}
    except IOError as e:
        return {"status": "error", "message": str(e)}


def remove_fixed_incident(test_id: str) -> Dict[str, Any]:
    """Remove a fixed incident (e.g., when undoing a fix)."""
    incidents = load_fixed_incidents()
    
    if test_id in incidents:
        removed = incidents.pop(test_id)
        path = _get_fixed_incidents_path()
        with open(path, "w") as f:
            json.dump(incidents, f, indent=2)
        return {"status": "success", "data": removed}
    
    return {"status": "not_found", "message": f"No fixed incident found for {test_id}"}


def is_incident_fixed(test_id: str) -> bool:
    """Check if an incident has been marked as fixed."""
    incidents = load_fixed_incidents()
    return test_id in incidents


# ============================================================
# LLM Helper for Fallback Generation
# ============================================================

def _generate_fix_with_llm(original_content: str, fix_instruction: str, file_type: str = "yaml") -> str:
    """
    Use LLM to generate a fix when deterministic methods fail.
    """
    try:
        settings = get_settings()
        client = genai.Client(api_key=settings.google_api_key)
        
        prompt = f"""
        You are an expert dbt developer. Your task is to apply a specific fix to a dbt {file_type} file.

        ORIGINAL FILE CONTENT:
        ```{file_type}
        {original_content}
        ```

        FIX INSTRUCTION:
        {fix_instruction}

        TASK:
        Apply the fix to the ORIGINAL FILE CONTENT and return the COMPLETE modified file content.
        Do not wrap the output in markdown code blocks. Just return the raw content.
        Do not remove any existing comments or sections unless instructed.
        Maintain the exact indentation style.
        """

        response = client.models.generate_content(
            model=settings.agent.gemini_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                candidate_count=1
            )
        )
        
        if response.text:
            text = response.text.strip()
            # Remove markdown code blocks if present
            if text.startswith("```"):
                lines = text.splitlines()
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                text = "\n".join(lines)
            return text
            
        return original_content
    except Exception as e:
        _fix_logger.error(f"LLM fix generation failed: {e}")
        return original_content


# ============================================================
# ADK Tools for Agentic Code Generation
# ============================================================

def adk_generate_schema_fix(
    model_name: str,
    column_name: str,
    fix_type: str,
    fix_details: str
) -> str:
    """
    Generate a schema.yml fix for a dbt model.
    
    Args:
        model_name: The name of the dbt model to fix.
        column_name: The name of the column to fix.
        fix_type: Type of fix - 'add_accepted_values', 'update_range', 'change_severity', 'add_test', 'remove_test'.
        fix_details: JSON string with fix-specific details (e.g., {"values": ["A", "B"]} for accepted_values).
        
    Returns:
        JSON string with the generated diff and file path, ready to apply.
    """
    import yaml
    
    # Find the schema file
    schema_result = find_schema_file(model_name)
    if schema_result["status"] != "success":
        return json.dumps({
            "status": "error",
            "message": f"Could not find schema file for model {model_name}: {schema_result.get('message', 'Unknown error')}"
        })
    
    file_path = schema_result["data"]["path"]
    relative_path = schema_result["data"]["relative_path"]
    
    # Read current content
    read_result = tool_read_repo(file_path)
    if read_result["status"] != "success":
        return json.dumps({
            "status": "error",
            "message": f"Could not read schema file: {read_result.get('message', 'Unknown error')}"
        })
    
    original_content = read_result["data"]["content"]
    
    # Parse fix details
    try:
        details = json.loads(fix_details) if isinstance(fix_details, str) else fix_details
    except json.JSONDecodeError:
        details = {"raw": fix_details}

    # Heuristic: avoid editing schema.yml if the change looks like a project-level variable
    # For example, a variable named like 'minimum_nights' may be defined in dbt_project.yml
    # If detected, return a guidance note instead of modifying schema.yml directly.
    # We only do this for certain fix types that commonly map to variables (update_range, add_accepted_values)
    potential_variable_names = []
    # common pattern: column_name -> variable like minimum_<column>
    potential_variable_names.append(f"minimum_{column_name}")
    potential_variable_names.append(f"min_{column_name}")
    potential_variable_names.append(column_name)

    for var in potential_variable_names:
        var_res = find_variable_definition(var)
        if var_res.get("status") == "success":
            # Provide guidance instead of making schema change
            return json.dumps({
                "status": "variable_found",
                "message": f"Variable '{var}' is defined in project files. Prefer updating {var_res['data']['relative_path']} instead of schema.yml",
                "variable": var,
                "variable_file": var_res['data']['path'],
                "variable_value": var_res['data'].get('value')
            })
    
    # Apply the fix based on type
    try:
        data = yaml.safe_load(original_content)
        modified = False
        
        for model in data.get("models", []):
            if model.get("name") == model_name:
                for column in model.get("columns", []):
                    if column.get("name") == column_name:
                        if fix_type == "add_accepted_values":
                            # Add values to accepted_values test
                            new_values = details.get("values", [])
                            tests = column.get("tests", [])
                            
                            # Find or create accepted_values test
                            av_test = None
                            for test in tests:
                                if isinstance(test, dict) and "accepted_values" in test:
                                    av_test = test
                                    break
                            
                            if av_test:
                                current_values = av_test["accepted_values"].get("values", [])
                                for val in new_values:
                                    if val not in current_values:
                                        current_values.append(val)
                                av_test["accepted_values"]["values"] = current_values
                            else:
                                # Create new test
                                tests.append({
                                    "accepted_values": {
                                        "values": new_values
                                    }
                                })
                            column["tests"] = tests
                            modified = True
                            
                        elif fix_type == "update_range":
                            # Update range test min/max
                            new_min = details.get("min")
                            new_max = details.get("max")
                            tests = column.get("tests", [])
                            
                            for test in tests:
                                if isinstance(test, dict):
                                    for test_name in test:
                                        # Support standard dbt generic tests and dbt-expectations
                                        is_range_test = test_name in ["accepted_range", "values_in_range"]
                                        is_expect_range = "expect_column_values_to_be_between" in test_name or "expect_column_value_lengths_to_be_between" in test_name
                                        
                                        if is_range_test or is_expect_range:
                                            # Handle regular tests vs arguments based tests
                                            test_config = test[test_name]
                                            
                                            # Helper to update value
                                            def update_val(config, key, val):
                                                if val is None:
                                                    return
                                                if "arguments" in config:
                                                    config["arguments"][key] = val
                                                else:
                                                    config[key] = val

                                            if is_expect_range or "values_in_range" in test_name:
                                                 update_val(test_config, "min_value", new_min)
                                                 update_val(test_config, "max_value", new_max)
                                            else:
                                                 update_val(test_config, "min_value", new_min)
                                                 update_val(test_config, "max_value", new_max)
                                            
                                            modified = True
                                            
                        elif fix_type == "change_severity":
                            # Change test severity to warn
                            severity = details.get("severity", "warn")
                            tests = column.get("tests", [])
                            
                            for i, test in enumerate(tests):
                                if isinstance(test, str):
                                    # Convert simple test to dict with severity
                                    tests[i] = {test: {"severity": severity}}
                                    modified = True
                                elif isinstance(test, dict):
                                    for test_name in test:
                                        if not isinstance(test[test_name], dict):
                                            test[test_name] = {}
                                        test[test_name]["severity"] = severity
                                        modified = True
                            column["tests"] = tests
                            
                        elif fix_type == "remove_test":
                            # Remove a specific test
                            test_to_remove = details.get("test_name", "")
                            tests = column.get("tests", [])
                            column["tests"] = [
                                t for t in tests 
                                if not (t == test_to_remove or 
                                       (isinstance(t, dict) and test_to_remove in t))
                            ]
                            modified = True
                            
                        elif fix_type == "add_where_config":
                            # Add a 'where' config to a test (e.g., to exclude certain rows)
                            where_clause = details.get("where", "")
                            test_name_target = details.get("test_name", "")  # Optional: target specific test
                            tests = column.get("tests", [])
                            
                            for i, test in enumerate(tests):
                                if isinstance(test, str):
                                    # Convert simple test to dict with where config
                                    if not test_name_target or test == test_name_target:
                                        tests[i] = {test: {"where": where_clause}}
                                        modified = True
                                elif isinstance(test, dict):
                                    for tn in test:
                                        if not test_name_target or test_name_target in tn:
                                            if not isinstance(test[tn], dict):
                                                test[tn] = {}
                                            test[tn]["where"] = where_clause
                                            modified = True
                            column["tests"] = tests
        
        if not modified:
            # Fallback to LLM generation
            _fix_logger.info("Deterministic schema fix failed, attempting LLM fallback")
            instruction = f"Apply schema fix of type '{fix_type}' with details: {json.dumps(details)}"
            generated_content = _generate_fix_with_llm(original_content, instruction, "yaml")
            
            if generated_content != original_content:
                # Use the LLM output
                data = yaml.safe_load(generated_content)
                new_content = generated_content
            else:
                return json.dumps({
                    "status": "error",
                    "message": f"Could not apply fix: model/column not found or fix type not applicable (LLM fallback also failed)"
                })
        else:
            # Generate new content from modified data
            new_content = yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)
        
        # Generate diff
        diff_result = get_file_diff(original_content, new_content)
        
        return json.dumps({
            "status": "success",
            "file_path": file_path,
            "relative_path": relative_path,
            "diff": diff_result["data"]["diff"],
            "additions": diff_result["data"]["additions"],
            "deletions": diff_result["data"]["deletions"],
            "original_content": original_content,
            "new_content": new_content,
            "fix_type": fix_type,
            "fix_details": details
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to generate schema fix: {str(e)}"
        })


def adk_generate_sql_fix(
    model_name: str,
    column_name: str,
    fix_type: str,
    fix_code: str
) -> str:
    """
    Generate a SQL model fix for a dbt model.
    
    Args:
        model_name: The name of the dbt model to fix.
        column_name: The name of the column to fix.
        fix_type: Type of fix - 'add_where_clause', 'add_coalesce', 'replace_column', 'add_filter'.
        fix_code: The SQL code to apply (e.g., "WHERE status IS NOT NULL" or "COALESCE(status, 'Unknown')").
        
    Returns:
        JSON string with the generated diff and file path, ready to apply.
    """
    import re
    
    # Find the model file
    file_result = find_file_by_model_name(model_name)
    if file_result["status"] != "success":
        return json.dumps({
            "status": "error",
            "message": f"Could not find SQL file for model {model_name}: {file_result.get('message', 'Unknown error')}"
        })
    
    file_path = file_result["data"]["path"]
    relative_path = file_result["data"]["relative_path"]
    
    # Read current content
    read_result = tool_read_repo(file_path)
    if read_result["status"] != "success":
        return json.dumps({
            "status": "error",
            "message": f"Could not read SQL file: {read_result.get('message', 'Unknown error')}"
        })
    
    original_content = read_result["data"]["content"]
    new_content = original_content
    
    try:
        if fix_type == "add_where_clause":
            # Add or extend WHERE clause
            condition = fix_code.strip()
            if condition.upper().startswith("WHERE"):
                condition = condition[5:].strip()
            
            # Check if there's already a WHERE clause
            where_pattern = re.compile(r'\bWHERE\b', re.IGNORECASE)
            where_match = where_pattern.search(new_content)
            
            if where_match:
                # Add to existing WHERE with AND
                insert_pos = where_match.end()
                new_content = (
                    new_content[:insert_pos] + 
                    f" {condition} AND" + 
                    new_content[insert_pos:]
                )
            else:
                # Find the end of the main query (before any final semicolon or ORDER BY/GROUP BY/LIMIT)
                # Simple approach: insert before the last line that's not empty
                lines = new_content.rstrip().split('\n')
                insert_idx = len(lines)
                for i in range(len(lines) - 1, -1, -1):
                    line_upper = lines[i].strip().upper()
                    if line_upper and not line_upper.startswith(('ORDER BY', 'GROUP BY', 'LIMIT', ';', '--', '/*')):
                        insert_idx = i + 1
                        break
                
                lines.insert(insert_idx, f"WHERE {condition}")
                new_content = '\n'.join(lines)
                
        elif fix_type == "add_coalesce":
            # Wrap column with COALESCE
            coalesce_expr = fix_code.strip()
            
            # Find the column in SELECT and wrap it
            # This is a simplified approach - works for simple cases
            column_pattern = re.compile(
                rf'\b({re.escape(column_name)})\b(?!\s*\()',  # Column not followed by (
                re.IGNORECASE
            )
            
            # Replace in SELECT clause only (simplification)
            new_content = column_pattern.sub(coalesce_expr, new_content, count=1)
            
        elif fix_type == "replace_column":
            # Replace column definition entirely
            new_definition = fix_code.strip()
            
            # Pattern 1: Column preceded by comma (e.g. ", column_name")
            p1 = re.compile(rf'(,\s*)\b{re.escape(column_name)}\b', re.IGNORECASE)
            
            # Pattern 2: Column followed by comma (e.g. "column_name,"). 
            p2 = re.compile(rf'\b{re.escape(column_name)}\b(\s*,)', re.IGNORECASE)
            
            # Pattern 3: Column with Alias preceded by comma
            p3 = re.compile(rf'(,\s*)\b(?:(?!\bAS\b)[^,])+\s+AS\s+{re.escape(column_name)}\b', re.IGNORECASE)

            # Pattern 4: Column with Alias followed by comma
            p4 = re.compile(rf'\b(?:(?!\bAS\b)[^,])+\s+AS\s+{re.escape(column_name)}\b(\s*,)', re.IGNORECASE)

            if p1.search(new_content):
                new_content = p1.sub(rf'\g<1>{new_definition}', new_content, count=1)
            elif p3.search(new_content):
                new_content = p3.sub(rf'\g<1>{new_definition}', new_content, count=1)
            elif p2.search(new_content):
                new_content = p2.sub(rf'{new_definition}\g<1>', new_content, count=1)
            elif p4.search(new_content):
                new_content = p4.sub(rf'{new_definition}\g<1>', new_content, count=1)
            else:
                # Basic fallback: try replacing just the column name if unique enough
                # Note: this is risky if column name appears elsewhere (e.g. WHERE clause) but count=1 mitigates
                # Only use if we are somewhat confident it's the definition
                 p5 = re.compile(rf'\b{re.escape(column_name)}\b', re.IGNORECASE)
                 if p5.search(new_content):
                     # Check if it looks like it's in a select list (surrounded by newlines or commas nearby)
                     # For now, let's skip fallback to avoid corrupting WHERE clauses
                     pass
                    
        elif fix_type == "add_filter":
            # Generic filter - same as add_where_clause
            return adk_generate_sql_fix(model_name, column_name, "add_where_clause", fix_code)
        
        if new_content == original_content:
            # Fallback to LLM generation
            _fix_logger.info("Deterministic SQL fix failed, attempting LLM fallback")
            instruction = f"Apply SQL fix of type '{fix_type}' for column '{column_name}'. detailed code/instruction: {fix_code}"
            
            generated_content = _generate_fix_with_llm(original_content, instruction, "sql")
            
            if generated_content != original_content:
                new_content = generated_content
            else:
                return json.dumps({
                    "status": "error",
                    "message": "Could not apply SQL fix: no changes made. The fix pattern may not match the code structure (LLM fallback also failed)."
                })
        
        # Generate diff
        diff_result = get_file_diff(original_content, new_content)
        
        return json.dumps({
            "status": "success",
            "file_path": file_path,
            "relative_path": relative_path,
            "diff": diff_result["data"]["diff"],
            "additions": diff_result["data"]["additions"],
            "deletions": diff_result["data"]["deletions"],
            "original_content": original_content,
            "new_content": new_content,
            "fix_type": fix_type,
            "fix_code": fix_code
        }, indent=2)
        
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to generate SQL fix: {str(e)}"
        })


def adk_apply_fix(
    file_path: str,
    new_content: str,
    fix_description: str
) -> str:
    """
    Apply a code fix to a file with automatic backup.
    
    Args:
        file_path: Absolute path to the file to modify.
        new_content: The new content to write to the file.
        fix_description: A brief description of the fix being applied.
        
    Returns:
        JSON string with the result including backup path.
    """
    # Write with backup
    result = tool_write_repo(file_path, new_content)
    
    if result["status"] == "success":
        return json.dumps({
            "status": "success",
            "message": f"Applied fix: {fix_description}",
            "file_path": file_path,
            "backup_path": result.get("data", {}).get("backup_path"),
            "fix_description": fix_description
        }, indent=2)
    else:
        return json.dumps({
            "status": "error",
            "message": f"Failed to apply fix: {result.get('message', 'Unknown error')}"
        })


def adk_undo_fix(backup_path: str) -> str:
    """
    Undo a fix by restoring from backup.
    
    Args:
        backup_path: Path to the backup file created during apply.
        
    Returns:
        JSON string with the result.
    """
    import shutil
    
    if not os.path.exists(backup_path):
        return json.dumps({
            "status": "error",
            "message": f"Backup file not found: {backup_path}"
        })
    
    # Extract original path from backup path (remove .bak.TIMESTAMP)
    original_path = backup_path
    if ".bak." in backup_path:
        original_path = backup_path.rsplit(".bak.", 1)[0]
    
    try:
        shutil.copy2(backup_path, original_path)
        return json.dumps({
            "status": "success",
            "message": f"Restored file from backup",
            "restored_path": original_path,
            "backup_path": backup_path
        }, indent=2)
    except Exception as e:
        return json.dumps({
            "status": "error",
            "message": f"Failed to restore from backup: {str(e)}"
        })


# ============================================================
# Fix Proposal Tool (Main Agentic Tool)
# ============================================================

import logging as _logging
_fix_logger = _logging.getLogger(__name__)

def adk_propose_fix(
    test_name: str,
    model_name: str,
    column_name: str,
    root_cause: str,
    fix_options: str
) -> str:
    """
    Propose fix options for a test failure based on investigation and business context.
    
    This is the main agentic tool for generating fixes. The agent should call this
    after completing its investigation to formalize its fix recommendations.
    The proposed fixes should be specific to the root cause and evidence found, as well as aligned with the business context.
    The rationale should mention clearly the pros and cons of each fix, and WHEN and WHY each fix is appropriate.
    
    Args:
        test_name: Name of the failing test.
        model_name: Name of the dbt model.
        column_name: Name of the column (if applicable).
        root_cause: The diagnosed root cause of the failure.
        fix_options: JSON string containing an array of fix options, each with:
            - id: Unique identifier (e.g., "fix_1")
            - title: Clear action title
            - description: What this fix does
            - fix_type: One of 'schema' or 'sql'
            - code_change: Object with fix details:
                - For schema: {"action": "add_accepted_values|update_range|change_severity|...", "details": {...}}
                - For sql: {"action": "add_where_clause|add_coalesce|replace_column|...", "code": "..."}
            - rationale: Why this fix works
            - pros: Key advantages or strengths of this option
            - cons: Key tradeoffs, limitations, or risks of this option
            - when_appropriate: When this option is the best choice (conditions, scenarios, or constraints)
            
    Returns:
        JSON string with validated fix options ready for UI display.
    """
    _fix_logger.debug(f"adk_propose_fix called: test_name={test_name}, model_name={model_name}")
    
    try:
        options = json.loads(fix_options) if isinstance(fix_options, str) else fix_options
    except json.JSONDecodeError as e:
        _fix_logger.error(f"JSON decode error: {e}")
        return json.dumps({
            "status": "error",
            "message": f"Invalid fix_options JSON: {str(e)}"
        })
    
    # Ensure options is a list
    if options is None:
        return json.dumps({
            "status": "error",
            "message": "fix_options parsed to None - expected a JSON array"
        })
    
    if not isinstance(options, list):
        # Try to extract from common wrapper structures
        if isinstance(options, dict):
            # Maybe it's wrapped in an object like {"fix_options": [...]} or {"options": [...]}
            options = options.get("fix_options") or options.get("options") or []
        else:
            return json.dumps({
                "status": "error",
                "message": f"fix_options must be a JSON array, got {type(options).__name__}"
            })
    
    validated_options = []
    skipped_options = []
    
    # Fix types that don't require code changes
    NON_CODE_FIX_TYPES = {"snooze", "manual_review", "alert_suppression", "ignore"}
    
    for opt in options:
        # Validate required fields
        if not opt.get("title") or not opt.get("fix_type"):
            skipped_options.append({"title": opt.get("title", "Unknown"), "reason": "Missing title or fix_type"})
            continue
        
        fix_type = opt.get("fix_type", "")
        code_change = opt.get("code_change", {})
        
        # Check if code_change is required for this fix type
        requires_code = fix_type not in NON_CODE_FIX_TYPES
        
        if requires_code and not code_change:
            skipped_options.append({"title": opt.get("title"), "reason": "Missing code_change (required for actionable fixes)"})
            _fix_logger.warning(f"Skipping fix option '{opt.get('title')}': missing code_change")
            continue
        
        if requires_code and not (code_change.get("action") or code_change.get("code")):
            skipped_options.append({"title": opt.get("title"), "reason": "code_change missing 'action' or 'code'"})
            _fix_logger.warning(f"Skipping fix option '{opt.get('title')}': code_change has no action or code")
            continue
        
        validated_opt = {
            "id": opt.get("id") or str(uuid.uuid4()),
            "title": opt.get("title"),
            "description": opt.get("description", ""),
            "fix_type": fix_type,
            "rationale": opt.get("rationale", ""),
            "pros": opt.get("pros", ""),
            "cons": opt.get("cons", ""),
            "when_appropriate": opt.get("when_appropriate", ""),
            "ai_technical_reason": opt.get("rationale", ""),
            "code_change": code_change,
            "root_cause": root_cause
        }
        
        # Pre-generate the diff if we have code_change details
        code_generated = False
        # Determine recommended files and baseline confidence
        recommended_file = None
        confidence = 0.5

        try:
            file_list_res = list_model_files(model_name)
            if file_list_res.get('status') == 'success':
                # Prefer schema files for schema fixes, SQL for sql fixes
                for f in file_list_res['data']:
                    if fix_type == 'schema' and f.get('type') == 'schema':
                        recommended_file = f.get('path')
                        confidence = 0.8
                        break
                    if fix_type == 'sql' and f.get('type') == 'sql':
                        recommended_file = f.get('path')
                        confidence = 0.8
                        break
                # If no direct match, pick the first file
                if not recommended_file and file_list_res['data']:
                    recommended_file = file_list_res['data'][0].get('path')
        except Exception:
            pass
        
        if code_change and fix_type == "schema":
            action = code_change.get("action", "")
            details = code_change.get("details", {})
            diff_result = adk_generate_schema_fix(
                model_name, column_name, action, json.dumps(details)
            )
            diff_data = json.loads(diff_result)
            if diff_data.get("status") == "success":
                validated_opt["code_changes"] = {
                    diff_data["file_path"]: diff_data["new_content"]
                }
                validated_opt["ai_code_snippet"] = diff_data.get("diff", "")
                validated_opt["diff_preview"] = diff_data.get("diff", "")
                validated_opt["recommended_file"] = recommended_file or diff_data.get("file_path")
                validated_opt["confidence"] = confidence
                code_generated = True
                
            elif diff_data.get("status") == "variable_found":
                # Handle variable update instead of schema change
                variable_name = diff_data.get("variable")
                variable_file = diff_data.get("variable_file")
                
                _fix_logger.info(f"Variable '{variable_name}' found in {variable_file}, attempting to update it instead of schema.yml")
                
                # Infer new value from details
                new_value = None
                if "min" in variable_name or "minimum" in variable_name:
                    new_value = details.get("min")
                elif "max" in variable_name or "maximum" in variable_name:
                    new_value = details.get("max")
                
                if new_value is not None:
                    # Attempt to update the variable file using LLM
                    var_read = tool_read_repo(variable_file)
                    if var_read["status"] == "success":
                        var_content = var_read["data"]["content"]
                        instruction = f"Update the value of variable '{variable_name}' to {new_value}. This variable is likely under 'vars:' section. Do NOT change anything else."
                        
                        generated_var_content = _generate_fix_with_llm(var_content, instruction, "yaml")
                        
                        if generated_var_content != var_content:
                            diff_res = get_file_diff(var_content, generated_var_content)
                            
                            validated_opt["code_changes"] = {
                                variable_file: generated_var_content
                            }
                            validated_opt["ai_code_snippet"] = diff_res["data"]["diff"]
                            validated_opt["diff_preview"] = diff_res["data"]["diff"]
                            validated_opt["recommended_file"] = variable_file
                            # High confidence for variable updates
                            validated_opt["confidence"] = 0.95
                            
                            # OVERWRITE the LLM-generated title/desc to reflect the actual action
                            validated_opt["title"] = f"Update variable '{variable_name}'"
                            validated_opt["description"] = f"Update project variable '{variable_name}' to {new_value} in {os.path.basename(variable_file)}."
                            validated_opt["rationale"] = f"Detected that '{column_name}' range is controlled by variable '{variable_name}'. Best practice is to update the variable rather than the schema."
                             
                            code_generated = True
                
                if not code_generated:
                    # Fallback if variable update failed
                    validated_opt["ai_code_snippet"] = f"# Variable '{variable_name}' found in {variable_file}.\n# Please update it manually to: {json.dumps(details)}"
                    validated_opt["requires_manual_apply"] = True
                    
            else:
                _fix_logger.warning(f"Schema fix generation failed for '{opt.get('title')}': {diff_data.get('message')}")
                # Fallback for Schema
                validated_opt["ai_code_snippet"] = f"# Auto-fix generation failed. Proposed config:\n{json.dumps(details, indent=2)}"
                validated_opt["requires_manual_apply"] = True
                
        elif code_change and fix_type == "sql":
            action = code_change.get("action", "")
            code = code_change.get("code", "")
            diff_result = adk_generate_sql_fix(
                model_name, column_name, action, code
            )
            diff_data = json.loads(diff_result)
            if diff_data.get("status") == "success":
                validated_opt["code_changes"] = {
                    diff_data["file_path"]: diff_data["new_content"]
                }
                validated_opt["ai_code_snippet"] = diff_data.get("diff", "")
                validated_opt["diff_preview"] = diff_data.get("diff", "")
                validated_opt["recommended_file"] = recommended_file or diff_data.get("file_path")
                validated_opt["confidence"] = confidence
                code_generated = True
            else:
                _fix_logger.warning(f"SQL fix generation failed for '{opt.get('title')}': {diff_data.get('message')}")
                # Fallback for SQL
                validated_opt["ai_code_snippet"] = f"-- Auto-fix generation failed. Proposed change:\n{code}"
                validated_opt["requires_manual_apply"] = True
        
        # Include the option even if code generation failed, so the user can see the suggestion
        validated_options.append(validated_opt)
        if not code_generated and requires_code:
             _fix_logger.info(f"Included '{opt.get('title')}' as manual/failed fix")
    
    # Log summary
    if skipped_options:
        _fix_logger.info(f"Skipped {len(skipped_options)} options: {[s['title'] for s in skipped_options]}")
    
    return json.dumps({
        "status": "success",
        "message": f"Generated {len(validated_options)} fix option(s)" + (f" ({len(skipped_options)} skipped)" if skipped_options else ""),
        "data": {
            "test_name": test_name,
            "model_name": model_name,
            "column_name": column_name,
            "root_cause": root_cause,
            "options": validated_options,
            "skipped_options": skipped_options if skipped_options else None
        }
    }, indent=2)
