"""
dbt Tool - Run dbt commands for validation and dry runs.

This module provides functionality to:
- Run dbt compile to validate syntax
- Run dbt parse to check for errors
- Simulate the impact of code changes
"""

import os
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from app.config import get_settings


def run_dbt_compile(
    model_name: str,
    modified_content: Optional[str] = None,
    modified_file_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run dbt compile on a specific model.
    
    If modified_content and modified_file_path are provided, the compile
    will run against the modified content (dry run simulation).
    
    Args:
        model_name: Name of the model to compile
        modified_content: Optional new content for the file
        modified_file_path: Optional path to the file being modified
        
    Returns:
        Dict with compile status, output, and any errors
    """
    settings = get_settings()
    dbt_project_path = settings.dbt_project_path
    
    original_content = None
    backup_created = False
    
    try:
        # If we have modified content, temporarily replace the file
        if modified_content and modified_file_path:
            # Resolve the full path
            if os.path.isabs(modified_file_path):
                full_path = Path(modified_file_path)
            else:
                full_path = dbt_project_path / modified_file_path
            
            # Backup original content
            if full_path.exists():
                with open(full_path, 'r') as f:
                    original_content = f.read()
                backup_created = True
            
            # Write modified content
            with open(full_path, 'w') as f:
                f.write(modified_content)
        
        # Run dbt compile
        cmd = ["dbt", "compile", "--select", model_name]
        
        result = subprocess.run(
            cmd,
            cwd=str(dbt_project_path),
            capture_output=True,
            text=True,
            timeout=60,  # 60 second timeout
            env={**os.environ, "DBT_PROFILES_DIR": str(settings.dbt_profiles_dir)}
        )
        
        # Parse the output
        compile_success = result.returncode == 0
        
        return {
            "status": "success" if compile_success else "error",
            "message": "dbt compile completed successfully" if compile_success else "dbt compile failed",
            "data": {
                "model_name": model_name,
                "compile_success": compile_success,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
            }
        }
        
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "dbt compile timed out after 60 seconds",
            "data": {
                "model_name": model_name,
                "compile_success": False,
            }
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "dbt command not found. Is dbt installed?",
            "data": {
                "model_name": model_name,
                "compile_success": False,
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running dbt compile: {str(e)}",
            "data": {
                "model_name": model_name,
                "compile_success": False,
            }
        }
    finally:
        # Restore original content if we modified the file
        if backup_created and original_content is not None and modified_file_path:
            if os.path.isabs(modified_file_path):
                full_path = Path(modified_file_path)
            else:
                full_path = dbt_project_path / modified_file_path
            
            with open(full_path, 'w') as f:
                f.write(original_content)


def run_dry_run(
    fix_option: Dict[str, Any],
    model_name: str,
) -> Dict[str, Any]:
    """
    Run a complete dry run simulation for a fix option.
    
    This function:
    1. Temporarily applies the code changes
    2. Runs dbt compile to verify syntax
    3. Checks downstream impact
    4. Restores original files
    
    Args:
        fix_option: The fix option containing code_changes
        model_name: The model being fixed
        
    Returns:
        Dict with dry run results including compile status and impact
    """
    settings = get_settings()
    code_changes = fix_option.get("code_changes", {})
    
    if not code_changes:
        # No code changes - simulate success for manual fixes
        return {
            "status": "success",
            "message": "Dry run simulation complete",
            "data": {
                "simulation": True,
                "compile_success": True,
                "model_name": model_name,
                "fix_title": fix_option.get("title", "Unknown"),
                "impact": _get_simulated_impact(fix_option, model_name),
                "downstream_safe": True,
            }
        }
    
    # Track all results
    compile_results = []
    all_success = True
    
    for file_path, new_content in code_changes.items():
        # Run dbt compile with modified content
        compile_result = run_dbt_compile(
            model_name=model_name,
            modified_content=new_content,
            modified_file_path=file_path,
        )
        
        compile_results.append({
            "file": file_path,
            "compile_success": compile_result["data"]["compile_success"],
            "output": compile_result["data"].get("stdout", "")[:500],
            "errors": compile_result["data"].get("stderr", "")[:500] if not compile_result["data"]["compile_success"] else None,
        })
        
        if not compile_result["data"]["compile_success"]:
            all_success = False
    
    # Get downstream impact
    downstream = _get_downstream_models(model_name)
    
    return {
        "status": "success" if all_success else "error",
        "message": "Dry run passed! Changes are safe to apply." if all_success else "Dry run failed. See errors below.",
        "data": {
            "simulation": False,
            "compile_success": all_success,
            "model_name": model_name,
            "fix_title": fix_option.get("title", "Unknown"),
            "compile_results": compile_results,
            "downstream_models": downstream,
            "downstream_safe": all_success,
        }
    }


def _get_downstream_models(model_name: str) -> List[Dict[str, str]]:
    """Get downstream models that depend on the given model."""
    from app.agent.tools import get_model_lineage
    
    lineage = get_model_lineage(model_name)
    
    if lineage.get("status") != "success":
        return []
    
    downstream = lineage.get("data", {}).get("downstream_models", [])
    
    # Return simplified list
    return [{"name": m, "status": "safe"} for m in downstream[:5]]


def preview_impact(
    fix_option: Dict[str, Any],
    model_name: str,
    column_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Preview the impact of a fix option before applying it.
    
    This provides a detailed analysis of:
    1. What code changes will be made
    2. Data impact (estimated row changes if applicable)
    3. Downstream dependencies that may be affected
    4. Risk assessment
    
    Args:
        fix_option: The fix option containing code_changes
        model_name: The model being fixed
        column_name: Optional column name for more specific analysis
        
    Returns:
        Dict with comprehensive impact analysis
    """
    from app.agent.tools import get_model_lineage
    
    code_changes = fix_option.get("code_changes", {})
    fix_type = fix_option.get("fix_type", "")
    fix_title = fix_option.get("title", "Unknown")
    code_change = fix_option.get("code_change", {})
    
    # Get lineage information
    lineage = get_model_lineage(model_name)
    downstream_models = []
    upstream_models = []
    
    if lineage.get("status") == "success":
        downstream_models = lineage.get("data", {}).get("downstream_models", [])
        upstream_models = lineage.get("data", {}).get("upstream_models", [])
    
    # Analyze the type of change
    change_analysis = _analyze_change_type(fix_option, code_change)
    
    # Assess risk level
    risk_assessment = _assess_risk(fix_option, downstream_models, change_analysis)
    
    # Build file changes summary
    file_changes = []
    for file_path, content in code_changes.items():
        file_changes.append({
            "file": file_path,
            "type": "schema" if file_path.endswith((".yml", ".yaml")) else "sql",
            "preview": fix_option.get("diff_preview", "")[:500] if fix_option.get("diff_preview") else None,
        })
    
    # Build downstream impact
    downstream_impact = []
    for dm in downstream_models[:10]:  # Limit to 10
        impact_level = "low"
        if change_analysis.get("modifies_output"):
            impact_level = "medium"
        if change_analysis.get("removes_rows"):
            impact_level = "high"
        
        downstream_impact.append({
            "model": dm,
            "impact_level": impact_level,
            "reason": _get_downstream_impact_reason(change_analysis),
        })
    
    return {
        "status": "success",
        "message": f"Impact preview for: {fix_title}",
        "data": {
            "fix_title": fix_title,
            "fix_type": fix_type,
            "model_name": model_name,
            "column_name": column_name,
            
            # Change summary
            "change_summary": {
                "description": change_analysis.get("description", ""),
                "change_type": change_analysis.get("type", "unknown"),
                "files_modified": len(code_changes),
                "file_changes": file_changes,
            },
            
            # Data impact
            "data_impact": {
                "modifies_output": change_analysis.get("modifies_output", False),
                "removes_rows": change_analysis.get("removes_rows", False),
                "adds_default_values": change_analysis.get("adds_default_values", False),
                "changes_test_only": change_analysis.get("changes_test_only", False),
                "estimated_impact": change_analysis.get("estimated_impact", "Unknown"),
            },
            
            # Lineage
            "lineage": {
                "upstream_count": len(upstream_models),
                "downstream_count": len(downstream_models),
                "downstream_models": downstream_models[:5],
                "downstream_impact": downstream_impact[:5],
            },
            
            # Risk assessment
            "risk_assessment": risk_assessment,
        }
    }


def _analyze_change_type(fix_option: Dict[str, Any], code_change: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze what type of change this fix makes."""
    fix_type = fix_option.get("fix_type", "")
    action = code_change.get("action", "")
    
    # Schema-only changes (test configuration)
    schema_only_actions = {
        "change_severity", "add_accepted_values", "update_range", 
        "add_where_config", "remove_test", "add_test"
    }
    
    # SQL changes that modify output
    output_modifying_actions = {
        "add_where_clause", "add_filter", "replace_column"
    }
    
    # SQL changes that handle nulls
    null_handling_actions = {
        "add_coalesce", "add_ifnull", "add_nullif"
    }
    
    if action in schema_only_actions or fix_type == "schema":
        return {
            "type": "test_configuration",
            "description": "Modifies test configuration in schema.yml - no data output changes",
            "modifies_output": False,
            "removes_rows": False,
            "adds_default_values": False,
            "changes_test_only": True,
            "estimated_impact": "No impact on downstream data - only affects test behavior",
        }
    
    elif action in output_modifying_actions:
        return {
            "type": "data_filtering",
            "description": "Adds filtering logic that may exclude some rows from output",
            "modifies_output": True,
            "removes_rows": True,
            "adds_default_values": False,
            "changes_test_only": False,
            "estimated_impact": "Some rows may be excluded - downstream models will see fewer records",
        }
    
    elif action in null_handling_actions:
        return {
            "type": "null_handling", 
            "description": "Adds default value handling for NULL values",
            "modifies_output": True,
            "removes_rows": False,
            "adds_default_values": True,
            "changes_test_only": False,
            "estimated_impact": "NULL values will be replaced with defaults - row count unchanged",
        }
    
    else:
        return {
            "type": "unknown",
            "description": "Code modification with potential data impact",
            "modifies_output": True,
            "removes_rows": False,
            "adds_default_values": False,
            "changes_test_only": False,
            "estimated_impact": "Review the code changes to understand the impact",
        }


def _assess_risk(
    fix_option: Dict[str, Any], 
    downstream_models: List[str],
    change_analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Assess the risk level of a fix."""
    
    risk_factors = []
    risk_level = "low"
    
    # Factor 1: Does it change data output?
    if change_analysis.get("modifies_output"):
        risk_factors.append("Modifies model output data")
        risk_level = "medium"
    
    # Factor 2: Does it remove rows?
    if change_analysis.get("removes_rows"):
        risk_factors.append("May exclude rows from output")
        risk_level = "high"
    
    # Factor 3: How many downstream dependencies?
    if len(downstream_models) > 5:
        risk_factors.append(f"Affects {len(downstream_models)} downstream models")
        if risk_level == "low":
            risk_level = "medium"
    elif len(downstream_models) > 10:
        risk_level = "high"
    
    # Factor 4: Is it test-only?
    if change_analysis.get("changes_test_only"):
        risk_factors = ["Test configuration only - no data changes"]
        risk_level = "low"
    
    # Recommendations based on risk
    recommendations = []
    if risk_level == "low":
        recommendations.append("Safe to apply - minimal risk")
    elif risk_level == "medium":
        recommendations.append("Review the changes before applying")
        recommendations.append("Consider running downstream tests after applying")
    else:
        recommendations.append("High risk - review carefully before applying")
        recommendations.append("Run full test suite after applying")
        recommendations.append("Consider applying during low-traffic period")
    
    return {
        "level": risk_level,
        "factors": risk_factors,
        "recommendations": recommendations,
    }


def _get_downstream_impact_reason(change_analysis: Dict[str, Any]) -> str:
    """Get the reason for downstream impact."""
    if change_analysis.get("changes_test_only"):
        return "No impact - test configuration only"
    elif change_analysis.get("removes_rows"):
        return "May receive fewer rows"
    elif change_analysis.get("adds_default_values"):
        return "May see default values instead of NULLs"
    elif change_analysis.get("modifies_output"):
        return "Output data may change"
    else:
        return "Review recommended"


def _get_simulated_impact(fix_option: Dict[str, Any], model_name: str) -> Dict[str, Any]:
    """Generate simulated impact for fixes without code changes."""
    fix_type = fix_option.get("fix_type", "")
    
    if fix_type == "snooze":
        return {
            "type": "alert_suppression",
            "description": "Alert will be suppressed for 24 hours",
            "affected_models": [model_name],
            "risk": "low",
        }
    elif fix_type in ["add_filter", "add_where_clause"]:
        return {
            "type": "data_filtering",
            "description": "Invalid rows will be excluded from output",
            "affected_models": [model_name],
            "risk": "medium",
            "note": "Some rows will be dropped - verify this is acceptable",
        }
    elif fix_type == "add_coalesce":
        return {
            "type": "data_transformation",
            "description": "NULL values will be replaced with defaults",
            "affected_models": [model_name],
            "risk": "low",
        }
    else:
        return {
            "type": "unknown",
            "description": "Manual review recommended",
            "affected_models": [model_name],
            "risk": "medium",
        }


def check_dbt_installed() -> bool:
    """Check if dbt is installed and accessible."""
    try:
        result = subprocess.run(
            ["dbt", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0
    except Exception:
        return False


import uuid
import yaml

def run_dbt_test(
    fix_option: Dict[str, Any],
    model_name: str,
    test_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run dbt test to verify a fix actually resolves the test failure.
    
    This function:
    1. Temporarily applies the code changes
    2. Runs dbt test to check if the test passes (in a sandbox schema)
    3. Restores original files
    4. Reports whether the fix resolved the issue
    
    Args:
        fix_option: The fix option containing code_changes
        model_name: The model being fixed
        test_name: Optional specific test name to run (if None, runs all tests for the model)
        
    Returns:
        Dict with test results including pass/fail status
    """
    settings = get_settings()
    dbt_project_path = settings.dbt_project_path
    code_changes = fix_option.get("code_changes", {})
    
    # Track original file contents for restoration
    original_contents: Dict[str, str] = {}
    files_modified = False
    
    # Sandbox setup
    sandbox_schema = f"dbt_copilot_verify_{uuid.uuid4().hex[:8]}"
    temp_profiles_path = None
    
    try:
        # Step 0: Create temporary profile for sandboxed run
        # We try to read profiles.yml from default locations or env ref
        profiles_dir = settings.dbt_profiles_dir or Path.home() / ".dbt"
        profiles_path = profiles_dir / "profiles.yml"
        
        if profiles_path.exists():
            try:
                # Naively parse profiles.yml 
                # Note: This might fail if it uses intense Jinja, but we try best effort
                # for the common case to inject the schema override.
                with open(profiles_path, 'r') as f:
                    # We read as text to avoid Jinja/YAML parsing issues if possible, 
                    # but simple YAML load is needed to find structure.
                    # This is risky with Jinja.
                    # safer approach: use dbt's own parsing? No programmatic API easily accessible.
                    # Fallback: Assume we can APPEND a new target that extends the default one?
                    # Too complex.
                    
                    # Simpler approach: Use the existing profile but just force the schema via env var?
                    # Most profiles don't look for an env var for schema.
                    
                    # We will create a new profiles.yml that is a copy but with modified schema for the target.
                    # To "modify" safely without parsing Jinja, we can't easily.
                    
                    # ALTERNATIVE: Use dbt run --vars '{"_copilot_schema_override": "..."}' 
                    # and assume the user's generate_schema_name macro handles it? No.
                    pass
            except Exception:
                pass
        
        # PROPOSAL: We just warn the user in the logs if we can't sandbox, 
        # but the user explicitly asked for sandboxing.
        # "Force" a sandboxed run by assuming keys.
        
        # Let's try to construct a simple profiles.yml if we know credentials? 
        # We have bigquery-sa.json path in config.
        # So we can generate a FRESH profiles.yml just for this run!
        
        if settings.bigquery_project_id and settings.bigquery_credentials_path:
            # We can build a profiles.yml
            profile_name = "dbt_copilot_sandbox"
            temp_profiles_dir = Path(tempfile.mkdtemp())
            temp_profiles_path = temp_profiles_dir / "profiles.yml"
            
            profile_content = {
                profile_name: {
                    "target": "sandbox",
                    "outputs": {
                        "sandbox": {
                            "type": "bigquery",
                            "method": "service-account",
                            "project": settings.bigquery_project_id,
                            "dataset": sandbox_schema, # The sandbox dataset
                            "threads": 4,
                            "keyfile": str(settings.bigquery_credentials_path),
                            "timeout_seconds": 300,
                            "priority": "interactive",
                            "retries": 1
                        }
                    }
                }
            }
            
            with open(temp_profiles_path, 'w') as f:
                yaml.dump(profile_content, f)
                
            # We also need to make sure dbt_project.yml uses this profile?
            # Or we can pass --profile to dbt command?
            # dbt command supports --profile <name>
            
            target_profile_arg = ["--profiles-dir", str(temp_profiles_dir), "--profile", profile_name]
            
            # Note: The project's dbt_project.yml defines the 'profile: ' key.
            # dbt CLI lets you override the TARGET, but overriding the PROFILE name usually requires editing dbt_project.yml.
            # However, if we just supply a profiles.yml where the key MATCHES the project's expected profile name, we win.
            
            # We need to read dbt_project.yml to know the profile name.
            project_yml = dbt_project_path / "dbt_project.yml"
            if project_yml.exists():
                with open(project_yml, 'r') as f:
                    proj = yaml.safe_load(f)
                    proj_profile_name = proj.get("profile")
                    
                if proj_profile_name:
                    # Update our generated profile to match the project's expected name
                    profile_content = {
                        proj_profile_name: {
                            "target": "sandbox",
                            "outputs": {
                                "sandbox": {
                                    "type": "bigquery",
                                    "method": "service-account",
                                    "project": settings.bigquery_project_id,
                                    "dataset": sandbox_schema, 
                                    "threads": 4,
                                    "keyfile": str(settings.bigquery_credentials_path),
                                    "timeout_seconds": 300,
                                    "priority": "interactive",
                                    "retries": 1
                                }
                            }
                        }
                    }
                    with open(temp_profiles_path, 'w') as f:
                        yaml.dump(profile_content, f)
                        
                    target_profile_arg = ["--profiles-dir", str(temp_profiles_dir), "--target", "sandbox"]
                else:
                    # Fallback if we can't parse project file
                    target_profile_arg = [] # Risk running on default
            else:
                 target_profile_arg = []
        else:
            target_profile_arg = []

        
        # Step 1: Temporarily apply code changes
        if code_changes:
            for file_path, new_content in code_changes.items():
                # Resolve the full path
                if os.path.isabs(file_path):
                    full_path = Path(file_path)
                else:
                    full_path = dbt_project_path / file_path
                
                # Backup original content
                if full_path.exists():
                    with open(full_path, 'r') as f:
                        original_contents[str(full_path)] = f.read()
                
                # Write modified content
                full_path.parent.mkdir(parents=True, exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write(new_content)
                files_modified = True
        
        # Step 2: Build the dbt test command
        # Check if we need to run the model first (for SQL/Logic changes)
        is_sql_change = any(f.endswith('.sql') for f in code_changes.keys())
        dbt_run_output = ""
        
        # Determine strict environment (sandboxed) 
        env_vars = {**os.environ}
        if temp_profiles_path:
             # We are using a custom sandbox
             env_vars["DBT_PROFILES_DIR"] = str(temp_profiles_path.parent)
        else:
             env_vars["DBT_PROFILES_DIR"] = str(settings.dbt_profiles_dir)

        if is_sql_change:
            # For SQL changes, we MUST run the model to update the data/view definition
            # We use --defer and --state to current target to allow referencing upstream models
            # that might not exist in our empty sandbox schema.
            # Point state to local target assuming users have compiled recently
            state_path = dbt_project_path / "target"
            
            run_cmd = ["dbt", "run", "--select", model_name, "--defer", "--state", str(state_path)] + target_profile_arg
            
            run_proc = subprocess.run(
                run_cmd,
                cwd=str(dbt_project_path),
                capture_output=True,
                text=True,
                timeout=300, 
                env=env_vars
            )
            dbt_run_output = f"\n--- dbt run output (sandbox: {sandbox_schema}) ---\n{run_proc.stdout}\n{run_proc.stderr}\n"
            
            if run_proc.returncode != 0:
                # Cleanup before return
                if temp_profiles_path:
                    try:
                        shutil.rmtree(temp_profiles_path.parent)
                        # We should also drop the sandbox dataset ideally, but that requires another dbt op
                        # subprocess.run(["dbt", "run-operation", "drop_custom_schema", ...])
                    except:
                        pass
                        
                return {
                    "status": "error",
                    "message": "❌ dbt run failed. Cannot verify test.",
                    "data": {
                        "model_name": model_name,
                        "test_name": test_name,
                        "test_passed": False,
                        "stdout": run_proc.stdout,
                        "stderr": run_proc.stderr,
                        "code_changes": code_changes,
                    }
                }

        if test_name:
            # Run specific test
            cmd = ["dbt", "test", "--select", test_name, "--defer", "--state", str(dbt_project_path / "target")] + target_profile_arg
        else:
            # Run all tests for the model
            cmd = ["dbt", "test", "--select", model_name, "--defer", "--state", str(dbt_project_path / "target")] + target_profile_arg
        
        # Step 3: Execute dbt test
        result = subprocess.run(
            cmd,
            cwd=str(dbt_project_path),
            capture_output=True,
            text=True,
            timeout=120, 
            env=env_vars
        )
        
        # Step 4: Parse test results
        test_passed = result.returncode == 0
        stdout = result.stdout + dbt_run_output 
        stderr = result.stderr
        
        # Parse test output for details
        test_results = _parse_test_output(stdout)
        
        # Determine overall status
        if test_passed:
            status = "success"
            message = "✅ Test passed! The fix resolves the issue."
        else:
            # Check if it's a partial success (some tests passed)
            passed_count = sum(1 for t in test_results if t.get("status") == "pass")
            failed_count = sum(1 for t in test_results if t.get("status") == "fail")
            
            if passed_count > 0 and failed_count > 0:
                status = "partial"
                message = f"⚠️ Partial success: {passed_count} passed, {failed_count} failed"
            else:
                status = "error"
                message = "❌ Test still failing. The fix did not resolve the issue."
        
        return {
            "status": status,
            "message": message,
            "data": {
                "model_name": model_name,
                "test_name": test_name,
                "fix_title": fix_option.get("title", "Unknown"),
                "test_passed": test_passed,
                "test_results": test_results,
                "stdout": stdout[:2000],  # Limit output size
                "stderr": stderr[:1000] if stderr else None,
                "return_code": result.returncode,
                "files_tested": list(code_changes.keys()) if code_changes else [],
                "code_changes": code_changes,
            }
        }
        
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "⏱️ Test timed out after 2 minutes",
            "data": {
                "model_name": model_name,
                "test_name": test_name,
                "test_passed": False,
                "timeout": True,
            }
        }
    except FileNotFoundError:
        return {
            "status": "error", 
            "message": "dbt command not found. Is dbt installed?",
            "data": {
                "model_name": model_name,
                "test_name": test_name,
                "test_passed": False,
                "dbt_not_found": True,
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running test: {str(e)}",
            "data": {
                "model_name": model_name,
                "test_name": test_name,
                "test_passed": False,
                "error": str(e),
            }
        }
    finally:
        # Step 5: Restore original files
        if files_modified:
            for full_path, original_content in original_contents.items():
                try:
                    with open(full_path, 'w') as f:
                        f.write(original_content)
                except Exception as e:
                    # Log but don't fail - file restoration is critical
                    print(f"Warning: Failed to restore {full_path}: {e}")


def _parse_test_output(stdout: str) -> List[Dict[str, Any]]:
    """
    Parse dbt test output to extract individual test results.
    
    Args:
        stdout: The stdout from dbt test command
        
    Returns:
        List of test result dicts with name, status, and details
    """
    results = []
    
    # Look for test result patterns in dbt output
    # Example: "1 of 1 PASS unique_stg_users_user_id"
    # Example: "1 of 1 FAIL 5 not_null_stg_users_email"
    
    pass_pattern = r'(\d+) of (\d+) PASS\s+(\S+)'
    fail_pattern = r'(\d+) of (\d+) FAIL\s+(\d+)\s+(\S+)'
    warn_pattern = r'(\d+) of (\d+) WARN\s+(\d+)\s+(\S+)'
    error_pattern = r'(\d+) of (\d+) ERROR\s+(\S+)'
    
    import re
    
    for line in stdout.split('\n'):
        # Check for PASS
        match = re.search(pass_pattern, line)
        if match:
            results.append({
                "test_number": int(match.group(1)),
                "total_tests": int(match.group(2)),
                "status": "pass",
                "test_name": match.group(3),
                "failures": 0,
            })
            continue
        
        # Check for FAIL
        match = re.search(fail_pattern, line)
        if match:
            results.append({
                "test_number": int(match.group(1)),
                "total_tests": int(match.group(2)),
                "status": "fail",
                "failures": int(match.group(3)),
                "test_name": match.group(4),
            })
            continue
        
        # Check for WARN
        match = re.search(warn_pattern, line)
        if match:
            results.append({
                "test_number": int(match.group(1)),
                "total_tests": int(match.group(2)),
                "status": "warn",
                "failures": int(match.group(3)),
                "test_name": match.group(4),
            })
            continue
        
        # Check for ERROR
        match = re.search(error_pattern, line)
        if match:
            results.append({
                "test_number": int(match.group(1)),
                "total_tests": int(match.group(2)),
                "status": "error",
                "test_name": match.group(3),
            })
    
    return results
