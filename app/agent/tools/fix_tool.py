from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum
import subprocess
import os

from app.config import get_settings
from .repo_tool import tool_read_repo, tool_write_repo, get_file_diff


class FixType(Enum):
    UPDATE_ACCEPTED_VALUES = "update_accepted_values"
    ADD_FILTER = "add_filter"
    ADD_COALESCE = "add_coalesce"
    UPDATE_RANGE = "update_range"
    SNOOZE = "snooze"
    ADD_WHERE_CLAUSE = "add_where_clause"


@dataclass
class FixOption:
    id: str
    title: str
    description: str
    fix_type: FixType
    risk_level: str  # 'low', 'medium', 'high'
    code_changes: Dict[str, str]  # file_path -> new_content
    rationale: str
    business_justification: Optional[str] = None


def apply_fix(
    fix_option: Dict[str, Any],
    incident: Dict[str, Any] = None,
    dry_run: bool = False
) -> Dict[str, Any]:

    code_changes = fix_option.get("code_changes", {})
    
    if not code_changes:
        return {
            "status": "info",
            "message": "No automatic code changes available. Manual action required.",
            "data": {
                "fix_id": fix_option.get("id"),
                "applied": False,
                "reason": "This fix type requires manual code changes.",
            }
        }
    
    results = []
    all_success = True
    
    for file_path, new_content in code_changes.items():
        original_result = tool_read_repo(file_path)
        
        if original_result["status"] == "error":
            results.append({
                "file": file_path,
                "status": "error",
                "message": original_result["message"],
            })
            all_success = False
            continue
        
        original_content = original_result["data"]["content"]
        relative_path = original_result["data"].get("relative_path", file_path)
        
        diff_result = get_file_diff(original_content, new_content)
        
        if dry_run:
            results.append({
                "file": file_path,
                "relative_path": relative_path,
                "status": "dry_run",
                "diff": diff_result["data"]["diff"] if diff_result["status"] == "success" else "",
                "additions": diff_result["data"]["additions"] if diff_result["status"] == "success" else 0,
                "deletions": diff_result["data"]["deletions"] if diff_result["status"] == "success" else 0,
                "original_content": original_content,
                "new_content": new_content,
            })
        else:
            write_result = tool_write_repo(file_path, new_content)
            success = write_result["status"] == "success"
            if not success:
                all_success = False
            results.append({
                "file": file_path,
                "relative_path": relative_path,
                "status": "applied" if success else "error",
                "message": write_result["message"],
                "backup": write_result.get("data", {}).get("backup_path"),
                "diff": diff_result["data"]["diff"] if diff_result["status"] == "success" else "",
                "additions": diff_result["data"]["additions"] if diff_result["status"] == "success" else 0,
                "deletions": diff_result["data"]["deletions"] if diff_result["status"] == "success" else 0,
            })
    
    return {
        "status": "success" if all_success else "partial",
        "message": f"{'Dry run' if dry_run else 'Applied'} fix: {fix_option.get('title')}",
        "data": {
            "fix_id": fix_option.get("id"),
            "dry_run": dry_run,
            "results": results,
            "files_modified": len([r for r in results if r.get("status") == "applied"]),
        }
    }


def run_dbt_test(model_name: str, test_name: str = None) -> Dict[str, Any]:
    """Run dbt test for a specific model and optional test."""
    
    settings = get_settings()
    
    try:
        # Build the command with disable_elementary flag to prevent syncing local verification results
        if test_name:
            cmd = ["dbt", "test", "--select", f"{model_name},{test_name}", "--vars", '{"disable_elementary": true}']
        else:
            cmd = ["dbt", "test", "--select", model_name, "--vars", '{"disable_elementary": true}']
        
        result = subprocess.run(
            cmd,
            cwd=str(settings.dbt_project_path),
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "DBT_PROFILES_DIR": str(settings.dbt_profiles_dir)}
        )
        
        success = result.returncode == 0
        
        # Parse output to find test results
        test_passed = "1 of 1 PASS" in result.stdout or "PASS" in result.stdout
        test_failed = "FAIL" in result.stdout or "ERROR" in result.stdout
        
        return {
            "status": "success" if success else "error",
            "message": "Test passed!" if test_passed else "Test failed" if test_failed else "Test completed",
            "data": {
                "model_name": model_name,
                "test_passed": test_passed,
                "stdout": result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout,
                "stderr": result.stderr[-500:] if result.stderr else None,
                "return_code": result.returncode,
            }
        }
    
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "message": "Test timed out after 120 seconds",
            "data": {"model_name": model_name, "test_passed": False}
        }
    except FileNotFoundError:
        return {
            "status": "error",
            "message": "dbt command not found",
            "data": {"model_name": model_name, "test_passed": False}
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error running test: {str(e)}",
            "data": {"model_name": model_name, "test_passed": False}
        }
