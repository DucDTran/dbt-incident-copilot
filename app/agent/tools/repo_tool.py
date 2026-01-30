from pathlib import Path
from typing import Dict, Any, Optional
import os
import shutil
from datetime import datetime

from app.config import get_settings


def tool_read_repo(file_path: str) -> Dict[str, Any]:
    settings = get_settings()
    
    if os.path.isabs(file_path):
        full_path = Path(file_path)
    else:
        full_path = settings.dbt_project_path / file_path
    
    if not full_path.exists():
        return {
            "status": "error",
            "message": f"File not found: {full_path}",
            "data": None
        }
    
    try:
        with open(full_path, 'r') as f:
            content = f.read()
        
        stat = full_path.stat()
        
        return {
            "status": "success",
            "message": f"File read successfully: {full_path.name}",
            "data": {
                "path": str(full_path),
                "relative_path": str(full_path.relative_to(settings.dbt_project_path)) 
                    if str(full_path).startswith(str(settings.dbt_project_path)) else file_path,
                "content": content,
                "extension": full_path.suffix,
                "size_bytes": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                "lines": len(content.splitlines()),
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading file: {str(e)}",
            "data": None
        }


def tool_write_repo(file_path: str, content: str, create_backup: bool = True) -> Dict[str, Any]:
    settings = get_settings()
    
    if os.path.isabs(file_path):
        full_path = Path(file_path)
    else:
        full_path = settings.dbt_project_path / file_path
    
    try:
        # Create backup if requested and file exists
        backup_path = None
        if create_backup and full_path.exists():
            backup_dir = settings.dbt_project_path / ".copilot_backups"
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{full_path.stem}_{timestamp}{full_path.suffix}"
            backup_path = backup_dir / backup_filename
            shutil.copy2(full_path, backup_path)
        
        # Create parent directories if needed
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write the new content
        with open(full_path, 'w') as f:
            f.write(content)
        
        return {
            "status": "success",
            "message": f"File written successfully: {full_path.name}",
            "data": {
                "path": str(full_path),
                "backup_path": str(backup_path) if backup_path else None,
                "lines_written": len(content.splitlines()),
            }
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error writing file: {str(e)}",
            "data": None
        }


def find_file_by_model_name(model_name: str) -> Dict[str, Any]:
    
    settings = get_settings()
    models_path = settings.models_path
    
    for sql_file in models_path.rglob(f"{model_name}.sql"):
        return {
            "status": "success",
            "message": f"Found model file for '{model_name}'",
            "data": {
                "path": str(sql_file),
                "relative_path": str(sql_file.relative_to(settings.dbt_project_path)),
            }
        }
    
    return {
        "status": "error",
        "message": f"Model file not found for '{model_name}'",
        "data": None
    }


def find_schema_file(model_name: str) -> Dict[str, Any]:

    settings = get_settings()
    models_path = settings.models_path
    
    import yaml
    
    # Search all schema.yml files
    for yml_file in models_path.rglob("schema.yml"):
        try:
            with open(yml_file, 'r') as f:
                schema = yaml.safe_load(f)
            
            if not schema:
                continue
                
            # Check if model is defined in this file
            models = schema.get("models", [])
            for model in models:
                if model.get("name") == model_name:
                    return {
                        "status": "success",
                        "message": f"Found schema file for '{model_name}'",
                        "data": {
                            "path": str(yml_file),
                            "relative_path": str(yml_file.relative_to(settings.dbt_project_path)),
                            "model_definition": model,
                        }
                    }
        except Exception:
            continue
    
    return {
        "status": "error",
        "message": f"Schema file not found for '{model_name}'",
        "data": None
    }


def find_variable_definition(var_name: str) -> Dict[str, Any]:
    """
    Search the dbt project for a variable definition matching `var_name`.

    Returns the file path and the YAML-parsed content where the variable is defined.
    """
    settings = get_settings()
    project_file = settings.dbt_project_path / "dbt_project.yml"

    import yaml
    try:
        if project_file.exists():
            with open(project_file, 'r') as f:
                proj = yaml.safe_load(f)
            vars_section = proj.get('vars', {}) or proj.get('vars', [])
            # vars may be a dict
            if isinstance(vars_section, dict) and var_name in vars_section:
                return {
                    "status": "success",
                    "message": f"Found variable in dbt_project.yml",
                    "data": {
                        "path": str(project_file),
                        "relative_path": str(project_file.relative_to(settings.dbt_project_path)),
                        "value": vars_section.get(var_name),
                    }
                }
    except Exception:
        pass

    # Search model-level folder YAML files for var overrides (simple heuristic)
    for yml_file in settings.models_path.rglob("*.yml"):
        try:
            with open(yml_file, 'r') as f:
                content = yaml.safe_load(f)
            if not content:
                continue
            # Look for top-level 'vars' as well as arbitrary keys
            if isinstance(content, dict) and 'vars' in content:
                if isinstance(content['vars'], dict) and var_name in content['vars']:
                    return {
                        "status": "success",
                        "message": f"Found variable in {yml_file.name}",
                        "data": {
                            "path": str(yml_file),
                            "relative_path": str(yml_file.relative_to(settings.dbt_project_path)),
                            "value": content['vars'].get(var_name),
                        }
                    }
        except Exception:
            continue

    return {
        "status": "error",
        "message": f"Variable '{var_name}' not found in project YAML files",
        "data": None
    }


def list_model_files(model_name: str) -> Dict[str, Any]:
    """
    Return all files related to a model (SQL, schema ymls) to allow agents to consider multiple targets.
    """
    settings = get_settings()
    results = []

    for sql_file in settings.models_path.rglob(f"{model_name}.sql"):
        results.append({
            "path": str(sql_file),
            "relative_path": str(sql_file.relative_to(settings.dbt_project_path)),
            "type": "sql"
        })

    for yml_file in settings.models_path.rglob("schema.yml"):
        import yaml
        try:
            with open(yml_file, 'r') as f:
                schema = yaml.safe_load(f)
            if not schema:
                continue
            for model in schema.get('models', []) or []:
                if model.get('name') == model_name:
                    results.append({
                        "path": str(yml_file),
                        "relative_path": str(yml_file.relative_to(settings.dbt_project_path)),
                        "type": "schema"
                    })
        except Exception:
            continue

    if results:
        return {"status": "success", "data": results}
    return {"status": "error", "message": f"No files found for model {model_name}", "data": None}


def get_file_diff(original_content: str, new_content: str) -> Dict[str, Any]:

    import difflib
    
    original_lines = original_content.splitlines(keepends=True)
    new_lines = new_content.splitlines(keepends=True)
    
    diff = list(difflib.unified_diff(
        original_lines,
        new_lines,
        fromfile='original',
        tofile='modified',
        lineterm=''
    ))
    
    additions = sum(1 for line in diff if line.startswith('+') and not line.startswith('+++'))
    deletions = sum(1 for line in diff if line.startswith('-') and not line.startswith('---'))
    
    return {
        "status": "success",
        "message": "Diff generated successfully",
        "data": {
            "diff": ''.join(diff),
            "additions": additions,
            "deletions": deletions,
            "has_changes": len(diff) > 0,
        }
    }

