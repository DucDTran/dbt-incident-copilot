import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class ModelInfo:
    unique_id: str
    name: str
    resource_type: str
    path: str
    schema: str
    database: str
    depends_on: List[str]
    description: str
    columns: Dict[str, Any]
    raw_sql: str


@dataclass
class TestInfo:
    unique_id: str
    name: str
    test_type: str  # 'generic' or 'singular'
    model_name: str
    column_name: Optional[str]
    test_metadata: Dict[str, Any]


def tool_read_manifest() -> Dict[str, Any]:
    
    settings = get_settings()
    manifest_path = settings.manifest_path
    
    if not manifest_path.exists():
        return {
            "status": "error",
            "message": f"Manifest not found at {manifest_path}. Run 'dbt compile' first.",
            "data": None
        }
    
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        # Extract key information
        models = {}
        tests = {}
        sources = {}
        
        # Parse nodes
        for node_id, node in manifest.get("nodes", {}).items():
            resource_type = node.get("resource_type")
            
            if resource_type == "model":
                models[node_id] = {
                    "name": node.get("name"),
                    "path": node.get("original_file_path"),
                    "schema": node.get("schema"),
                    "database": node.get("database"),
                    "depends_on": node.get("depends_on", {}).get("nodes", []),
                    "description": node.get("description", ""),
                    "columns": node.get("columns", {}),
                    "raw_sql": node.get("raw_code", ""),
                }
            elif resource_type == "test":
                # Determine test type and associated model
                test_metadata = node.get("test_metadata", {})
                attached_node = node.get("attached_node", "")
                column_name = node.get("column_name")
                
                tests[node_id] = {
                    "name": node.get("name"),
                    "test_type": "generic" if test_metadata else "singular",
                    "attached_node": attached_node,
                    "column_name": column_name,
                    "test_metadata": test_metadata,
                    "path": node.get("original_file_path"),
                    "raw_sql": node.get("raw_code", ""),
                }
        
        # Parse sources
        for source_id, source in manifest.get("sources", {}).items():
            sources[source_id] = {
                "name": source.get("name"),
                "source_name": source.get("source_name"),
                "schema": source.get("schema"),
                "identifier": source.get("identifier"),
            }
        
        return {
            "status": "success",
            "message": "Manifest parsed successfully",
            "data": {
                "models": models,
                "tests": tests,
                "sources": sources,
                "metadata": {
                    "dbt_version": manifest.get("metadata", {}).get("dbt_version"),
                    "project_name": manifest.get("metadata", {}).get("project_name"),
                    "generated_at": manifest.get("metadata", {}).get("generated_at"),
                }
            }
        }
    except json.JSONDecodeError as e:
        return {
            "status": "error",
            "message": f"Failed to parse manifest: {str(e)}",
            "data": None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error reading manifest: {str(e)}",
            "data": None
        }


def get_model_lineage(model_name: str) -> Dict[str, Any]:

    manifest_result = tool_read_manifest()
    
    if manifest_result["status"] == "error":
        return manifest_result
    
    manifest_data = manifest_result["data"]
    models = manifest_data["models"]
    
    target_model_id = None
    for model_id, model in models.items():
        if model["name"] == model_name:
            target_model_id = model_id
            break
    
    if not target_model_id:
        return {
            "status": "error",
            "message": f"Model '{model_name}' not found in manifest",
            "data": None
        }
    
    target_model = models[target_model_id]
    
    upstream = []
    for dep_id in target_model["depends_on"]:
        if dep_id in models:
            upstream.append({
                "id": dep_id,
                "name": models[dep_id]["name"],
                "type": "model"
            })
        elif dep_id in manifest_data["sources"]:
            source = manifest_data["sources"][dep_id]
            upstream.append({
                "id": dep_id,
                "name": f"{source['source_name']}.{source['name']}",
                "type": "source"
            })
    
    downstream = []
    for model_id, model in models.items():
        if target_model_id in model["depends_on"]:
            downstream.append({
                "id": model_id,
                "name": model["name"],
                "type": "model"
            })
    
    associated_tests = []
    for test_id, test in manifest_data["tests"].items():
        if test["attached_node"] == target_model_id:
            associated_tests.append({
                "id": test_id,
                "name": test["name"],
                "column": test["column_name"],
                "type": test["test_type"],
            })
    
    return {
        "status": "success",
        "message": f"Lineage retrieved for model '{model_name}'",
        "data": {
            "model": {
                "id": target_model_id,
                "name": model_name,
                "path": target_model["path"],
                "description": target_model["description"],
            },
            "upstream": upstream,
            "downstream": downstream,
            "tests": associated_tests,
        }
    }


def find_test_model(test_name: str) -> Dict[str, Any]:

    manifest_result = tool_read_manifest()
    
    if manifest_result["status"] == "error":
        return manifest_result
    
    manifest_data = manifest_result["data"]
    tests = manifest_data["tests"]
    models = manifest_data["models"]
    
    for test_id, test in tests.items():
        if test["name"] == test_name or test_name in test_id:
            attached_node = test["attached_node"]
            if attached_node in models:
                model = models[attached_node]
                return {
                    "status": "success",
                    "message": f"Found model for test '{test_name}'",
                    "data": {
                        "test": test,
                        "model": {
                            "id": attached_node,
                            "name": model["name"],
                            "path": model["path"],
                            "schema": model["schema"],
                        }
                    }
                }
    
    return {
        "status": "error",
        "message": f"Test '{test_name}' not found in manifest",
        "data": None
    }

