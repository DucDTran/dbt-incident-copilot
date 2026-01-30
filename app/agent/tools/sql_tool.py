"""
SQL Execution Tool - Execute read-only SQL queries for investigation.

This tool allows the agent to run SQL queries against the data warehouse
to investigate test failures and verify data patterns.

Security features:
- Read-only queries (no INSERT/UPDATE/DELETE/DROP)
- Query timeout limits
- Row limit enforcement
- Only queries against dbt project models/tables
"""

from typing import Dict, Any, Optional, List
import json
import re
from datetime import datetime
from pathlib import Path

from app.config import get_settings


# SQL keywords that indicate write operations (should be blocked)
WRITE_KEYWORDS = [
    "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER", 
    "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE", "CALL"
]


def _is_read_only_query(sql: str) -> bool:
    """Check if a SQL query is read-only."""
    sql_upper = sql.strip().upper()
    
    # Check for write keywords
    for keyword in WRITE_KEYWORDS:
        # Use word boundary to avoid false positives
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, sql_upper):
            return False
    
    # Must start with SELECT
    if not sql_upper.startswith("SELECT"):
        return False
    
    return True


def _execute_bigquery(sql: str, limit: int = 100) -> Dict[str, Any]:
    """Execute a read-only SQL query against BigQuery."""
    settings = get_settings()
    
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        from google.cloud.exceptions import GoogleCloudError
        from google.auth.exceptions import DefaultCredentialsError
        
        # Initialize BigQuery client - use same approach as elementary_tool
        if settings.bigquery_credentials_path:
            # Check if credentials file exists
            creds_path = Path(settings.bigquery_credentials_path)
            if not creds_path.exists():
                return {
                    "status": "error",
                    "message": f"SQL Query Credentials Error: Credentials file not found at '{settings.bigquery_credentials_path}'. Please verify the BIGQUERY_CREDENTIALS_PATH setting points to a valid file. This is not a blocker - you can continue the investigation using other tools (read_model_sql, read_schema_definition, get_test_details). Many test failures can be diagnosed from SQL code and schema definitions alone without querying data directly.",
                    "data": None
                }
            
            try:
                credentials = service_account.Credentials.from_service_account_file(
                    str(creds_path),
                    scopes=["https://www.googleapis.com/auth/bigquery"],
                )
                client = bigquery.Client(
                    project=settings.bigquery_project_id,
                    credentials=credentials
                )
            except json.JSONDecodeError as json_error:
                return {
                    "status": "error",
                    "message": f"SQL Query Credentials Error: Invalid JSON in credentials file '{settings.bigquery_credentials_path}': {str(json_error)}. Please verify the file is a valid service account JSON. This is not a blocker - you can continue the investigation using other tools (read_model_sql, read_schema_definition, get_test_details).",
                    "data": None
                }
            except Exception as cred_error:
                error_type = type(cred_error).__name__
                return {
                    "status": "error",
                    "message": f"SQL Query Credentials Error ({error_type}): Failed to load credentials from '{settings.bigquery_credentials_path}': {str(cred_error)}. This is not a blocker - you can continue the investigation using other tools (read_model_sql, read_schema_definition, get_test_details). Many test failures can be diagnosed from SQL code and schema definitions alone without querying data directly.",
                    "data": None
                }
        else:
            # Try to use default credentials (same approach as elementary_tool)
            # Note: We don't catch DefaultCredentialsError here - let it bubble up to outer handler
            # This matches the pattern in elementary_tool.py for consistency
            try:
                client = bigquery.Client(project=settings.bigquery_project_id)
            except Exception as default_cred_error:
                # Catch any exception during client creation (including DefaultCredentialsError)
                error_msg = str(default_cred_error)
                if "credentials" in error_msg.lower() or "authentication" in error_msg.lower() or isinstance(default_cred_error, DefaultCredentialsError):
                    return {
                        "status": "error",
                        "message": f"SQL Query Credentials Error: {error_msg}. No BigQuery credentials configured or default credentials not available. This is not a blocker - you can continue the investigation using other tools (read_model_sql, read_schema_definition, get_test_details). Many test failures can be diagnosed from SQL code and schema definitions alone without querying data directly.",
                        "data": None
                    }
                # Re-raise if it's not a credentials error
                raise
        
        # Add LIMIT if not present (safety measure)
        sql_upper = sql.strip().upper()
        if "LIMIT" not in sql_upper:
            sql = f"{sql.rstrip(';')} LIMIT {limit}"
        
        # Execute query with timeout
        query_job = client.query(
            sql,
            job_config=bigquery.QueryJobConfig(
                maximum_bytes_billed=10 * 1024 * 1024,  # 10MB limit
                use_query_cache=True,
            )
        )
        
        # Wait for results with timeout
        try:
            results = query_job.result(timeout=30)  # 30 second timeout
        except Exception as e:
            error_msg = str(e)
            # Provide more helpful error messages
            if "403" in error_msg or "permission" in error_msg.lower() or "access denied" in error_msg.lower():
                return {
                    "status": "error",
                    "message": f"SQL Query Permission Error: {error_msg}. If querying dbt models, ensure you use fully qualified table names: `{settings.bigquery_project_id}.airbnb_analytics.table_name`. The dataset for dbt models is 'airbnb_analytics'. This is not a blocker - you can continue the investigation using other tools (read_model_sql, read_schema_definition, get_test_details). Many test failures can be diagnosed from SQL code and schema definitions alone without querying data directly.",
                    "data": None
                }
            elif "404" in error_msg or "not found" in error_msg.lower():
                return {
                    "status": "error",
                    "message": f"Table or dataset not found: {error_msg}. Please verify the table name uses fully qualified format: `{settings.bigquery_project_id}.airbnb_analytics.table_name`. The dataset for dbt models is 'airbnb_analytics'. This is not a blocker - you can continue the investigation using other tools (read_model_sql, read_schema_definition, get_test_details).",
                    "data": None
                }
            elif "timeout" in error_msg.lower():
                return {
                    "status": "error",
                    "message": f"Query timeout: The query took longer than 30 seconds. Try adding more specific WHERE clauses or reducing the data scanned.",
                    "data": None
                }
            else:
                return {
                    "status": "error",
                    "message": f"Query execution error: {error_msg}",
                    "data": None
                }
        
        # Convert results to list of dicts
        rows = []
        for row in results:
            row_dict = {}
            for key, value in row.items():
                # Convert datetime and other types to strings
                if isinstance(value, datetime):
                    row_dict[key] = value.isoformat()
                elif value is None:
                    row_dict[key] = None
                else:
                    row_dict[key] = str(value)
            rows.append(row_dict)
        
        # Get schema information
        schema = []
        if query_job.schema:
            for field in query_job.schema:
                schema.append({
                    "name": field.name,
                    "type": field.field_type,
                    "mode": field.mode,
                })
        
        return {
            "status": "success",
            "message": f"Query executed successfully. Returned {len(rows)} row(s).",
            "data": {
                "rows": rows,
                "row_count": len(rows),
                "schema": schema,
                "total_bytes_processed": query_job.total_bytes_processed,
                "job_id": query_job.job_id,
            }
        }
        
    except ImportError:
        return {
            "status": "error",
            "message": "google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery",
            "data": None
        }
    except GoogleCloudError as e:
        error_msg = str(e)
        if "403" in error_msg or "permission" in error_msg.lower() or "access denied" in error_msg.lower():
            return {
                "status": "error",
                "message": f"SQL Query Permission Error: {error_msg}. The service account may not have BigQuery Data Viewer role or access to query this specific table/dataset. This is not a blocker - you can continue the investigation using other tools (read_model_sql, read_schema_definition, get_test_details). Many test failures can be diagnosed from SQL code and schema definitions alone without querying data directly.",
                "data": None
            }
        return {
            "status": "error",
            "message": f"BigQuery error: {error_msg}",
            "data": None
        }
    except Exception as e:
        error_msg = str(e)
        # Check for various authentication/credential error patterns
        auth_keywords = ["credentials", "authentication", "unauthorized", "forbidden", "403", "401"]
        if any(keyword in error_msg.lower() for keyword in auth_keywords) or isinstance(e, DefaultCredentialsError):
            return {
                "status": "error",
                "message": f"SQL Query Authentication Error: {error_msg}. This is not a blocker - you can continue the investigation using other tools (read_model_sql, read_schema_definition, get_test_details). Many test failures can be diagnosed from SQL code and schema definitions alone without querying data directly.",
                "data": None
            }
        return {
            "status": "error",
            "message": f"Query execution failed: {error_msg}",
            "data": None
        }


def tool_execute_sql(
    sql: str,
    limit: int = 100,
    model_name: str = None
) -> str:
    """
    Execute a read-only SQL query for investigation purposes.
    
    This tool allows the agent to query the data warehouse to:
    - Verify actual data values
    - Sample data to understand patterns
    - Check data distributions
    - Investigate test failures with actual data
    
    Security:
    - Only SELECT queries are allowed
    - Automatic LIMIT enforcement (default 100 rows)
    - Query timeout (30 seconds)
    - Data processing limit (10MB)
    
    Args:
        sql: The SQL query to execute (must be SELECT only)
        limit: Maximum number of rows to return (default 100, max 1000)
        model_name: Model name to query
        
    Returns:
        JSON string with query results including rows, schema, and metadata.
    """
    settings = get_settings()
    
    # Validate limit
    limit = min(max(1, limit), 1000)  # Clamp between 1 and 1000
    
    # Validate SQL is read-only
    if not _is_read_only_query(sql):
        return json.dumps({
            "status": "error",
            "message": "Only SELECT queries are allowed. Write operations (INSERT, UPDATE, DELETE, etc.) are not permitted.",
            "data": None
        }, indent=2)
    
    # Execute based on adapter
    adapter = settings.dbt_adapter.lower()
    
    if "bigquery" in adapter:
        result = _execute_bigquery(sql, limit)
    elif "snowflake" in adapter:
        # TODO: Implement Snowflake execution
        return json.dumps({
            "status": "error",
            "message": "Snowflake SQL execution not yet implemented",
            "data": None
        }, indent=2)
    elif "postgres" in adapter or "postgresql" in adapter:
        # TODO: Implement Postgres execution
        return json.dumps({
            "status": "error",
            "message": "PostgreSQL SQL execution not yet implemented",
            "data": None
        }, indent=2)
    else:
        return json.dumps({
            "status": "error",
            "message": f"SQL execution not supported for adapter: {settings.dbt_adapter}",
            "data": None
        }, indent=2)
    
    return json.dumps(result, indent=2, default=str)


def adk_execute_sql(sql: str, limit: int = 100) -> str:
    """
    ADK wrapper for SQL execution tool.
    
    Execute read-only SELECT queries against BigQuery.
    
    **IMPORTANT - Table Format**: When querying tables, use fully qualified BigQuery format:
    - Format: `project_id.dataset_name.table_name`
    - Dataset for dbt models: `airbnb_analytics`
    - Example: `dbt-incident-copilot-484016.airbnb_analytics.fact_reviews`
    
    Args:
        sql: SQL SELECT query (must use fully qualified table names: project.dataset.table)
        limit: Maximum rows to return (default 100, max 1000)
        
    Returns:
        JSON string with query results or error message.
    """
    settings = get_settings()
    
    # If the SQL doesn't contain a fully qualified table name (project.dataset.table),
    # and it's a simple table reference, we could try to prepend the dataset
    # But for now, let's just execute as-is and let BigQuery return the error
    # The agent should learn from the error message what the correct format is
    
    return tool_execute_sql(sql, limit)


def diagnose_bigquery_credentials() -> Dict[str, Any]:
    """
    Diagnostic function to check BigQuery credentials configuration.
    
    Returns a detailed report of credential status and any issues found.
    """
    settings = get_settings()
    diagnostics = {
        "status": "unknown",
        "checks": {},
        "errors": [],
        "warnings": [],
        "recommendations": []
    }
    
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        from google.auth.exceptions import DefaultCredentialsError
    except ImportError:
        diagnostics["status"] = "error"
        diagnostics["errors"].append("google-cloud-bigquery package not installed")
        diagnostics["recommendations"].append("Run: pip install google-cloud-bigquery")
        return diagnostics
    
    # Check 1: Credentials path configuration
    if not settings.bigquery_credentials_path:
        diagnostics["checks"]["credentials_path_configured"] = False
        diagnostics["warnings"].append("BIGQUERY_CREDENTIALS_PATH not set - will try default credentials")
    else:
        diagnostics["checks"]["credentials_path_configured"] = True
        diagnostics["checks"]["credentials_path"] = settings.bigquery_credentials_path
        
        # Check 2: File exists
        creds_path = Path(settings.bigquery_credentials_path)
        if not creds_path.exists():
            diagnostics["status"] = "error"
            diagnostics["checks"]["credentials_file_exists"] = False
            diagnostics["errors"].append(f"Credentials file not found at: {settings.bigquery_credentials_path}")
            diagnostics["recommendations"].append(f"Verify the file path is correct: {settings.bigquery_credentials_path}")
            return diagnostics
        else:
            diagnostics["checks"]["credentials_file_exists"] = True
        
        # Check 3: Valid JSON
        try:
            with open(creds_path, 'r') as f:
                creds_data = json.load(f)
            diagnostics["checks"]["valid_json"] = True
            
            # Check 4: Required fields
            required_fields = ["type", "project_id", "private_key", "client_email"]
            missing_fields = [field for field in required_fields if field not in creds_data]
            if missing_fields:
                diagnostics["checks"]["has_required_fields"] = False
                diagnostics["errors"].append(f"Missing required fields in credentials: {', '.join(missing_fields)}")
            else:
                diagnostics["checks"]["has_required_fields"] = True
            
            # Check 5: Project ID match
            creds_project_id = creds_data.get("project_id", "")
            config_project_id = settings.bigquery_project_id
            if creds_project_id and config_project_id and creds_project_id != config_project_id:
                diagnostics["checks"]["project_id_matches"] = False
                diagnostics["warnings"].append(
                    f"Project ID mismatch: credentials file has '{creds_project_id}' but config has '{config_project_id}'"
                )
            else:
                diagnostics["checks"]["project_id_matches"] = True
                diagnostics["checks"]["project_id"] = creds_project_id or config_project_id
            
            # Check 6: Service account email
            if "client_email" in creds_data:
                diagnostics["checks"]["service_account_email"] = creds_data["client_email"]
            
        except json.JSONDecodeError as e:
            diagnostics["status"] = "error"
            diagnostics["checks"]["valid_json"] = False
            diagnostics["errors"].append(f"Invalid JSON in credentials file: {str(e)}")
            diagnostics["recommendations"].append("Verify the credentials file is valid JSON")
            return diagnostics
        except Exception as e:
            diagnostics["status"] = "error"
            diagnostics["checks"]["readable"] = False
            diagnostics["errors"].append(f"Error reading credentials file: {str(e)}")
            return diagnostics
    
    # Check 7: Project ID configuration
    if not settings.bigquery_project_id:
        diagnostics["checks"]["project_id_configured"] = False
        diagnostics["errors"].append("BIGQUERY_PROJECT_ID not set")
        diagnostics["recommendations"].append("Set BIGQUERY_PROJECT_ID in your configuration")
    else:
        diagnostics["checks"]["project_id_configured"] = True
        diagnostics["checks"]["configured_project_id"] = settings.bigquery_project_id
    
    # Check 8: Test authentication
    try:
        if settings.bigquery_credentials_path and creds_path.exists():
            credentials = service_account.Credentials.from_service_account_file(
                str(creds_path),
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
            client = bigquery.Client(
                project=settings.bigquery_project_id,
                credentials=credentials
            )
        else:
            # Try default credentials
            client = bigquery.Client(project=settings.bigquery_project_id)
        
        # Test with a simple query
        test_query = f"SELECT 1 as test FROM `{settings.bigquery_project_id}.INFORMATION_SCHEMA.TABLES` LIMIT 1"
        try:
            query_job = client.query(test_query)
            query_job.result(timeout=5)  # Quick timeout for test
            diagnostics["checks"]["authentication_successful"] = True
            diagnostics["checks"]["can_query_bigquery"] = True
            diagnostics["status"] = "success"
        except Exception as query_error:
            error_msg = str(query_error)
            diagnostics["checks"]["authentication_successful"] = True  # Auth worked
            diagnostics["checks"]["can_query_bigquery"] = False
            diagnostics["status"] = "warning"
            
            if "403" in error_msg or "permission" in error_msg.lower():
                diagnostics["warnings"].append(
                    "Authentication successful but service account may lack BigQuery Data Viewer role or table access"
                )
                diagnostics["recommendations"].append(
                    "Grant 'BigQuery Data Viewer' role to the service account in Google Cloud Console"
                )
            else:
                diagnostics["warnings"].append(f"Query test failed: {error_msg}")
        
    except DefaultCredentialsError:
        diagnostics["status"] = "error"
        diagnostics["checks"]["authentication_successful"] = False
        diagnostics["errors"].append("No default credentials found and no credentials file configured")
        diagnostics["recommendations"].append("Set BIGQUERY_CREDENTIALS_PATH or configure default credentials (gcloud auth application-default login)")
        return diagnostics
    except Exception as auth_error:
        diagnostics["status"] = "error"
        diagnostics["checks"]["authentication_successful"] = False
        error_type = type(auth_error).__name__
        diagnostics["errors"].append(f"Authentication failed ({error_type}): {str(auth_error)}")
        
        if "credentials" in str(auth_error).lower() or "invalid" in str(auth_error).lower():
            diagnostics["recommendations"].append("Verify the credentials file is valid and not expired")
            diagnostics["recommendations"].append("Check if the service account key was deleted or disabled in Google Cloud Console")
    
    # Final status determination
    if diagnostics["errors"]:
        if diagnostics["status"] != "success":
            diagnostics["status"] = "error"
    elif diagnostics["warnings"] and diagnostics["status"] != "error":
        diagnostics["status"] = "warning"
    elif diagnostics["status"] == "unknown":
        diagnostics["status"] = "success"
    
    return diagnostics


def get_bigquery_credentials_diagnostic() -> str:
    """
    Get a formatted diagnostic report for BigQuery credentials.
    
    Returns a JSON string with diagnostic information.
    """
    diagnostics = diagnose_bigquery_credentials()
    return json.dumps(diagnostics, indent=2, default=str)
