from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass

from app.config import get_settings


@dataclass
class TestResult:
    test_id: str
    test_name: str
    model_name: str
    column_name: Optional[str]
    status: str  # 'pass', 'fail', 'warn', 'error'
    error_message: Optional[str]
    failed_rows: int
    failed_row_samples: List[Dict[str, Any]]
    test_type: str
    severity: str  # 'ERROR', 'WARN'
    executed_at: datetime
    execution_time: float


# Mock test results for demo purposes
MOCK_TEST_RESULTS: List[Dict[str, Any]] = [
    {
        "test_id": "test.airbnb_analytics.accepted_values_fact_reviews_sentiment",
        "test_name": "accepted_values_fact_reviews_sentiment__positive__neutral__negative",
        "model_name": "fact_reviews",
        "column_name": "sentiment",
        "status": "fail",
        "error_message": "Got 47 results, configured to fail if != 0. Found unexpected values: ['mixed', 'unknown']",
        "failed_rows": 47,
        "failed_row_samples": [
            {"review_id": 98234, "sentiment": "mixed", "review_date": "2024-12-15"},
            {"review_id": 98456, "sentiment": "unknown", "review_date": "2024-12-16"},
            {"review_id": 99012, "sentiment": "mixed", "review_date": "2024-12-18"},
        ],
        "test_type": "generic",
        "severity": "ERROR",
        "executed_at": (datetime.now() - timedelta(minutes=15)).isoformat(),
        "execution_time": 2.34,
    },
    {
        "test_id": "test.airbnb_analytics.not_null_dim_listing_host_id",
        "test_name": "not_null_dim_listing_host_id",
        "model_name": "dim_listing",
        "column_name": "host_id",
        "status": "fail",
        "error_message": "Got 12 results, configured to fail if != 0. Found 12 NULL values in host_id column.",
        "failed_rows": 12,
        "failed_row_samples": [
            {"listing_id": 45231, "host_id": None, "listing_name": "Cozy Downtown Apt"},
            {"listing_id": 45678, "host_id": None, "listing_name": "Beach House"},
            {"listing_id": 46012, "host_id": None, "listing_name": "Mountain Retreat"},
        ],
        "test_type": "generic",
        "severity": "ERROR",
        "executed_at": (datetime.now() - timedelta(minutes=12)).isoformat(),
        "execution_time": 1.87,
    },
    {
        "test_id": "test.airbnb_analytics.accepted_values_dim_listing_room_type",
        "test_name": "accepted_values_dim_listing_room_type__Entire_home_apt__Private_room__Shared_room__Hotel_room",
        "model_name": "dim_listing",
        "column_name": "room_type",
        "status": "fail",
        "error_message": "Got 8 results, configured to fail if != 0. Found unexpected values: ['Studio']",
        "failed_rows": 8,
        "failed_row_samples": [
            {"listing_id": 52341, "room_type": "Studio", "price": 89.00},
            {"listing_id": 52567, "room_type": "Studio", "price": 95.00},
        ],
        "test_type": "generic",
        "severity": "ERROR",
        "executed_at": (datetime.now() - timedelta(minutes=10)).isoformat(),
        "execution_time": 1.56,
    },
    {
        "test_id": "test.airbnb_analytics.values_in_range_dim_listing_price",
        "test_name": "values_in_range_dim_listing_price",
        "model_name": "dim_listing",
        "column_name": "price",
        "status": "fail",
        "error_message": "Got 3 results, configured to fail if != 0. Found prices outside allowed range (0.01-10000): [0.00, 15000.00, -50.00]",
        "failed_rows": 3,
        "failed_row_samples": [
            {"listing_id": 61234, "price": 0.00, "listing_name": "Free Stay Special"},
            {"listing_id": 61456, "price": 15000.00, "listing_name": "Ultra Luxury Penthouse"},
            {"listing_id": 61789, "price": -50.00, "listing_name": "Data Error Listing"},
        ],
        "test_type": "generic",
        "severity": "WARN",
        "executed_at": (datetime.now() - timedelta(minutes=8)).isoformat(),
        "execution_time": 0.98,
    },
    {
        "test_id": "test.airbnb_analytics.relationships_fact_reviews_listing_id",
        "test_name": "relationships_fact_reviews_listing_id__listing_id__ref_dim_listing_",
        "model_name": "fact_reviews",
        "column_name": "listing_id",
        "status": "fail",
        "error_message": "Got 23 results, configured to fail if != 0. Found 23 orphan records with listing_id not in dim_listing.",
        "failed_rows": 23,
        "failed_row_samples": [
            {"review_id": 112233, "listing_id": 99999, "reviewer_name": "John D."},
            {"review_id": 112456, "listing_id": 88888, "reviewer_name": "Sarah M."},
        ],
        "test_type": "generic",
        "severity": "ERROR",
        "executed_at": (datetime.now() - timedelta(minutes=5)).isoformat(),
        "execution_time": 3.21,
    },
]


def tool_query_elementary(
    status: Optional[str] = None,
    model_name: Optional[str] = None,
    limit: int = 50
) -> Dict[str, Any]:

    settings = get_settings()
    
    if settings.use_mock_data:
        return _query_mock_results(status, model_name, limit)
    else:
        return _query_bigquery_results(status, model_name, limit)


def _query_mock_results(
    status: Optional[str],
    model_name: Optional[str],
    limit: int
) -> Dict[str, Any]:

    results = MOCK_TEST_RESULTS.copy()
    
    if status:
        results = [r for r in results if r["status"] == status]
    
    if model_name:
        results = [r for r in results if r["model_name"] == model_name]
    
    results = results[:limit]
    
    return {
        "status": "success",
        "message": f"Found {len(results)} test results (mock mode)",
        "data": {
            "results": results,
            "total_count": len(results),
            "filters_applied": {
                "status": status,
                "model_name": model_name,
            },
            "is_mock": True,
        }
    }


def _query_bigquery_results(
    status: Optional[str],
    model_name: Optional[str],
    limit: int
) -> Dict[str, Any]:
    settings = get_settings()
    
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        
        if settings.bigquery_credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                settings.bigquery_credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
            client = bigquery.Client(
                project=settings.bigquery_project_id,
                credentials=credentials
            )
        else:
            client = bigquery.Client(project=settings.bigquery_project_id)
        
        # Build WHERE clause conditions
        where_conditions = ["1=1"]
        if status:
            where_conditions.append(f"status = '{status}'")
        if model_name:
            where_conditions.append(f"model_unique_id LIKE '%{model_name}%'")
        
        where_clause = " AND ".join(where_conditions)
        
        query = f"""
        WITH ranked_results AS (
            SELECT
                etr.id as test_result_id,
                etr.test_unique_id as test_id,
                etr.test_name,
                etr.model_unique_id,
                REGEXP_EXTRACT(etr.model_unique_id, r'model\\.\\w+\\.(\\w+)$') as model_name,
                etr.table_name,
                etr.column_name,
                etr.status,
                etr.test_results_description as error_message,
                COALESCE(etr.failures, 0) as failed_rows,
                etr.test_sub_type,
                etr.test_short_name,
                etr.severity,
                etr.detected_at as executed_at,
                dt.description as test_description,
                ROW_NUMBER() OVER (
                    PARTITION BY etr.test_unique_id 
                    ORDER BY etr.detected_at DESC
                ) as row_num
            FROM `{settings.bigquery_project_id}.{settings.bigquery_dataset}.elementary_test_results` etr
            LEFT JOIN `{settings.bigquery_project_id}.{settings.bigquery_dataset}.dbt_tests` dt
                ON etr.test_unique_id = dt.unique_id
            WHERE {where_clause}
        ),
        limited_failed_rows AS (
            SELECT
                elementary_test_results_id,
                result_row,
                ROW_NUMBER() OVER (
                    PARTITION BY elementary_test_results_id 
                    ORDER BY detected_at DESC
                ) as row_num
            FROM `{settings.bigquery_project_id}.{settings.bigquery_dataset}.test_result_rows`
            WHERE result_row IS NOT NULL
        ),
        results_with_rows AS (
            SELECT
                r.test_id,
                r.test_name,
                r.model_unique_id,
                r.model_name,
                r.table_name,
                r.column_name,
                r.status,
                r.error_message,
                r.failed_rows,
                r.test_sub_type,
                r.test_short_name,
                r.severity,
                r.executed_at,
                r.test_description,
                COALESCE(
                    ARRAY_AGG(
                        trr.result_row
                        IGNORE NULLS
                        ORDER BY trr.row_num
                    ),
                    []
                ) as failed_row_samples
            FROM ranked_results r
            LEFT JOIN limited_failed_rows trr
                ON r.test_result_id = trr.elementary_test_results_id
                AND trr.row_num <= 100
            WHERE r.row_num = 1
            GROUP BY
                r.test_id,
                r.test_name,
                r.model_unique_id,
                r.model_name,
                r.table_name,
                r.column_name,
                r.status,
                r.error_message,
                r.failed_rows,
                r.test_sub_type,
                r.test_short_name,
                r.severity,
                r.executed_at,
                r.test_description
        )
        SELECT
            test_id,
            test_name,
            model_unique_id,
            model_name,
            table_name,
            column_name,
            status,
            error_message,
            failed_rows,
            test_sub_type,
            test_short_name,
            severity,
            executed_at,
            test_description,
            failed_row_samples
        FROM results_with_rows
        ORDER BY executed_at DESC
        LIMIT {limit}
        """
        
        query_job = client.query(query)
        results = []
        
        import json
        
        for row in query_job:
            # Parse failed rows from JSON array
            failed_row_samples = []
            # failed_row_samples is already an array from ARRAY_AGG, or NULL/empty array
            if row.failed_row_samples:
                for result_row in row.failed_row_samples:
                    try:
                        # result_row is stored as JSON in the result_row column
                        if isinstance(result_row, str):
                            parsed_row = json.loads(result_row)
                        elif isinstance(result_row, dict):
                            parsed_row = result_row
                        elif hasattr(result_row, 'to_dict'):
                            # If it's a BigQuery JSON type, convert to dict
                            parsed_row = result_row.to_dict()
                        else:
                            parsed_row = {"value": str(result_row)}
                        
                        if isinstance(parsed_row, dict):
                            failed_row_samples.append(parsed_row)
                    except (json.JSONDecodeError, Exception):
                        # Skip invalid JSON
                        continue
            
            results.append({
                "test_id": row.test_id,
                "test_name": row.test_name,
                "model_name": row.model_name,
                "column_name": row.column_name,
                "table_name": row.table_name,
                "status": row.status,
                "error_message": row.error_message,
                "failed_rows": row.failed_rows or 0,
                "failed_row_samples": failed_row_samples,
                "test_sub_type": row.test_sub_type,
                "test_short_name": row.test_short_name,
                "severity": row.severity,
                "executed_at": row.executed_at.isoformat() if row.executed_at else None,
                "execution_time": 0.0,
                "test_description": getattr(row, 'test_description', None),
            })
        
        return {
            "status": "success",
            "message": f"Found {len(results)} test results from BigQuery",
            "data": {
                "results": results,
                "total_count": len(results),
                "filters_applied": {
                    "status": status,
                    "model_name": model_name,
                },
                "is_mock": False,
            }
        }
        
    except ImportError:
        return {
            "status": "error",
            "message": "google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery",
            "data": None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"BigQuery query failed: {str(e)}",
            "data": None
        }


def get_failed_tests() -> Dict[str, Any]:

    return tool_query_elementary(status="fail")


def get_test_details(test_id: str) -> Dict[str, Any]:

    result = tool_query_elementary()
    
    if result["status"] == "error":
        return result
    
    for test in result["data"]["results"]:
        if test["test_id"] == test_id:
            return {
                "status": "success",
                "message": f"Found test details for '{test_id}'",
                "data": test
            }
    
    return {
        "status": "error",
        "message": f"Test '{test_id}' not found",
        "data": None
    }


def _fetch_failed_rows_for_test(
    client: Any,
    test_id: str,
    settings: Any,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Fetch failed rows from test_result_rows table for a specific test.
    
    Note: This function is kept for backward compatibility and for use by get_failed_rows_for_test.
    The main query now uses a JOIN in _query_bigquery_results for better performance.
    
    The join uses: elementary_test_results.id = test_result_rows.elementary_test_results_id
    """
    try:
        import json
        from google.cloud import bigquery
        
        # First get the test_result_id (id) from elementary_test_results using test_unique_id
        # Then use that to query test_result_rows
        query = f"""
        SELECT
            trr.result_row
        FROM `{settings.bigquery_project_id}.{settings.bigquery_dataset}.test_result_rows` trr
        INNER JOIN `{settings.bigquery_project_id}.{settings.bigquery_dataset}.elementary_test_results` etr
            ON trr.elementary_test_results_id = etr.id
        WHERE etr.test_unique_id = @test_id
        ORDER BY trr.detected_at DESC
        LIMIT @limit
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("test_id", "STRING", test_id),
                bigquery.ScalarQueryParameter("limit", "INT64", limit),
            ]
        )
        
        query_job = client.query(query, job_config=job_config)
        failed_rows = []
        
        for row in query_job:
            result_row = row.result_row
            
            if result_row is None:
                continue
            
            # Handle different data types - result_row is stored as JSON string
            if isinstance(result_row, str):
                try:
                    parsed_row = json.loads(result_row)
                except json.JSONDecodeError:
                    # If it's not valid JSON, treat as a single value
                    parsed_row = {"value": result_row}
            elif hasattr(result_row, 'to_dict'):
                # If it's a BigQuery JSON type, convert to dict
                parsed_row = result_row.to_dict()
            elif isinstance(result_row, dict):
                # Already a dict, use as is
                parsed_row = result_row
            else:
                # Convert other types to dict
                parsed_row = {"value": str(result_row)}
            
            if isinstance(parsed_row, dict):
                failed_rows.append(parsed_row)
        
        return failed_rows
        
    except Exception as e:
        # If table doesn't exist or query fails, return empty list
        # This allows the code to continue working even if test_result_rows table is not available
        return []


def _fetch_rows_from_generated_query(
    client: Any,
    test_id: str,
    settings: Any,
    limit: int = 100
) -> List[Dict[str, Any]]:
    """Fetch failed rows by executing the compiled test query."""
    try:
        from google.cloud import bigquery
        
        # Try to get the compiled SQL from elementary_test_results first (most likely to have it)
        query = f"""
        SELECT test_results_query
        FROM `{settings.bigquery_project_id}.{settings.bigquery_dataset}.elementary_test_results`
        WHERE test_unique_id = @test_id
        AND test_results_query IS NOT NULL
        ORDER BY detected_at DESC
        LIMIT 1
        """
        
        job_config = bigquery.QueryJobConfig(
            query_parameters=[
                bigquery.ScalarQueryParameter("test_id", "STRING", test_id),
            ]
        )
        
        compiled_sql = None
        try:
            query_job = client.query(query, job_config=job_config)
            rows = list(query_job)
            if rows and rows[0].test_results_query:
                compiled_sql = rows[0].test_results_query
        except Exception as e:
            print(f"Error fetching from elementary_test_results: {e}")

        # If not found in elementary_test_results, try dbt_tests (legacy/fallback)
        if not compiled_sql:
            query = f"""
            SELECT *
            FROM `{settings.bigquery_project_id}.{settings.bigquery_dataset}.dbt_tests`
            WHERE unique_id = @test_id
            """
            try:
                query_job = client.query(query, job_config=job_config)
                rows = list(query_job)
                if rows:
                    row = rows[0]
                    candidates = ["compiled_code", "compiled_sql", "generated_test_query", "test_compiled_sql"]
                    for col in candidates:
                        if hasattr(row, col) and getattr(row, col):
                            compiled_sql = getattr(row, col)
                            break
                        elif col in row and row[col]:
                            compiled_sql = row[col]
                            break
            except Exception as e:
                pass
        
        if not compiled_sql:
            return []
        
        # Safety check: simplistic read-only check
        sql_upper = compiled_sql.upper()
        if "DROP" in sql_upper or "DELETE" in sql_upper or "UPDATE" in sql_upper or "INSERT" in sql_upper:
            return []
            
        # Add limit if not present
        if "LIMIT" not in sql_upper:
            compiled_sql = f"{compiled_sql.rstrip(';')} LIMIT {limit}"
            
        # Execute the compiled SQL
        fail_rows_job = client.query(compiled_sql)
        
        failed_rows = []
        for row in fail_rows_job:
            failed_rows.append(dict(row.items()))
            
        return failed_rows
        
    except Exception as e:
        print(f"Error executing fallback query: {e}")
        return []


def get_failed_rows_for_test(test_id: str, limit: int = 100) -> Dict[str, Any]:
    """Get failed rows for a specific test from BigQuery."""
    settings = get_settings()
    
    if settings.use_mock_data:
        # For mock data, return empty or use mock samples
        result = get_test_details(test_id)
        if result["status"] == "success":
            return {
                "status": "success",
                "message": f"Found {len(result['data'].get('failed_row_samples', []))} failed rows (mock mode)",
                "data": {
                    "failed_rows": result["data"].get("failed_row_samples", []),
                    "total_count": len(result["data"].get("failed_row_samples", [])),
                }
            }
        return {
            "status": "error",
            "message": f"Test '{test_id}' not found",
            "data": None
        }
    
    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account
        
        if settings.bigquery_credentials_path:
            credentials = service_account.Credentials.from_service_account_file(
                settings.bigquery_credentials_path,
                scopes=["https://www.googleapis.com/auth/bigquery"],
            )
            client = bigquery.Client(
                project=settings.bigquery_project_id,
                credentials=credentials
            )
        else:
            client = bigquery.Client(project=settings.bigquery_project_id)
        
        failed_rows = _fetch_failed_rows_for_test(client, test_id, settings, limit)
        
        # Check if we got useful rows. If not, trigger fallback to generated query.
        use_fallback = False
        if not failed_rows:
            use_fallback = True
        elif failed_rows:
            # Check if rows are just single values (expression: False, count, etc)
            # We prefer full row details if available via the query
            first_row = failed_rows[0]
            if len(first_row) <= 1:
                use_fallback = True
        
        if use_fallback:
            fallback_rows = _fetch_rows_from_generated_query(client, test_id, settings, limit)
            if fallback_rows:
                failed_rows = fallback_rows
        
        return {
            "status": "success",
            "message": f"Found {len(failed_rows)} failed rows for test '{test_id}'",
            "data": {
                "failed_rows": failed_rows,
                "total_count": len(failed_rows),
            }
        }
        
    except ImportError:
        return {
            "status": "error",
            "message": "google-cloud-bigquery not installed. Run: pip install google-cloud-bigquery",
            "data": None
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to fetch failed rows: {str(e)}",
            "data": None
        }


def get_failure_summary() -> Dict[str, Any]:

    result = get_failed_tests()
    
    if result["status"] == "error":
        return result
    
    failures = result["data"]["results"]
    
    # Group by severity
    by_severity = {"ERROR": [], "WARN": []}
    for f in failures:
        severity = f.get("severity", "ERROR")
        if severity in by_severity:
            by_severity[severity].append(f)
    
    # Group by model
    by_model = {}
    for f in failures:
        model = f.get("model_name", "unknown")
        if model not in by_model:
            by_model[model] = []
        by_model[model].append(f)
    
    return {
        "status": "success",
        "message": f"Summary: {len(failures)} total failures",
        "data": {
            "total_failures": len(failures),
            "by_severity": {k: len(v) for k, v in by_severity.items()},
            "by_model": {k: len(v) for k, v in by_model.items()},
            "models_affected": list(by_model.keys()),
            "critical_failures": by_severity["ERROR"],
        }
    }

