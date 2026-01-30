"""
Mock Elementary database for demo and testing purposes.
Simulates the elementary_test_results table.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
import json
import random


@dataclass
class MockTestResult:
    """A mock test result record."""
    test_id: str
    test_name: str
    model_name: str
    column_name: Optional[str]
    status: str
    error_message: Optional[str]
    failed_rows: int
    failed_row_samples: List[Dict[str, Any]]
    test_type: str
    severity: str
    executed_at: datetime
    execution_time: float


class MockElementaryDB:
    """
    Mock Elementary database that simulates test results.
    Used for demo mode when BigQuery is not available.
    """
    
    def __init__(self):
        self._results: List[MockTestResult] = []
        self._initialize_mock_data()
    
    def _initialize_mock_data(self):
        """Initialize with realistic mock test failure data."""
        now = datetime.now()
        
        # Failure 1: New sentiment values (simulating NLP system upgrade)
        self._results.append(MockTestResult(
            test_id="test.airbnb_analytics.accepted_values_fact_reviews_sentiment__positive__neutral__negative",
            test_name="accepted_values_fact_reviews_sentiment__positive__neutral__negative",
            model_name="fact_reviews",
            column_name="sentiment",
            status="fail",
            error_message="Got 47 results, configured to fail if != 0. Found unexpected values: ['mixed', 'unknown']",
            failed_rows=47,
            failed_row_samples=[
                {"review_id": 98234, "sentiment": "mixed", "review_date": "2024-12-15", "reviewer_name": "Alex M."},
                {"review_id": 98456, "sentiment": "unknown", "review_date": "2024-12-16", "reviewer_name": "Jordan K."},
                {"review_id": 99012, "sentiment": "mixed", "review_date": "2024-12-18", "reviewer_name": "Taylor S."},
                {"review_id": 99234, "sentiment": "unknown", "review_date": "2024-12-19", "reviewer_name": "Casey R."},
            ],
            test_type="generic",
            severity="ERROR",
            executed_at=now - timedelta(minutes=15),
            execution_time=2.34,
        ))
        
        # Failure 2: NULL host_id (simulating migration)
        self._results.append(MockTestResult(
            test_id="test.airbnb_analytics.not_null_dim_listing_host_id",
            test_name="not_null_dim_listing_host_id",
            model_name="dim_listing",
            column_name="host_id",
            status="fail",
            error_message="Got 12 results, configured to fail if != 0. Found 12 NULL values in host_id column during host account migration.",
            failed_rows=12,
            failed_row_samples=[
                {"listing_id": 45231, "host_id": None, "listing_name": "Cozy Downtown Studio", "created_at": "2024-12-10"},
                {"listing_id": 45678, "host_id": None, "listing_name": "Beachfront Paradise", "created_at": "2024-12-11"},
                {"listing_id": 46012, "host_id": None, "listing_name": "Mountain View Cabin", "created_at": "2024-12-12"},
            ],
            test_type="generic",
            severity="ERROR",
            executed_at=now - timedelta(minutes=12),
            execution_time=1.87,
        ))
        
        # Failure 3: New room type (Studio)
        self._results.append(MockTestResult(
            test_id="test.airbnb_analytics.accepted_values_dim_listing_room_type__Entire_home_apt__Private_room__Shared_room__Hotel_room",
            test_name="accepted_values_dim_listing_room_type__Entire_home_apt__Private_room__Shared_room__Hotel_room",
            model_name="dim_listing",
            column_name="room_type",
            status="fail",
            error_message="Got 8 results, configured to fail if != 0. Found unexpected values: ['Studio']",
            failed_rows=8,
            failed_row_samples=[
                {"listing_id": 52341, "room_type": "Studio", "price": 89.00, "listing_name": "Modern Studio Downtown"},
                {"listing_id": 52567, "room_type": "Studio", "price": 95.00, "listing_name": "Artist's Studio Loft"},
                {"listing_id": 52789, "room_type": "Studio", "price": 110.00, "listing_name": "Minimalist City Studio"},
            ],
            test_type="generic",
            severity="ERROR",
            executed_at=now - timedelta(minutes=10),
            execution_time=1.56,
        ))
        
        # Failure 4: Price out of range
        self._results.append(MockTestResult(
            test_id="test.airbnb_analytics.values_in_range_dim_listing_price",
            test_name="values_in_range_dim_listing_price",
            model_name="dim_listing",
            column_name="price",
            status="fail",
            error_message="Got 3 results, configured to fail if != 0. Found prices outside allowed range (0.01-10000): [0.00, 15000.00, -50.00]",
            failed_rows=3,
            failed_row_samples=[
                {"listing_id": 61234, "price": 0.00, "listing_name": "First Night Free Promo", "host_id": 1001},
                {"listing_id": 61456, "price": 15000.00, "listing_name": "Ultra Luxury Penthouse Suite", "host_id": 1002},
                {"listing_id": 61789, "price": -50.00, "listing_name": "Data Import Error", "host_id": 1003},
            ],
            test_type="generic",
            severity="WARN",
            executed_at=now - timedelta(minutes=8),
            execution_time=0.98,
        ))
        
        # Failure 5: Orphan reviews (referential integrity)
        self._results.append(MockTestResult(
            test_id="test.airbnb_analytics.relationships_fact_reviews_listing_id__listing_id__ref_dim_listing_",
            test_name="relationships_fact_reviews_listing_id__listing_id__ref_dim_listing_",
            model_name="fact_reviews",
            column_name="listing_id",
            status="fail",
            error_message="Got 23 results, configured to fail if != 0. Found 23 orphan records with listing_id not in dim_listing. These may be reviews for deleted listings.",
            failed_rows=23,
            failed_row_samples=[
                {"review_id": 112233, "listing_id": 99999, "reviewer_name": "John D.", "review_date": "2024-11-01"},
                {"review_id": 112456, "listing_id": 88888, "reviewer_name": "Sarah M.", "review_date": "2024-11-15"},
                {"review_id": 112789, "listing_id": 77777, "reviewer_name": "Mike L.", "review_date": "2024-12-01"},
            ],
            test_type="generic",
            severity="ERROR",
            executed_at=now - timedelta(minutes=5),
            execution_time=3.21,
        ))
    
    def get_all_results(self) -> List[Dict[str, Any]]:
        """Get all test results as dictionaries."""
        return [self._result_to_dict(r) for r in self._results]
    
    def get_failed_results(self) -> List[Dict[str, Any]]:
        """Get only failed test results."""
        failed = [r for r in self._results if r.status == "fail"]
        return [self._result_to_dict(r) for r in failed]
    
    def get_results_by_model(self, model_name: str) -> List[Dict[str, Any]]:
        """Get test results for a specific model."""
        filtered = [r for r in self._results if r.model_name == model_name]
        return [self._result_to_dict(r) for r in filtered]
    
    def get_result_by_id(self, test_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific test result by ID."""
        for r in self._results:
            if r.test_id == test_id:
                return self._result_to_dict(r)
        return None
    
    def add_result(self, result: MockTestResult):
        """Add a new test result."""
        self._results.append(result)
    
    def clear_results(self):
        """Clear all test results."""
        self._results = []
    
    def _result_to_dict(self, result: MockTestResult) -> Dict[str, Any]:
        """Convert a MockTestResult to dictionary."""
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
            "execution_time": result.execution_time,
        }


# Global mock database instance
_mock_db: Optional[MockElementaryDB] = None


def get_mock_db() -> MockElementaryDB:
    """Get or create the mock database instance."""
    global _mock_db
    if _mock_db is None:
        _mock_db = MockElementaryDB()
    return _mock_db


def get_mock_test_results(
    status: Optional[str] = None,
    model_name: Optional[str] = None,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get mock test results with optional filtering.
    
    Args:
        status: Filter by status ('pass', 'fail', etc.)
        model_name: Filter by model name
        limit: Maximum results to return
        
    Returns:
        List of test result dictionaries.
    """
    db = get_mock_db()
    
    if status == "fail":
        results = db.get_failed_results()
    elif model_name:
        results = db.get_results_by_model(model_name)
    else:
        results = db.get_all_results()
    
    # Apply additional filters
    if status and status != "fail":
        results = [r for r in results if r["status"] == status]
    
    if model_name:
        results = [r for r in results if r["model_name"] == model_name]
    
    return results[:limit]

