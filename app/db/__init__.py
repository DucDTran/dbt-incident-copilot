"""
Database module for dbt Co-Work.
Contains mock data and database connection utilities.
"""

from .mock_elementary import MockElementaryDB, get_mock_test_results

__all__ = ["MockElementaryDB", "get_mock_test_results"]

