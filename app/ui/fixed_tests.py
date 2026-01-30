"""Resolved - View all tests that have been successfully fixed."""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, List

from app.ui.components import icon


def get_fixed_tests() -> List[Dict[str, Any]]:
    """Get all Resolved from session state."""
    if "fixed_tests" not in st.session_state:
        st.session_state.fixed_tests = []
    return st.session_state.fixed_tests


def add_fixed_test(incident: Dict[str, Any], fix_option: Dict[str, Any] = None):
    """Add a test to the Resolved list."""
    if "fixed_tests" not in st.session_state:
        st.session_state.fixed_tests = []
    
    # Check if test is already in the list
    test_id = incident.get("test_id")
    if test_id:
        # Remove if already exists (to update with latest fix info)
        st.session_state.fixed_tests = [
            item for item in st.session_state.fixed_tests 
            if item.get("test_id") != test_id
        ]
    
    fixed_test = {
        **incident,
        "fixed_at": datetime.now().isoformat(),
        "fix_option": fix_option,
    }
    
    st.session_state.fixed_tests.append(fixed_test)


def remove_fixed_test(test_id: str):
    """Remove a test from the fixed list."""
    if "fixed_tests" in st.session_state:
        st.session_state.fixed_tests = [
            item for item in st.session_state.fixed_tests 
            if item.get("test_id") != test_id
        ]


def is_fixed(test_id: str) -> bool:
    """Check if a test has been fixed."""
    fixed_tests = get_fixed_tests()
    return any(item.get("test_id") == test_id for item in fixed_tests)


def render_fixed_tests_page():
    """Render the Resolved page."""
    
    st.markdown(f'<h2 style="font-family: Inter, sans-serif;">{icon("check_circle", 24)} Resolved</h2>', unsafe_allow_html=True)
    
    fixed_tests = get_fixed_tests()
    
    if not fixed_tests:
        st.info("No Resolved yet. Tests that have been marked Verified will appear here.")
        return
    
    st.caption(f"{len(fixed_tests)} test(s) have been fixed")
    st.divider()
    
    # Sort by fixed_at (most recent first)
    sorted_tests = sorted(
        fixed_tests, 
        key=lambda x: x.get("fixed_at", ""), 
        reverse=True
    )
    
    for idx, test in enumerate(sorted_tests):
        test_name = test.get("test_name", "Unknown Test")
        model_name = test.get("model_name", "Unknown Model")
        column_name = test.get("column_name", "")
        fixed_at = test.get("fixed_at", "")
        fix_option = test.get("fix_option", {})
        
        # Format fixed date
        try:
            fixed_datetime = datetime.fromisoformat(fixed_at)
            fixed_date_str = fixed_datetime.strftime("%Y-%m-%d %H:%M:%S")
            time_ago = _get_time_ago(fixed_datetime)
        except:
            fixed_date_str = "Unknown"
            time_ago = ""
        
        # Display name
        display_name = test_name[:57] + "..." if len(test_name) > 60 else test_name
        
        with st.container(border=True):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"**{display_name}**")
                
                model_display = model_name
                if column_name:
                    model_display += f" / {column_name}"
                st.caption(model_display)
                
                # Show fix details if available
                if fix_option:
                    fix_title = fix_option.get("title", "Unknown Fix")
                    fix_type = fix_option.get("fix_type", "")
                    st.caption(f"Fix: {fix_title} ({fix_type})")
                
                st.success(f"Fixed {time_ago} ({fixed_date_str})", icon=':material/check_circle:')
            
            with col2:
                if st.button("Remove", key=f"remove_{idx}", use_container_width=True):
                    remove_fixed_test(test.get("test_id"))
                    st.rerun()
        
        # Add divider between items (but not after the last one)
        if idx < len(sorted_tests) - 1:
            st.divider()


def _get_time_ago(dt: datetime) -> str:
    """Get a human-readable time ago string."""
    now = datetime.now()
    diff = now - dt
    
    if diff.days > 0:
        return f"{diff.days} day{'s' if diff.days > 1 else ''} ago"
    elif diff.seconds >= 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif diff.seconds >= 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "just now"
