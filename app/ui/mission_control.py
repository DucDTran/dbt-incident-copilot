"""Dashboard - Home view showing active incidents."""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.agent.tools import get_failed_tests, is_incident_fixed
from app.ui.snoozed import is_snoozed, add_snoozed_item
from app.ui.components import icon



def start_investigation(incident: Dict[str, Any]):
    """Callback to start investigation."""
    st.session_state.selected_incident = incident
    st.session_state.page = "resolution"
    st.session_state.sidebar_nav = "Labs"
    # Clear previous action results when switching incidents
    st.session_state.preview_impact_result = None
    st.session_state.run_test_result = None
    st.session_state.dry_run_result = None
    st.session_state.show_diff = False


def render_mission_control():
    """Render the Dashboard (home) view."""
    
    # Fetch failed tests
    result = get_failed_tests()
    
    if result["status"] == "error":
        st.error(f"Failed to fetch test results: {result['message']}")
        return
    
    failures = result["data"]["results"]
    
    # Filter out Snoozed AND fixed incidents
    active_failures = [
        f for f in failures 
        if not is_snoozed(f.get("test_id", "")) and not is_incident_fixed(f.get("test_id", ""))
    ]
    
    # Render stats
    render_stats(active_failures)
    
    st.divider()
    
    # Render incidents list
    st.markdown(f'<h2 style="font-family: Inter, sans-serif;">{icon("crisis_alert", 24)} Active Incidents</h2>', unsafe_allow_html=True)
    
    if not active_failures:
        st.success("No active incidents. All tests passing.")
        return
    
    # Sort by severity (ERROR first) then by time
    failures_sorted = sorted(
        active_failures,
        key=lambda x: (0 if x.get("severity") == "ERROR" else 1, x.get("executed_at", "")),
        reverse=True
    )
    
    for idx, failure in enumerate(failures_sorted):
        render_incident_card(failure, idx)


def render_stats(failures: List[Dict[str, Any]]):
    """Render the statistics with bordered cards."""
    
    total_failures = len(failures)
    error_count = sum(1 for f in failures if f.get("severity") == "ERROR")
    warn_count = sum(1 for f in failures if f.get("severity") == "WARN")
    models_affected = len(set(f.get("model_name") for f in failures))
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        with st.container(border=True):
            st.markdown(f'{icon("bug_report", 18)} Total Failures', unsafe_allow_html=True)
            st.metric("Total Failures", total_failures, label_visibility="collapsed")
    
    with col2:
        with st.container(border=True):
            st.markdown(f'{icon("error", 18)} Critical', unsafe_allow_html=True)
            st.metric("Critical", error_count, label_visibility="collapsed")
    
    with col3:
        with st.container(border=True):
            st.markdown(f'{icon("warning", 18)} Warnings', unsafe_allow_html=True)
            st.metric("Warnings", warn_count, label_visibility="collapsed")
    
    with col4:
        with st.container(border=True):
            st.markdown(f'{icon("table_chart", 18)} Models Affected', unsafe_allow_html=True)
            st.metric("Models Affected", models_affected, label_visibility="collapsed")


def render_incident_card(failure: Dict[str, Any], idx: int):
    
    severity = failure.get("severity", "ERROR")
    test_name = failure.get("test_name", "Unknown Test")
    model_name = failure.get("model_name", "Unknown Model")
    column_name = failure.get("column_name", "")
    error_message = failure.get("error_message", "")[:100]
    failed_rows = failure.get("failed_rows", 0)
    executed_at = failure.get("executed_at", "")
    test_sub_type = failure.get("test_sub_type", "")
    test_short_name = failure.get("test_short_name", "")
    
    # Calculate time since failure
    time_ago = "Unknown"
    if executed_at:
        try:
            exec_time = datetime.fromisoformat(executed_at.replace('Z', '+00:00'))
            time_diff = datetime.now(exec_time.tzinfo) - exec_time if exec_time.tzinfo else datetime.now() - exec_time
            time_ago = format_time_ago(time_diff)
        except:
            pass
    
    # Truncate test name
    display_name = test_name
    
    # Create bordered container
    with st.container(border=True):
        # Header row
        col_title, col_, col_severity = st.columns([3, 1, 1])

        with col_title:
            st.markdown(f"**{display_name}**")
            model_display = f"Table Name: {model_name}"
            if column_name:
                model_display += f"\n\nColumn Name: {column_name}"
            st.caption(model_display)
        
        with col_severity:
            if severity == "ERROR":
                st.error(severity, icon=":material/error:")
            else:
                st.warning(severity, icon=":material/warning:")
        
            # Meta info
            st.caption(f"{icon('format_list_numbered_rtl', 16)} {failed_rows} failed rows", unsafe_allow_html=True)
            st.caption(f"{icon('timelapse', 16)} {time_ago}", unsafe_allow_html=True)
            st.caption(f"{icon('label', 16)} {test_sub_type.capitalize()} | {test_short_name}", unsafe_allow_html=True)
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            st.button("Investigate", key=f"investigate_{idx}", on_click=start_investigation, args=(failure,), icon=':material/content_paste_search:', type="primary")
        
        with col2:
            if st.button("Snooze", key=f"snooze_{idx}", icon=':material/snooze:'):
                add_snoozed_item(failure, duration_hours=24)
                st.toast("Snoozed for 24 hours")
                st.rerun()


def format_time_ago(delta: timedelta) -> str:
    """Format a timedelta as a human-readable string."""
    seconds = int(delta.total_seconds())
    
    if seconds < 60:
        return f"{seconds}s ago"
    elif seconds < 3600:
        return f"{seconds // 60}m ago"
    elif seconds < 86400:
        return f"{seconds // 3600}h ago"
    else:
        return f"{seconds // 86400}d ago"
