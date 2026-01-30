"""Snoozed - View and manage snoozed incidents."""

import streamlit as st
from datetime import datetime, timedelta
from typing import Dict, Any, List

from app.ui.components import icon


def get_snoozed_items() -> List[Dict[str, Any]]:
    """Get all Snoozed from session state."""
    if "snoozed_items" not in st.session_state:
        st.session_state.snoozed_items = []
    return st.session_state.snoozed_items


def add_snoozed_item(incident: Dict[str, Any], duration_hours: int = 24):
    """Add an incident to Snoozed."""
    if "snoozed_items" not in st.session_state:
        st.session_state.snoozed_items = []
    
    snoozed_item = {
        **incident,
        "snoozed_at": datetime.now().isoformat(),
        "snooze_until": (datetime.now() + timedelta(hours=duration_hours)).isoformat(),
        "duration_hours": duration_hours,
    }
    
    # Remove if already exists
    st.session_state.snoozed_items = [
        item for item in st.session_state.snoozed_items 
        if item.get("test_id") != incident.get("test_id")
    ]
    
    st.session_state.snoozed_items.append(snoozed_item)


def remove_snoozed_item(test_id: str):
    """Remove an item from snoozed list."""
    if "snoozed_items" in st.session_state:
        st.session_state.snoozed_items = [
            item for item in st.session_state.snoozed_items 
            if item.get("test_id") != test_id
        ]


def is_snoozed(test_id: str) -> bool:
    """Check if a test is currently snoozed."""
    snoozed_items = get_snoozed_items()
    for item in snoozed_items:
        if item.get("test_id") == test_id:
            try:
                snooze_until = datetime.fromisoformat(item.get("snooze_until", ""))
                if datetime.now() < snooze_until:
                    return True
            except:
                pass
    return False


def render_snoozed_page():
    """Render the Snoozed page."""
    
    st.markdown(f'<h2 style="font-family: Inter, sans-serif;">{icon("snooze", 24)} Snoozed</h2>', unsafe_allow_html=True)
    
    snoozed_items = get_snoozed_items()
    
    # Filter out expired snoozes
    active_snoozed = []
    for item in snoozed_items:
        try:
            snooze_until = datetime.fromisoformat(item.get("snooze_until", ""))
            if datetime.now() < snooze_until:
                active_snoozed.append(item)
        except:
            pass
    
    # Update session state with only active Snoozed
    st.session_state.snoozed_items = active_snoozed
    
    if not active_snoozed:
        st.info("No Snoozed. Snoozed incidents will appear here.")
        return
    
    st.caption(f"{len(active_snoozed)} item(s) currently snoozed")
    st.divider()
    
    for idx, item in enumerate(active_snoozed):
        test_name = item.get("test_name", "Unknown Test")
        model_name = item.get("model_name", "Unknown Model")
        column_name = item.get("column_name", "")
        snooze_until = item.get("snooze_until", "")
        
        # Calculate time remaining
        try:
            until = datetime.fromisoformat(snooze_until)
            remaining = until - datetime.now()
            hours_left = int(remaining.total_seconds() // 3600)
            mins_left = int((remaining.total_seconds() % 3600) // 60)
            time_remaining = f"{hours_left}h {mins_left}m remaining"
        except:
            time_remaining = "Unknown"
        
        # Display name
        display_name = test_name[:57] + "..." if len(test_name) > 60 else test_name
        
        with st.container(border=True):
            st.markdown(f"**{display_name}**")
            
            model_display = model_name
            if column_name:
                model_display += f" / {column_name}"
            st.caption(model_display)
            
            st.warning(f"Snoozed: {time_remaining}")
            
            if st.button("Unsnooze", key=f"unsnooze_{idx}"):
                remove_snoozed_item(item.get("test_id"))
                st.rerun()
