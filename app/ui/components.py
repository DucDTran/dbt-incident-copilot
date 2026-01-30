"""Shared UI components for the dbt Co-Work dashboard."""

import streamlit as st
from datetime import datetime
from app.config import get_settings


def setup_app_styling():
    """Load fonts and apply global CSS styling. Call this at the start of each page."""
    # Load fonts and Material Icons
    st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">
    """, unsafe_allow_html=True)
    
    # Note: Full CSS is loaded in main.py, pages inherit it


def icon(name: str, size: int = 20) -> str:
    """Return HTML for a Material Icon with primary color."""
    return f'<span class="material-icons" style="font-size: {size}px; color: #FF683B; vertical-align: middle;">{name}</span>'


def icon_text(icon_name: str, text: str, size: int = 20) -> str:
    """Return HTML for icon + text inline."""
    return f'<span class="icon-text">{icon(icon_name, size)} {text}</span>'


def render_header():
    """Render the application header."""
    col1, col2 = st.columns([3, 1], vertical_alignment="center")
    
    with col1:
        st.markdown(
            f'<h1 style="font-family: Inter, sans-serif; font-weight: bold; margin: 0; color: #FF683B;">{icon("bolt", 48)}dbt Co-Work</h1>',
            unsafe_allow_html=True
        )
        st.caption("Agentic AI for Analytics Engineering Incident Resolution")
    
    with col2:
        st.markdown(
            f'<div style="text-align: right; padding-top: 8px;">'
            f'{icon("timelapse", 20)} <span style="color: #22c55e;">Last Synced</span>'
            f'</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="text-align: right; color: #888; font-size: 12px;">'
            f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
            f'</div>',
            unsafe_allow_html=True
        )


def render_sidebar():
    """Render the sidebar navigation."""
    settings = get_settings()
    
    with st.sidebar:
        # Active Project section with styled box
        st.markdown(f'{icon("folder_open", 18)} **Active Project**', unsafe_allow_html=True)
        project_html = f"""
        <div class="project-box">
            <div class="project-name">{settings.dbt_project_name}</div>
        </div>
        """
        st.markdown(project_html, unsafe_allow_html=True)
        
        st.divider()
        
        # Navigation
        st.markdown(f'{icon("menu", 18)} **Navigation**', unsafe_allow_html=True)
        
        # Navigation items with icons and descriptions
        nav_items = [
            ("Dashboard", "dashboard", "Active incidents and metrics"),
            ("Labs", "science", "Investigation workspace"),
            ("Snoozed", "snooze", "Hidden incidents")
        ]
        
        # Get current page from session state or default to Dashboard
        current_page = st.session_state.get("sidebar_nav", "Dashboard")
        
        # Create format function to display nav items with icons
        def format_nav_item(label):
            for nav_label, nav_icon, nav_desc in nav_items:
                if nav_label == label:
                    return f'{icon(nav_icon, 18)} {nav_label}'
            return label
        
        # Use native Streamlit radio buttons
        page = st.radio(
            "Select View",
            [item[0] for item in nav_items],
            label_visibility="collapsed",
            key="sidebar_nav",
            index=[item[0] for item in nav_items].index(current_page) if current_page in [item[0] for item in nav_items] else 0,
            format_func=format_nav_item
        )
        
        return page


def render_code_block(code: str, language: str = "sql") -> None:
    """Render a code block."""
    st.code(code, language=language, wrap_lines=True)


def render_diff_view(diff_text: str) -> None:
    """Render a diff view."""
    st.code(diff_text, language="diff", wrap_lines=True)
