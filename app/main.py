"""
dbt Co-Work - Main Streamlit Application

An Agentic AI Platform for Analytics Engineering Incident Resolution.
"""

import streamlit as st

# Page configuration - MUST be first Streamlit command
st.set_page_config(
    page_title="dbt Co-Work",
    page_icon=":material/bolt:",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/dbt-copilot",
        "Report a bug": "https://github.com/your-org/dbt-copilot/issues",
        "About": "# dbt Co-Work\nAgentic AI for Analytics Engineering"
    }
)

# Now do remaining imports
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app.ui.components import render_header
from app.config import get_settings
from app.ui.mission_control import render_mission_control
from app.ui.resolution_studio import render_resolution_studio, render_fixes_page
from app.ui.snoozed import render_snoozed_page

# Load fonts and Material Icons
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Rounded:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200" rel="stylesheet">
""", unsafe_allow_html=True)

# Custom CSS
st.markdown("""
<style>
/* Inter font for all text */
html, body, [class*="css"], .stMarkdown, p, div, label {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* Inter font for headings with proper weight */
h1, h2, h3, h4, h5, h6, 
[data-testid="stHeadingWithActionElements"],
.stTitle, .stHeader {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    font-weight: 600 !important;
}

/* JetBrains Mono for code */
code, pre, .stCodeBlock, [data-testid="stJson"] {
    font-family: 'JetBrains Mono', monospace !important;
}

/* Theme-aware background - use Streamlit's theme variables when available */
.stApp {
    /* Dark mode default */
    background-color: #0d0d1a;
}

[data-testid="stMainBlockContainer"] {
    padding-top: 1rem !important;
}

/* Light mode - override when Streamlit theme is light */
.stApp[data-theme="light"],
[data-theme="light"] .stApp {
    background-color: #ffffff;
}

/* Use CSS media query as fallback */
@media (prefers-color-scheme: light) {
    .stApp:not([data-theme="dark"]) {
        background-color: #ffffff;
    }
}

/* Primary color for borders on containers */
[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #FF683B !important;
}

/* Bordered container borders - for metrics cards and all st.container(border=True) */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #FF683B !important;
}

/* Expander border color */
.streamlit-expanderHeader {
    border-color: #FF683B !important;
}

[data-testid="stExpander"] {
    border-color: #FF683B !important;
}

/* Container borders */
[data-testid="stHorizontalBlock"] [data-testid="stVerticalBlockBorderWrapper"],
.stContainer > div {
    border-color: #FF683B !important;
}

/* Primary button styling */
.stButton > button[kind="primary"] {
    background-color: #FF683B;
    border-color: #FF683B;
    color: #ffffff;
}
.stButton > button[kind="primary"]:hover {
    background-color: #FF8A5B;
    border-color: #FF8A5B;
}

/* Secondary button with primary border */
.stButton > button {
    border-color: rgba(255, 104, 59, 0.5) !important;
}
.stButton > button:hover {
    border-color: #FF683B !important;
}

/* Metric styling */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    color: #FF683B;
    font-size: 48px !important;
    font-weight: 100 !important;
    line-height: 1.2 !important;
}

/* Material Icons styling */
.material-icons, .material-symbols-rounded {
    font-family: 'Material Icons', 'Material Symbols Rounded' !important;
    color: #FF683B;
    vertical-align: middle;
}

/* Icon inline with text */
.icon-text {
    display: inline-flex;
    align-items: center;
    gap: 8px;
}

.icon-text .material-icons,
.icon-text .material-symbols-rounded {
    font-size: 20px;
}

/* Sidebar styling - adapts to theme */
[data-testid="stSidebar"] {
    border-right: 1px solid rgba(255, 104, 59, 0.2);
}

/* Light mode sidebar */
[data-theme="light"] [data-testid="stSidebar"],
[data-testid="stSidebar"][data-theme="light"] {
    border-right: 1px solid rgba(255, 104, 59, 0.3);
}

/* Light mode sidebar via media query */
@media (prefers-color-scheme: light) {
    [data-testid="stSidebar"]:not([data-theme="dark"]) {
        border-right: 1px solid rgba(255, 104, 59, 0.3);
    }
}

/* Info box with primary accent */
[data-testid="stAlert"] {
    border-left-color: #FF683B !important;
}

/* Status container borders */
[data-testid="stStatusWidget"] {
    border-color: rgba(255, 104, 59, 0.3) !important;
}

/* Text color adjustments - let Streamlit handle most text colors */
/* Only override where necessary for theme compatibility */

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}

/* Active Project box styling */
.project-box {
    padding: 12px 16px;
    margin: 8px 0;
    border-radius: 10px;
    border: 2px solid rgba(255, 104, 59, 0.3);
    background-color: rgba(255, 104, 59, 0.08);
}

.project-name {
    font-size: 14px;
    font-weight: 500;
    color: inherit;
}

/* Navigation styling for st.navigation() */
[data-testid="stSidebar"] [data-testid="stNavigation"] {
    margin-top: 16px;
}

[data-testid="stSidebar"] [data-testid="stNavigation"] > div {
    display: flex;
    flex-direction: column;
    gap: 4px;
}

[data-testid="stSidebar"] [data-testid="stNavigation"] button {
    display: flex;
    align-items: center;
    padding: 12px 16px;
    margin: 0;
    border-radius: 10px;
    border: 2px solid transparent;
    background-color: transparent;
    cursor: pointer;
    transition: background-color 0.2s, border-color 0.2s;
    width: 100%;
    box-sizing: border-box;
    gap: 12px;
    text-align: left;
    justify-content: flex-start;
}

[data-testid="stSidebar"] [data-testid="stNavigation"] button:hover {
    background-color: rgba(255, 104, 59, 0.08);
    border-color: rgba(255, 104, 59, 0.2);
}

[data-testid="stSidebar"] [data-testid="stNavigation"] button[kind="primary"] {
    background-color: rgba(255, 104, 59, 0.12);
    border-color: #FF683B;
    color: #FF683B;
    font-weight: 600;
}

[data-testid="stSidebar"] [data-testid="stNavigation"] button[kind="primary"]:hover {
    background-color: rgba(255, 104, 59, 0.15);
}

/* Light mode adjustments */
[data-theme="light"] .project-box {
    background-color: rgba(255, 104, 59, 0.06);
    border-color: rgba(255, 104, 59, 0.3);
}

[data-theme="light"] .nav-item:hover {
    background-color: rgba(255, 104, 59, 0.06);
}

[data-theme="light"] .nav-item-selected {
    background-color: rgba(255, 104, 59, 0.1);
}

[data-theme="light"] .nav-item-selected:hover {
    background-color: rgba(255, 104, 59, 0.12);
}

[data-theme="light"] .nav-desc {
    color: #666;
}

@media (prefers-color-scheme: light) {
    .stApp:not([data-theme="dark"]) .project-box {
        background-color: rgba(255, 104, 59, 0.06);
        border-color: rgba(255, 104, 59, 0.3);
    }
    
    .stApp:not([data-theme="dark"]) .nav-item:hover {
        background-color: rgba(255, 104, 59, 0.06);
    }
    
    .stApp:not([data-theme="dark"]) .nav-item-selected {
        background-color: rgba(255, 104, 59, 0.1);
    }
    
    .stApp:not([data-theme="dark"]) .nav-item-selected:hover {
        background-color: rgba(255, 104, 59, 0.12);
    }
    
    .stApp:not([data-theme="dark"]) .nav-desc {
        color: #666;
    }
}

/* Hide radio buttons visually but keep functionality */
[data-testid="stSidebar"] div[data-testid="stRadio"] {
    position: absolute;
    opacity: 0;
    pointer-events: none;
    height: 0;
    width: 0;
    overflow: hidden;
    margin: 0;
    padding: 0;
}
</style>
""", unsafe_allow_html=True)


def main():
    """Main application entry point."""
    
    # Initialize session state
    if "selected_incident" not in st.session_state:
        st.session_state.selected_incident = None
    
    # Initialize page state for Labs sub-navigation
    if "page" not in st.session_state:
        st.session_state.page = "resolution"  # Default to investigation view
    
    # Render sidebar with Active Project info
    with st.sidebar:
        from app.ui.components import icon
        settings = get_settings()
        st.markdown(f'{icon("folder_open", 18)} **Active Project**', unsafe_allow_html=True)
        project_html = f"""
        <div class="project-box">
            <div class="project-name">{settings.dbt_project_name}</div>
        </div>
        """
        st.markdown(project_html, unsafe_allow_html=True)
    
    # Define pages using the existing render functions
    def dashboard_page():
        render_header()
        render_mission_control()
    
    def labs_page():
        render_header()
        # Check if we should show fixes page or resolution studio
        # Ensure page is initialized if not set
        if "page" not in st.session_state:
            st.session_state.page = "resolution"
        
        # Show fixes page only if explicitly set to "fixes"
        if st.session_state.page == "fixes":
            render_fixes_page()
        else:
            # Default to resolution studio (investigation view)
            render_resolution_studio()
    
    def snoozed_page():
        render_header()
        render_snoozed_page()
    
    # Create mapping of page titles to functions
    page_functions = {
        "Dashboard": dashboard_page,
        "Labs": labs_page,
        "Snoozed": snoozed_page,
    }
    
    # Track current page in session state - initialize if not set
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Dashboard"
    
    # Check if we need to navigate programmatically (e.g., from Investigate button)
    if "sidebar_nav" in st.session_state:
        nav_request = st.session_state.sidebar_nav
        if nav_request in page_functions:
            st.session_state.current_page = nav_request
            # Clear the navigation request after using it
            del st.session_state.sidebar_nav
    
    # Store page before navigation to detect user clicks
    page_before_nav = st.session_state.current_page
    
    # Create pages list - use current_page from session state as default
    current_page = st.session_state.current_page
    pages = [
        st.Page(dashboard_page, title="Dashboard", icon=":material/dashboard:", default=(current_page == "Dashboard")),
        st.Page(labs_page, title="Labs", icon=":material/science:", default=(current_page == "Labs")),
        st.Page(snoozed_page, title="Snoozed", icon=":material/snooze:", default=(current_page == "Snoozed")),
    ]
    
    # Use Streamlit's native navigation
    selected_page = st.navigation(pages, position="sidebar", expanded=False)
    
    # Get the title from selected_page if available
    selected_title = None
    if selected_page:
        selected_title = getattr(selected_page, 'title', None)
        # Only update current_page if user explicitly clicked a different tab
        # Compare with page_before_nav to detect actual user interaction
        if selected_title and selected_title != page_before_nav:
            st.session_state.current_page = selected_title
    
    # Always use current_page from session state to determine which page to show
    # This ensures we maintain the page across reruns (e.g., after Start Investigation calls st.rerun())
    page_to_show = st.session_state.current_page
    
    # Safety check: if we have a selected_incident, we should be on Labs page
    # Don't let navigation reset us to Dashboard if we're in the middle of an investigation
    if "selected_incident" in st.session_state and st.session_state.selected_incident is not None:
        if page_to_show == "Dashboard" and page_before_nav == "Labs":
            # We were on Labs, have an incident, but navigation reset us - force back to Labs
            page_to_show = "Labs"
            st.session_state.current_page = "Labs"
    
    # Execute the page function - this is what actually renders the page
    if page_to_show and page_to_show in page_functions:
        page_functions[page_to_show]()
    else:
        # Fallback to dashboard if something went wrong
        page_functions["Dashboard"]()


if __name__ == "__main__":
    main()
