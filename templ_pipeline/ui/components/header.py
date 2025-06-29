"""
Header Component for TEMPL Pipeline

Renders the application header with branding and status.
"""

import streamlit as st
from ...config.constants import VERSION, COLORS
from ...core.session_manager import SessionManager
from ...config.settings import AppConfig


def render_header(config: AppConfig, session: SessionManager):
    """Render application header
    
    Args:
        config: Application configuration
        session: Session manager
    """
    # Custom CSS for header
    st.markdown(f"""
    <style>
    .header-container {{
        background: {COLORS['background']};
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.18);
        backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        border-radius: 1.2rem;
        border: 1px solid rgba(255, 255, 255, 0.18);
        text-align: center;
        padding: 2.5rem 0 1.5rem 0;
        margin-bottom: 1.5rem;
        animation: fadeIn 1.2s ease;
    }}
    
    .header-title {{
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-size: 2.8rem;
        letter-spacing: 0.04em;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #fff;
        text-shadow: 0 2px 12px {COLORS['secondary']}aa, 0 1px 0 #0003;
    }}
    
    .header-subtitle {{
        font-family: 'Segoe UI', 'Roboto', 'Arial', sans-serif;
        font-size: 1.18rem;
        font-weight: 400;
        color: {COLORS['text']};
        opacity: 0.92;
        margin: 0;
        letter-spacing: 0.02em;
    }}
    
    .header-version {{
        font-size: 0.9rem;
        color: {COLORS['text']};
        opacity: 0.7;
        margin-top: 0.5rem;
    }}
    
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-16px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """, unsafe_allow_html=True)
    
    # Render header
    st.markdown(f"""
    <div class="header-container">
        <div class="header-title">{config.app_name}</div>
        <div class="header-subtitle">{config.app_description}</div>
        <div class="header-version">Version {VERSION}</div>
    </div>
    """, unsafe_allow_html=True)
    

    
    # Show session info if in debug mode
    if config.ui_settings.get('show_technical_details', False):
        with st.expander("Session Information", expanded=False):
            session_info = session.get_session_info()
            st.json(session_info) 