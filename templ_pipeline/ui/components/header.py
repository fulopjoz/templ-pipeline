# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Header Component for TEMPL Pipeline

Renders the application header with branding and status.
"""

import streamlit as st

from ..config.constants import COLORS, VERSION
from ..config.settings import AppConfig
from ..core.session_manager import SessionManager


def render_header(config: AppConfig, session: SessionManager):
    """Render application header

    Args:
        config: Application configuration
        session: Session manager
    """
    # Custom CSS for header
    st.markdown(
        f"""
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

    .header-links {{
        margin-top: 0.8rem;
        display: flex;
        justify-content: center;
        align-items: center;
        gap: 1rem;
    }}

    .github-link {{
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.4rem 0.8rem;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-radius: 0.5rem;
        color: {COLORS['text']};
        text-decoration: none;
        font-size: 0.85rem;
        font-weight: 500;
        transition: all 0.3s ease;
        backdrop-filter: blur(4px);
    }}

    .github-link:hover {{
        background: rgba(255, 255, 255, 0.15);
        border-color: rgba(255, 255, 255, 0.3);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
        text-decoration: none;
        color: {COLORS['text']};
    }}

    .github-icon {{
        width: 16px;
        height: 16px;
        opacity: 0.8;
    }}

    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(-16px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    # Render header
    st.markdown(
        f"""
    <div class="header-container">
        <div class="header-title">{config.app_name}</div>
        <div class="header-subtitle">{config.app_description}</div>
        <div class="header-version">Version {VERSION}</div>
        <div class="header-links">
            <a href="https://github.com/fulopjoz/templ-pipeline" target="_blank" class="github-link">
                <svg class="github-icon" viewBox="0 0 24 24" fill="currentColor">
                    <path d=(
                        "M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387"
                        ".599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416"
                        "-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745"
                        ".083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834"
                        " 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305"
                        "-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124"
                        "-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266"
                        " 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552"
                        " 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84"
                        " 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43"
                        " .372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765"
                        "-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"
                    )/>
                </svg>
                Repository
            </a>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Show session info if in debug mode
    if config.ui_settings.get("show_technical_details", False):
        with st.expander("Session Information", expanded=False):
            session_info = session.get_session_info()
            st.json(session_info)
