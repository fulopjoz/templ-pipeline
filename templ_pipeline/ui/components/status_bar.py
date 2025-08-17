# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Status Bar Component for TEMPL Pipeline

Displays application status and session information.
"""

import logging
from datetime import datetime, timedelta

import streamlit as st

from ..core.session_manager import SessionManager

logger = logging.getLogger(__name__)


def render_status_bar(session: SessionManager):
    """Render the status bar at the bottom of the application

    Args:
        session: Session manager instance
    """
    # Create a container for the status bar
    st.markdown("---")

    # Get session info
    session_info = session.get_session_info()

    # Create columns for status items
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        # Session duration
        if session_info.get("start_time"):
            duration = session_info.get("duration_seconds", 0)
            duration_str = str(timedelta(seconds=int(duration)))
            st.caption(f"Session: {duration_str}")
        else:
            st.caption("Session: Just started")

    with col2:
        # Pipeline runs
        runs = session_info.get("pipeline_runs", 0)
        st.caption(f"Runs: {runs}")

    with col3:
        # Memory usage
        memory_stats = session_info.get("memory_stats", {})
        if memory_stats:
            cache_mb = memory_stats.get("cache_size_mb", 0)
            st.caption(f"Cache: {cache_mb:.1f}MB")
        else:
            st.caption("Cache: 0MB")

    with col4:
        # Status indicator
        if session_info.get("has_results"):
            st.caption("Results ready")
        elif session_info.get("has_input"):
            st.caption("Ready to run")
        else:
            st.caption("Awaiting input")

    # Add a subtle footer
    st.caption(
        f"TEMPL Pipeline | " f"Session ID: {session_info.get('session_id', 'N/A')}"
    )
