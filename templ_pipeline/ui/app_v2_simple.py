"""
TEMPL Pipeline Web Application - Version 2.0 Simplified

Simplified version to fix layout width issues by following the successful 
pattern from test_layout_fix.py - minimal interference, immediate fixes.
"""

import streamlit as st

# PHASE 1: IMMEDIATE PAGE CONFIG - No delays, no try/catch
st.set_page_config(
    page_title="TEMPL Pipeline",
    page_icon="🧪", 
    layout="wide"  # CRITICAL - Set immediately
)

# PHASE 2: IMMEDIATE LAYOUT FIXES - Apply before ANY other imports or content
from templ_pipeline.ui.ui.styles.early_layout import apply_layout_fixes
apply_layout_fixes()

# PHASE 3: Minimal imports after layout fixes are applied
import logging
import sys
from pathlib import Path
import traceback

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Configure minimal logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    """Simplified main application"""
    try:
        # Test that layout is working correctly
        st.title("TEMPL Pipeline")
        st.info("Layout Test: This layout should be full-width from first load")
        
        # Test with columns to verify width
        col1, col2, col3 = st.columns(3)
        with col1:
            st.success("Left column - should be full width immediately")
        with col2:
            st.info("Middle column - no narrow->wide transition")  
        with col3:
            st.warning("Right column - consistent width on first load")
        
        # Add divider
        st.divider()
        
        # Simple form
        st.header("Simple Test Form")
        
        smiles = st.text_input("SMILES", placeholder="Enter SMILES string")
        pdb_id = st.text_input("PDB ID", placeholder="Enter PDB ID")
        
        if st.button("Test Button", type="primary"):
            st.success(f"Input received: SMILES={smiles}, PDB={pdb_id}")
        
        # Wide element test
        st.code("""
This code block should span the full browser width from the moment the page loads.
If you see a narrow layout that then expands to full width, the fix needs more work.
The layout should be consistent on first load and after refresh.
        """)
        
        st.success("✅ If this layout looks the same on first load and after refresh, the fix is working!")
        
        # Debug info
        with st.expander("Debug Info"):
            st.write("Page config:", {
                "layout": "wide",
                "applied": True
            })
            st.write("Layout fixes applied:", True)
        
    except Exception as e:
        st.error(f"Error: {e}")
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
