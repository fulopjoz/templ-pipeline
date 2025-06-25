import streamlit as st
import sys
from pathlib import Path

# Add project root to path
current_file_path = Path(__file__).resolve()
project_root = current_file_path.parent
sys.path.insert(0, str(project_root))

def debug_main():
    st.title("Debug Test")
    st.write("If you see this, basic Streamlit is working")
    
    try:
        st.write("Testing imports...")
        from templ_pipeline.ui.app import initialize_session_state
        st.success("✓ Import successful")
        
        st.write("Testing session state...")
        initialize_session_state()
        st.success("✓ Session state initialized")
        
    except Exception as e:
        st.error(f"Error: {e}")
        import traceback
        st.code(traceback.format_exc())

debug_main()
