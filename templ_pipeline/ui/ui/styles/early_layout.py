"""
Early Layout CSS Module for TEMPL Pipeline - IMMEDIATE FIX VERSION

Critical CSS and JavaScript injection to prevent layout timing issues.
This version applies fixes IMMEDIATELY without any delays to ensure
proper layout from the very first load, preventing the narrow->wide transition.
"""

import streamlit as st


def inject_immediate_viewport_fix():
    """Inject viewport meta tag and immediate critical CSS
    
    This MUST be called first to set proper viewport behavior.
    """
    st.markdown("""
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover">
    <style>
    /* ===== IMMEDIATE LAYOUT FIXES - NO DELAYS ===== */
    /* Maximum specificity CSS to override Streamlit defaults IMMEDIATELY */
    
    /* Critical: Force immediate full width for all containers */
    html, body {
        width: 100vw !important;
        max-width: 100vw !important;
        overflow-x: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Streamlit app container - immediate full width */
    .stApp,
    div.stApp,
    div[data-testid="stApp"] {
        width: 100vw !important;
        max-width: 100vw !important;
        min-width: 100vw !important;
    }
    
    /* Main section containers - immediate full width */
    section.main,
    div[data-testid="stAppViewContainer"],
    div[data-testid="stAppViewContainer"] > section,
    div[data-testid="stAppViewContainer"] > section.main {
        width: 100% !important;
        max-width: 100% !important;
        min-width: 100% !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Block containers - immediate responsive width */
    section.main > div.block-container,
    div.block-container,
    section[data-testid="stMainBlockContainer"],
    div[data-testid="stMainBlockContainer"],
    .main > div.block-container,
    div[data-testid="stAppViewContainer"] > section.main > div,
    div[data-testid="stAppViewContainer"] > section > div.block-container {
        max-width: none !important;
        width: 100% !important;
        min-width: 100% !important;
        padding-left: max(1rem, 2vw) !important;
        padding-right: max(1rem, 2vw) !important;
        margin-left: 0 !important;
        margin-right: 0 !important;
        box-sizing: border-box !important;
    }
    
    /* Responsive padding - immediate application */
    @media (min-width: 768px) {
        section.main > div.block-container,
        div.block-container,
        .main > div.block-container {
            padding-left: max(2rem, 3vw) !important;
            padding-right: max(2rem, 3vw) !important;
        }
    }
    
    @media (min-width: 1200px) {
        section.main > div.block-container,
        div.block-container,
        .main > div.block-container {
            padding-left: max(3rem, 4vw) !important;
            padding-right: max(3rem, 4vw) !important;
        }
    }
    
    @media (max-width: 767px) {
        section.main > div.block-container,
        div.block-container,
        .main > div.block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
    }
    
    /* Element containers - prevent any width restrictions */
    .element-container,
    div.element-container,
    div[data-testid="element-container"] {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    /* Additional Streamlit containers that might cause issues */
    .css-1kyxreq,
    div.css-1kyxreq,
    .css-12oz5g7,
    div.css-12oz5g7 {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    </style>
    """, unsafe_allow_html=True)


def inject_immediate_javascript_fix():
    """Inject immediate JavaScript fix that executes synchronously
    
    This JavaScript executes immediately without any setTimeout delays.
    """
    st.markdown("""
    <script>
    /* ===== IMMEDIATE JAVASCRIPT LAYOUT FIX ===== */
    
    // Immediate layout fix function - NO DELAYS
    function applyImmediateLayoutFix() {
        try {
            // Target all possible Streamlit container selectors
            const selectors = [
                '.stApp',
                'div.stApp', 
                'div[data-testid="stApp"]',
                'section.main',
                'div[data-testid="stAppViewContainer"]',
                'section.main > div.block-container',
                'div.block-container',
                'section[data-testid="stMainBlockContainer"]',
                'div[data-testid="stMainBlockContainer"]',
                '.main > div.block-container',
                'div[data-testid="stAppViewContainer"] > section.main > div',
                'div[data-testid="stAppViewContainer"] > section > div.block-container'
            ];
            
            // Apply fixes to app containers
            const appContainers = document.querySelectorAll('.stApp, div.stApp, div[data-testid="stApp"]');
            appContainers.forEach(container => {
                container.style.setProperty('width', '100vw', 'important');
                container.style.setProperty('max-width', '100vw', 'important');
                container.style.setProperty('min-width', '100vw', 'important');
            });
            
            // Apply fixes to main containers
            const mainContainers = document.querySelectorAll('section.main, div[data-testid="stAppViewContainer"]');
            mainContainers.forEach(container => {
                container.style.setProperty('width', '100%', 'important');
                container.style.setProperty('max-width', '100%', 'important');
                container.style.setProperty('margin', '0', 'important');
            });
            
            // Apply fixes to block containers
            const blockContainers = document.querySelectorAll([
                'section.main > div.block-container',
                'div.block-container',
                '.main > div.block-container',
                'div[data-testid="stMainBlockContainer"]'
            ].join(', '));
            
            blockContainers.forEach(container => {
                container.style.setProperty('max-width', 'none', 'important');
                container.style.setProperty('width', '100%', 'important');
                container.style.setProperty('min-width', '100%', 'important');
                container.style.setProperty('margin-left', '0', 'important');
                container.style.setProperty('margin-right', '0', 'important');
                container.style.setProperty('padding-left', 'max(1rem, 2vw)', 'important');
                container.style.setProperty('padding-right', 'max(1rem, 2vw)', 'important');
                container.style.setProperty('box-sizing', 'border-box', 'important');
            });
            
            // Force immediate layout recalculation
            document.body.offsetHeight;
            
            // Dispatch resize event immediately
            window.dispatchEvent(new Event('resize'));
            
        } catch (error) {
            console.warn('Layout fix error:', error);
        }
    }
    
    // Execute immediately - no delays
    applyImmediateLayoutFix();
    
    // Also execute when DOM is ready (for safety)
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', applyImmediateLayoutFix);
    }
    
    // Execute on load as well
    window.addEventListener('load', applyImmediateLayoutFix);
    
    // Immediate mutation observer for dynamic content
    const immediateObserver = new MutationObserver(() => {
        applyImmediateLayoutFix();
    });
    
    // Start observing immediately
    if (document.body) {
        immediateObserver.observe(document.body, { 
            childList: true, 
            subtree: true,
            attributes: true,
            attributeFilter: ['class', 'style']
        });
    }
    
    </script>
    """, unsafe_allow_html=True)


def inject_early_layout_css():
    """Legacy function - now calls immediate fix"""
    inject_immediate_viewport_fix()


def inject_javascript_layout_fix():
    """Legacy function - now calls immediate fix"""
    inject_immediate_javascript_fix()


def apply_layout_fixes():
    """Apply immediate layout fixes - ENHANCED VERSION
    
    This is the main function that should be called to fix layout issues.
    It applies fixes immediately without any delays.
    """
    # Apply viewport and critical CSS first
    inject_immediate_viewport_fix()
    
    # Apply immediate JavaScript fixes
    inject_immediate_javascript_fix()


def inject_emergency_layout_fix():
    """Emergency layout fix for critical situations"""
    st.markdown("""
    <style>
    * { box-sizing: border-box !important; }
    html, body { width: 100vw !important; max-width: 100vw !important; overflow-x: hidden !important; }
    .stApp { width: 100vw !important; max-width: 100vw !important; }
    .main { width: 100% !important; }
    .main .block-container { max-width: none !important; width: 100% !important; padding: 1rem 2vw !important; }
    </style>
    """, unsafe_allow_html=True)


def inject_advanced_layout_css():
    """Advanced layout optimizations - now integrated into immediate fix"""
    st.markdown("""
    <style>
    /* Enhanced layout optimizations */
    .main .block-container {
        min-height: 100vh;
        display: flex;
        flex-direction: column;
    }
    
    /* Column improvements */
    .css-1kyxreq {
        gap: 1rem !important;
    }
    
    @media (max-width: 767px) {
        .css-1kyxreq {
            gap: 0.5rem !important;
        }
    }
    </style>
    """, unsafe_allow_html=True)


def get_layout_css_string():
    """Get the layout CSS as a string for external injection"""
    return """
    /* TEMPL Pipeline - Immediate Layout CSS */
    html, body {
        width: 100vw !important;
        max-width: 100vw !important;
        overflow-x: hidden !important;
        margin: 0 !important;
        padding: 0 !important;
    }
    
    .stApp {
        width: 100vw !important;
        max-width: 100vw !important;
        min-width: 100vw !important;
    }
    
    .main {
        width: 100% !important;
        max-width: 100% !important;
    }
    
    .main .block-container {
        max-width: none !important;
        width: 100% !important;
        padding-left: max(1rem, 2vw) !important;
        padding-right: max(1rem, 2vw) !important;
        margin: 0 !important;
        box-sizing: border-box !important;
    }
    """


# Main export - this is what should be imported and used
__all__ = ['apply_layout_fixes', 'inject_immediate_viewport_fix', 'inject_immediate_javascript_fix']
