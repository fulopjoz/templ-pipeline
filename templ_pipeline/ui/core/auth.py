# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
TEMPL Pipeline Authentication Module
Provides secure password authentication for Streamlit web interface
Security hardened for CERIT deployment
"""

import hashlib
import streamlit as st
import os
from typing import Optional, Tuple
import time


class TemplAuth:
    """Simple password authentication for TEMPL Pipeline"""
    
    def __init__(self):
        # Session timeout (24 hours)
        self.session_timeout = 24 * 60 * 60
        
        # Validate that password hash is configured
        self._validate_password_configuration()
    
    def _validate_password_configuration(self) -> None:
        """Validate that password hash is properly configured"""
        password_hash = os.getenv("TEMPL_PASSWORD_HASH")
        if not password_hash:
            raise ValueError(
                "TEMPL_PASSWORD_HASH environment variable is required for authentication. "
                "Please set it to a SHA-256 hash of your desired password. "
                "Use the generate-auth.sh script in deploy/scripts/ to generate it."
            )
        
        if len(password_hash) != 64:
            raise ValueError(
                "TEMPL_PASSWORD_HASH must be a valid SHA-256 hash (64 characters). "
                "Use the generate-auth.sh script in deploy/scripts/ to generate it."
            )
    
    def get_password_hash(self) -> str:
        """Get password hash from environment"""
        password_hash = os.getenv("TEMPL_PASSWORD_HASH")
        if not password_hash:
            raise ValueError("TEMPL_PASSWORD_HASH environment variable is required")
        return password_hash
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        if "authenticated" not in st.session_state:
            return False
        
        if "auth_timestamp" not in st.session_state:
            return False
        
        # Check session timeout
        if time.time() - st.session_state.auth_timestamp > self.session_timeout:
            self.logout()
            return False
        
        return st.session_state.authenticated
    
    def authenticate(self, password: str) -> bool:
        """Authenticate user with password"""
        password_hash = self.hash_password(password)
        expected_hash = self.get_password_hash()
        
        if password_hash == expected_hash:
            st.session_state.authenticated = True
            st.session_state.auth_timestamp = time.time()
            return True
        
        return False
    
    def logout(self):
        """Logout user and clear session"""
        if "authenticated" in st.session_state:
            del st.session_state.authenticated
        if "auth_timestamp" in st.session_state:
            del st.session_state.auth_timestamp
    
    def show_login_form(self) -> bool:
        """Display login form and handle authentication"""
        st.markdown("""
        <div style="text-align: center; padding: 2rem;">
            <h2>TEMPL Pipeline Access</h2>
            <p>Please enter the password to access the application</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Create centered login form
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            with st.form("login_form"):
                password = st.text_input(
                    "Password:", 
                    type="password",
                    placeholder="Enter password",
                    help="Contact administrator if you don't have the password"
                )
                
                login_button = st.form_submit_button("Login", use_container_width=True)
                
                if login_button:
                    if password:
                        if self.authenticate(password):
                            st.success("Authentication successful! Redirecting...")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("Invalid password. Please try again.")
                            time.sleep(2)  # Brief delay to prevent brute force
                    else:
                        st.warning("Please enter a password.")
        
        return False
    
    def require_auth(self) -> bool:
        """
        Decorator-like function to require authentication
        Returns True if authenticated, False if login form shown
        """
        if not self.is_authenticated():
            self.show_login_form()
            return False
        
        # Show logout option in sidebar
        with st.sidebar:
            st.markdown("---")
            st.markdown("**Security Status:** Authenticated")
            
            remaining_time = self.session_timeout - (time.time() - st.session_state.auth_timestamp)
            hours_remaining = int(remaining_time // 3600)
            
            st.markdown(f"**Session:** ~{hours_remaining}h remaining")
            
            if st.button("🔐 Logout", use_container_width=True):
                self.logout()
                st.rerun()
        
        return True


# Global authentication instance
auth = TemplAuth()


def require_authentication():
    """
    Decorator function for Streamlit pages that require authentication
    Usage: Call this at the start of your Streamlit app
    """
    return auth.require_auth()


def get_password_setup_instructions() -> str:
    """
    Get instructions for setting up custom password
    """
    return """
    ## Setting up Custom Password
    
    ### For Docker/Kubernetes Deployment:
    
    1. **Generate password hash:**
    ```python
    import hashlib
    password = "your_secure_password"
    hash_value = hashlib.sha256(password.encode()).hexdigest()
    print(f"TEMPL_PASSWORD_HASH={hash_value}")
    ```
    
    2. **Set environment variable:**
    ```bash
    # In Kubernetes deployment.yaml
    env:
    - name: TEMPL_PASSWORD_HASH
      value: "your_generated_hash"
    
    # Or in Docker run
    docker run -e TEMPL_PASSWORD_HASH="your_hash" ...
    ```
    
    ### Security Best Practices:
    - Use strong passwords (12+ characters)
    - Store hash in Kubernetes secrets (not plain text)
    - Regularly rotate passwords
    - Monitor access logs
    """


if __name__ == "__main__":
    # Test authentication setup
    print("TEMPL Pipeline Authentication Test")
    print("=" * 40)
    
    # Check if password hash is configured
    password_hash = os.getenv("TEMPL_PASSWORD_HASH")
    if password_hash:
        print("✅ TEMPL_PASSWORD_HASH is configured")
        try:
            auth_test = TemplAuth()
            print("✅ Authentication module initialized successfully")
        except ValueError as e:
            print(f"❌ Configuration error: {e}")
    else:
        print("❌ TEMPL_PASSWORD_HASH environment variable not set")
        print("\nTo set up authentication:")
        print("1. Run: ./deploy/scripts/generate-auth.sh --app-password 'your_password'")
        print("2. Set the TEMPL_PASSWORD_HASH environment variable")
        print("3. Test again")
    
    print("\n" + get_password_setup_instructions())
