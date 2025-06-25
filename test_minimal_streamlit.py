#!/usr/bin/env python
"""
Minimal Streamlit app to test basic functionality
"""
import streamlit as st

def main():
    st.title("Minimal Test App")
    st.write("Hello, World!")
    st.write("If you can see this, Streamlit is working correctly.")
    
    if st.button("Test Button"):
        st.success("Button clicked successfully!")
    
    st.sidebar.title("Sidebar Test")
    st.sidebar.write("Sidebar content")

if __name__ == "__main__":
    main() 