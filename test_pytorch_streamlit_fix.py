#!/usr/bin/env python3
"""
Test script to verify PyTorch + Streamlit compatibility fix
"""

def test_imports():
    """Test that all imports work without conflicts"""
    print("Testing imports...")
    
    try:
        import torch
        print("✅ PyTorch import successful")
    except Exception as e:
        print(f"❌ PyTorch import failed: {e}")
        return False
    
    try:
        import streamlit as st
        print("✅ Streamlit import successful")
    except Exception as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        from templ_pipeline.ui.core.hardware_manager import get_hardware_manager
        print("✅ Hardware manager import successful")
    except Exception as e:
        print(f"❌ Hardware manager import failed: {e}")
        return False
    
    return True

def test_hardware_detection():
    """Test hardware detection without file watcher conflicts"""
    print("\nTesting hardware detection...")
    
    try:
        from templ_pipeline.ui.core.hardware_manager import get_hardware_manager
        manager = get_hardware_manager()
        hardware_info = manager.detect_hardware()
        print(f"✅ Hardware detection successful: {hardware_info.recommended_config}")
        return True
    except Exception as e:
        print(f"❌ Hardware detection failed: {e}")
        return False

if __name__ == "__main__":
    print("=== PyTorch + Streamlit Compatibility Test ===")
    
    success = True
    success &= test_imports()
    success &= test_hardware_detection()
    
    if success:
        print("\n🎉 All tests passed! PyTorch + Streamlit compatibility verified.")
    else:
        print("\n❌ Some tests failed. Check the fixes above.")
