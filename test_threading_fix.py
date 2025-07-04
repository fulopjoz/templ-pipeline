#!/usr/bin/env python3
"""
Test script to verify the threading fixes in scoring.py
"""
import sys
import os
sys.path.insert(0, '/home/ubuntu/mcs/templ_pipeline')

import logging
import threading
from templ_pipeline.core.scoring import _get_executor_for_context
from templ_pipeline.core.thread_manager import ThreadResourceManager, log_thread_status

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def test_threading_fix():
    """Test that the threading fix works correctly."""
    print("Testing threading fix in scoring.py...")
    
    # Test 1: Basic thread manager functionality
    print("\n1. Testing ThreadResourceManager...")
    manager = ThreadResourceManager()
    health = manager.check_thread_health()
    print(f"   Thread health: {health}")
    
    # Test 2: Safe worker count calculation
    print("\n2. Testing safe worker count...")
    safe_workers = manager.get_safe_worker_count(16, task_type="scoring")
    print(f"   Requested: 16, Safe: {safe_workers}")
    
    # Test 3: Thread status logging
    print("\n3. Testing thread status logging...")
    log_thread_status("test_context")
    
    # Test 4: Executor creation with thread management
    print("\n4. Testing executor creation...")
    try:
        executor = _get_executor_for_context(8)
        print(f"   Executor created successfully: {type(executor)}")
        executor.shutdown(wait=True)
        print("   Executor cleaned up successfully")
    except Exception as e:
        print(f"   Error creating executor: {e}")
    
    # Test 5: Multiple executor creations to test resource management
    print("\n5. Testing multiple executor creations...")
    log_thread_status("before_multiple_executors")
    
    executors = []
    try:
        for i in range(3):
            executor = _get_executor_for_context(4)
            executors.append(executor)
            print(f"   Executor {i+1} created")
        
        log_thread_status("after_multiple_executors")
        
        # Clean up
        for i, executor in enumerate(executors):
            executor.shutdown(wait=True)
            print(f"   Executor {i+1} cleaned up")
        
        log_thread_status("after_cleanup")
        
    except Exception as e:
        print(f"   Error in multiple executor test: {e}")
        # Clean up any remaining executors
        for executor in executors:
            try:
                executor.shutdown(wait=True)
            except:
                pass
    
    print("\nThreading fix test completed!")

if __name__ == "__main__":
    test_threading_fix()