#!/usr/bin/env python3
"""
Simple test to verify threading fixes work.
"""

import threading

print("Testing basic threading...")
print(f"Active threads: {threading.active_count()}")

# Test that we can create and use a simple ThreadPoolExecutor
try:
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=2) as executor:
        result = executor.submit(lambda: "test").result()
        print(f"ThreadPoolExecutor test result: {result}")
    
    print("Basic threading test passed!")
except Exception as e:
    print(f"Basic threading test failed: {e}")