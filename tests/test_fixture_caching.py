"""
Advanced fixture caching system for test performance optimization.
Provides intelligent caching for expensive test operations.
"""

import hashlib
import json
import pickle
import time
import threading
from pathlib import Path
from typing import Any, Dict, Optional, Union, Callable
import pytest
import functools
from collections import defaultdict
import psutil
import weakref


class FixtureCache:
    """Smart caching system for expensive test fixtures."""
    
    def __init__(self, cache_dir: str = "test-cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.memory_cache = {}
        self.cache_stats = defaultdict(int)
        self.lock = threading.Lock()
        self.max_memory_cache_size = 100  # Max items in memory
        self.max_memory_usage = 500 * 1024 * 1024  # 500MB
        
    def _get_cache_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Generate cache key from function signature."""
        # Create a stable key from function name and arguments
        key_data = {
            'func': func_name,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()
    
    def _get_file_cache_path(self, cache_key: str) -> Path:
        """Get file path for cached item."""
        return self.cache_dir / f"{cache_key}.cache"
    
    def _get_memory_usage(self) -> int:
        """Get current memory usage of the cache."""
        total_size = 0
        for value in self.memory_cache.values():
            try:
                total_size += len(pickle.dumps(value))
            except:
                total_size += 1024  # Estimate for non-picklable objects
        return total_size
    
    def _evict_memory_cache(self):
        """Evict items from memory cache if needed."""
        if len(self.memory_cache) > self.max_memory_cache_size:
            # Remove oldest 20% of items
            items_to_remove = len(self.memory_cache) - int(self.max_memory_cache_size * 0.8)
            keys_to_remove = list(self.memory_cache.keys())[:items_to_remove]
            for key in keys_to_remove:
                del self.memory_cache[key]
        
        # Check memory usage
        if self._get_memory_usage() > self.max_memory_usage:
            # Remove half the cache
            keys_to_remove = list(self.memory_cache.keys())[:len(self.memory_cache) // 2]
            for key in keys_to_remove:
                del self.memory_cache[key]
    
    def get(self, func_name: str, args: tuple, kwargs: dict) -> tuple:
        """Get cached result if available."""
        cache_key = self._get_cache_key(func_name, args, kwargs)
        
        with self.lock:
            # Check memory cache first
            if cache_key in self.memory_cache:
                self.cache_stats['memory_hits'] += 1
                return True, self.memory_cache[cache_key]
            
            # Check file cache
            cache_file = self._get_file_cache_path(cache_key)
            if cache_file.exists():
                try:
                    with open(cache_file, 'rb') as f:
                        result = pickle.load(f)
                    
                    # Store in memory cache for faster access
                    self.memory_cache[cache_key] = result
                    self._evict_memory_cache()
                    
                    self.cache_stats['file_hits'] += 1
                    return True, result
                except Exception:
                    # Cache file corrupted, remove it
                    cache_file.unlink()
            
            self.cache_stats['misses'] += 1
            return False, None
    
    def set(self, func_name: str, args: tuple, kwargs: dict, result: Any):
        """Store result in cache."""
        cache_key = self._get_cache_key(func_name, args, kwargs)
        
        with self.lock:
            # Store in memory cache
            self.memory_cache[cache_key] = result
            self._evict_memory_cache()
            
            # Store in file cache for persistence
            cache_file = self._get_file_cache_path(cache_key)
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(result, f)
                self.cache_stats['stores'] += 1
            except Exception:
                # If we can't pickle it, just keep in memory
                pass
    
    def clear(self):
        """Clear all cached data."""
        with self.lock:
            self.memory_cache.clear()
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            self.cache_stats.clear()
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self.lock:
            total_requests = sum(self.cache_stats.values())
            hit_rate = (self.cache_stats['memory_hits'] + self.cache_stats['file_hits']) / total_requests if total_requests > 0 else 0
            
            return {
                'memory_hits': self.cache_stats['memory_hits'],
                'file_hits': self.cache_stats['file_hits'],
                'misses': self.cache_stats['misses'],
                'stores': self.cache_stats['stores'],
                'hit_rate': hit_rate,
                'memory_cache_size': len(self.memory_cache),
                'memory_usage_mb': self._get_memory_usage() / 1024 / 1024,
                'file_cache_size': len(list(self.cache_dir.glob("*.cache")))
            }


class CachedFixture:
    """Decorator for creating cached fixtures."""
    
    def __init__(self, cache: FixtureCache, ttl: Optional[int] = None):
        self.cache = cache
        self.ttl = ttl  # Time to live in seconds
        
    def __call__(self, func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Check cache
            found, result = self.cache.get(func.__name__, args, kwargs)
            if found:
                return result
            
            # Execute function and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Only cache if execution was expensive (>0.1s)
            if execution_time > 0.1:
                self.cache.set(func.__name__, args, kwargs, result)
            
            return result
        
        return wrapper


class FixtureManager:
    """Manages cached fixtures with lifecycle hooks."""
    
    def __init__(self):
        self.cache = FixtureCache()
        self.fixture_registry = {}
        self.session_fixtures = weakref.WeakValueDictionary()
        
    def register_fixture(self, name: str, fixture_func: Callable, scope: str = "function"):
        """Register a fixture with the manager."""
        self.fixture_registry[name] = {
            'func': fixture_func,
            'scope': scope,
            'cached': False
        }
    
    def cached_fixture(self, scope: str = "function", ttl: Optional[int] = None):
        """Decorator for creating cached fixtures."""
        def decorator(func):
            cached_func = CachedFixture(self.cache, ttl)(func)
            self.register_fixture(func.__name__, cached_func, scope)
            
            if scope == "session":
                # Session-scoped fixtures are stored in weak references
                def session_wrapper(*args, **kwargs):
                    key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
                    if key not in self.session_fixtures:
                        self.session_fixtures[key] = cached_func(*args, **kwargs)
                    return self.session_fixtures[key]
                
                return session_wrapper
            else:
                return cached_func
        
        return decorator
    
    def cleanup_session_fixtures(self):
        """Clean up session-scoped fixtures."""
        self.session_fixtures.clear()
    
    def get_cache_stats(self) -> Dict:
        """Get comprehensive cache statistics."""
        stats = self.cache.get_stats()
        stats['registered_fixtures'] = len(self.fixture_registry)
        stats['active_session_fixtures'] = len(self.session_fixtures)
        return stats


# Global fixture manager
fixture_manager = FixtureManager()


# Enhanced fixture decorators
def cached_fixture(scope: str = "function", ttl: Optional[int] = None):
    """Decorator for creating performance-optimized cached fixtures."""
    return fixture_manager.cached_fixture(scope, ttl)


@cached_fixture(scope="session")
def expensive_test_data():
    """Example of a cached expensive fixture."""
    print("Generating expensive test data...")
    time.sleep(0.5)  # Simulate expensive operation
    
    # Generate large dataset
    data = {
        'molecules': [f"molecule_{i}" for i in range(1000)],
        'embeddings': [[i] * 100 for i in range(1000)],
        'metadata': {'created': time.time(), 'size': 1000}
    }
    
    return data


@cached_fixture(scope="session")
def protein_embeddings():
    """Cached protein embeddings fixture."""
    print("Loading protein embeddings...")
    time.sleep(0.3)  # Simulate loading time
    
    # Mock embedding data
    embeddings = {
        'protein_1': [0.1] * 1280,
        'protein_2': [0.2] * 1280,
        'protein_3': [0.3] * 1280
    }
    
    return embeddings


@cached_fixture(scope="session")
def mock_molecules():
    """Cached RDKit molecules fixture."""
    print("Creating mock molecules...")
    time.sleep(0.2)  # Simulate molecule creation
    
    # Import RDKit here to avoid issues if not available
    try:
        from rdkit import Chem
        
        smiles_list = [
            "CCO",  # Ethanol
            "CC(=O)O",  # Acetic acid
            "c1ccccc1",  # Benzene
            "CCN(CC)CC",  # Triethylamine
            "c1ccc2nc3ccccc3cc2c1"  # Phenanthrene
        ]
        
        molecules = {}
        for i, smiles in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                molecules[f"mol_{i}"] = mol
        
        return molecules
    except ImportError:
        # Return mock data if RDKit not available
        return {f"mol_{i}": f"mock_mol_{i}" for i in range(5)}


# Pytest hooks for cache management
def pytest_configure(config):
    """Configure cache system for pytest."""
    # Clear cache at start of test session
    fixture_manager.cache.clear()


def pytest_sessionstart(session):
    """Initialize cache statistics tracking."""
    fixture_manager.cache.cache_stats.clear()


def pytest_sessionfinish(session, exitstatus):
    """Report cache statistics at end of session."""
    stats = fixture_manager.get_cache_stats()
    
    if stats['memory_hits'] + stats['file_hits'] + stats['misses'] > 0:
        print("\n" + "="*80)
        print("FIXTURE CACHE STATISTICS")
        print("="*80)
        print(f"Cache Hit Rate: {stats['hit_rate']:.1%}")
        print(f"Memory Hits: {stats['memory_hits']}")
        print(f"File Hits: {stats['file_hits']}")
        print(f"Cache Misses: {stats['misses']}")
        print(f"Items Stored: {stats['stores']}")
        print(f"Memory Cache Size: {stats['memory_cache_size']} items")
        print(f"Memory Usage: {stats['memory_usage_mb']:.1f} MB")
        print(f"File Cache Size: {stats['file_cache_size']} files")
        print("="*80)
    
    # Cleanup
    fixture_manager.cleanup_session_fixtures()


@pytest.fixture
def cache_stats():
    """Fixture to provide cache statistics."""
    return fixture_manager.get_cache_stats()


def test_fixture_caching_functionality():
    """Test the fixture caching system."""
    cache = FixtureCache()
    
    # Test cache miss
    found, result = cache.get("test_func", (), {})
    assert not found
    assert result is None
    
    # Test cache store and hit
    test_data = {"test": "data"}
    cache.set("test_func", (), {}, test_data)
    
    found, result = cache.get("test_func", (), {})
    assert found
    assert result == test_data
    
    # Test cache statistics
    stats = cache.get_stats()
    assert stats['misses'] == 1
    assert stats['memory_hits'] == 1
    assert stats['stores'] == 1


def test_cached_fixture_decorator():
    """Test the cached fixture decorator."""
    call_count = 0
    
    @cached_fixture(scope="function")
    def test_fixture():
        nonlocal call_count
        call_count += 1
        return f"result_{call_count}"
    
    # First call should execute
    result1 = test_fixture()
    assert result1 == "result_1"
    assert call_count == 1
    
    # Second call should use cache
    result2 = test_fixture()
    assert result2 == "result_1"  # Same result
    assert call_count == 1  # Not called again


if __name__ == "__main__":
    # Test the caching system
    print("Testing fixture caching system...")
    
    # Test expensive fixture
    start_time = time.time()
    data1 = expensive_test_data()
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    data2 = expensive_test_data()
    second_call_time = time.time() - start_time
    
    print(f"First call: {first_call_time:.3f}s")
    print(f"Second call: {second_call_time:.3f}s")
    print(f"Speedup: {first_call_time / second_call_time:.1f}x")
    
    # Print cache stats
    stats = fixture_manager.get_cache_stats()
    print(f"Cache statistics: {stats}")