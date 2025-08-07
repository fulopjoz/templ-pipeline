"""
pytest configuration and shared fixtures for TEMPL pipeline tests.

This module provides session-scoped fixtures for expensive operations
and optimizations to improve test performance.
"""

import tempfile
import pytest
import logging
import asyncio
import sys
from unittest.mock import MagicMock, patch
from pathlib import Path
from rdkit import Chem

# Import smart caching system
try:
    from .test_fixture_caching import fixture_manager, cached_fixture
    CACHING_AVAILABLE = True
except ImportError:
    CACHING_AVAILABLE = False

# Import test fixtures from centralized factory
try:
    from .fixtures.data_factory import TestDataFactory
    from .fixtures.pipeline_fixtures import (
        standard_test_molecules, 
        standard_test_proteins,
        mock_embeddings,
        mcs_test_pairs
    )
    from .fixtures.benchmark_fixtures import benchmark_target_data
    FIXTURES_AVAILABLE = True
except ImportError:
    # Fallback for development
    FIXTURES_AVAILABLE = False

# Only import streamlit for UI tests - make it conditional
try:
    import streamlit as st

    STREAMLIT_AVAILABLE = True
except ImportError:
    st = None
    STREAMLIT_AVAILABLE = False


@pytest.fixture(scope="session", autouse=True)
def optimize_test_environment():
    """Global test environment optimization applied to all tests."""
    # Reduce logging noise during tests
    logging.getLogger("templ_pipeline").setLevel(logging.WARNING)
    logging.getLogger("rdkit").setLevel(logging.ERROR)

    # Configure async loop policy for better performance
    if sys.platform.startswith("win"):
        # Use ProactorEventLoopPolicy on Windows for better async performance
        asyncio.set_event_loop_policy(asyncio.ProactorEventLoopPolicy())
    else:
        # Use default policy on Unix systems
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

    yield

    # Restore logging levels after tests
    logging.getLogger("templ_pipeline").setLevel(logging.INFO)
    logging.getLogger("rdkit").setLevel(logging.INFO)


@pytest.fixture(scope="session")
def rdkit_modules():
    """
    Load RDKit modules once per session to avoid repeated imports.

    Returns:
        tuple: (Chem, AllChem, Draw) RDKit modules
    """
    try:
        from rdkit import Chem, AllChem, Draw
        from rdkit import RDLogger

        # Disable RDKit warnings during tests
        RDLogger.DisableLog("rdApp.*")
        return Chem, AllChem, Draw
    except ImportError:
        pytest.skip("RDKit not available")


@pytest.fixture(scope="session")
def mock_streamlit():
    """
    Session-wide Streamlit mocking fixture.

    Returns:
        MagicMock: Configured mock streamlit module
    """
    mock_st = MagicMock()

    # Configure cache decorators
    def mock_cache_data(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        else:

            def decorator(func):
                return func

            return decorator

    def mock_cache_resource(*args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        else:

            def decorator(func):
                return func

            return decorator

    # Configure mock attributes
    mock_st.cache_data = mock_cache_data
    mock_st.cache_resource = mock_cache_resource
    mock_st.session_state = {}
    mock_st.set_page_config = MagicMock()
    mock_st.markdown = MagicMock()
    mock_st.title = MagicMock()
    mock_st.sidebar = MagicMock()
    mock_st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    mock_st.button = MagicMock(return_value=False)
    mock_st.text_input = MagicMock(return_value="")
    mock_st.selectbox = MagicMock(return_value="")
    mock_st.file_uploader = MagicMock(return_value=None)

    return mock_st


@pytest.fixture(scope="session")
def ui_test_environment(mock_streamlit):
    """
    Set up UI testing environment with proper mocks.

    Args:
        mock_streamlit: Mock streamlit fixture

    Yields:
        dict: UI test environment configuration
    """
    # Set up module mocks
    with patch.dict(
        "sys.modules",
        {"streamlit": mock_streamlit, "py3Dmol": MagicMock(), "stmol": MagicMock()},
    ):
        yield {
            "streamlit": mock_streamlit,
            "py3Dmol": sys.modules["py3Dmol"],
            "stmol": sys.modules["stmol"],
        }


@pytest.fixture(scope="session")
def mock_embedding_manager():
    """
    Session-wide mock embedding manager for tests.

    Returns:
        MagicMock: Configured mock embedding manager
    """
    mock_manager = MagicMock()

    # Configure mock methods
    mock_manager.get_embedding.return_value = (
        MagicMock(),  # Mock embedding
        "A,B",  # Mock chain data
    )
    mock_manager.find_neighbors.return_value = ["1abc", "2xyz", "3pqr"]
    mock_manager.embedding_db = {"1abc": MagicMock(), "2xyz": MagicMock()}
    mock_manager.embedding_chain_data = {"1abc": "A", "2xyz": "B"}

    return mock_manager


@pytest.fixture(scope="session")
def mock_transformers():
    """
    Mock transformers package to avoid import issues in tests.

    Returns:
        MagicMock: Mock transformers module
    """
    mock_transformers = MagicMock()
    mock_transformers.EsmModel = MagicMock()
    mock_transformers.EsmTokenizer = MagicMock()
    return mock_transformers


@pytest.fixture(scope="function")
def async_test_helper():
    """
    Helper fixture for async test operations.

    Returns:
        dict: Async test utilities
    """

    def create_resolved_future(result=None):
        """Create a resolved asyncio Future with given result."""
        future = asyncio.Future()
        future.set_result(result or {"poses": {}})
        return future

    def create_failed_future(exception):
        """Create a failed asyncio Future with given exception."""
        future = asyncio.Future()
        future.set_exception(exception)
        return future

    return {
        "create_resolved_future": create_resolved_future,
        "create_failed_future": create_failed_future,
        "default_result": {"poses": {}},
    }


@pytest.fixture(scope="function")
def temp_test_files(tmp_path):
    """
    Create temporary test files for file-based tests.

    Args:
        tmp_path: pytest temporary path fixture

    Returns:
        dict: Paths to temporary test files
    """
    # Create test SDF file
    sdf_file = tmp_path / "test.sdf"
    sdf_file.write_text(
        """
  Mrv2014 01010100002D          

  3  2  0  0  0  0            999 V2000
   -0.4125    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.4125    0.7145    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.0000    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0
  1  3  1  0  0  0  0
  2  3  1  0  0  0  0
M  END
$$$$
"""
    )

    # Create test PDB file
    pdb_file = tmp_path / "test.pdb"
    pdb_file.write_text(
        """
HEADER    TEST PROTEIN                            01-JAN-00   TEST            
ATOM      1  N   ALA A   1      20.154  16.967  12.784  1.00 10.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  12.345  1.00 10.00           C  
ATOM      3  C   ALA A   1      18.899  14.793  13.116  1.00 10.00           C  
ATOM      4  O   ALA A   1      19.505  14.548  14.163  1.00 10.00           O  
END
"""
    )

    return {"sdf_file": sdf_file, "pdb_file": pdb_file, "temp_dir": tmp_path}


# Performance optimization markers
def pytest_configure(config):
    """Configure pytest with performance optimizations."""
    # Add custom markers
    config.addinivalue_line("markers", "performance: mark test as performance-related")
    config.addinivalue_line("markers", "integration: mark test as integration test")


def pytest_collection_modifyitems(config, items):
    """Modify test collection for performance optimization."""
    # Mark slow tests automatically
    for item in items:
        # Mark UI tests as medium speed
        if "ui" in item.keywords:
            item.add_marker(pytest.mark.medium)

        # Mark integration tests as slow
        if "integration" in item.keywords:
            item.add_marker(pytest.mark.slow)

        # Mark embedding tests as medium speed
        if "embedding" in item.nodeid.lower():
            item.add_marker(pytest.mark.medium)


# Session-wide test data for performance
@pytest.fixture(scope="session")
def test_molecules():
    """
    Session-wide test molecule data.

    Returns:
        dict: Test molecule SMILES and properties
    """
    if FIXTURES_AVAILABLE:
        # Use standardized test data factory
        molecules = {}
        for mol_type in ['simple_alkane', 'aromatic', 'complex_drug', 'invalid']:
            molecules[mol_type] = TestDataFactory.create_molecule_data(mol_type)
        return molecules
    else:
        # Fallback to original data
        return {
            "simple": {"smiles": "CCO", "name": "ethanol", "atoms": 3},
            "aromatic": {"smiles": "c1ccccc1", "name": "benzene", "atoms": 6},
            "complex": {
                "smiles": "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",
                "name": "ibuprofen",
                "atoms": 13,
            },
            "invalid": {"smiles": "INVALID", "name": "invalid", "atoms": 0},
        }


@pytest.fixture(scope="session") 
def session_test_embeddings():
    """
    Session-wide test embeddings for expensive operations.
    
    Returns:
        numpy.ndarray: Standardized test embeddings
    """
    if FIXTURES_AVAILABLE:
        return TestDataFactory.create_embedding_data(size=1280, num_proteins=10)
    else:
        import numpy as np
        np.random.seed(42)
        embeddings = np.random.rand(10, 1280).astype(np.float32)
        np.random.seed()
        return embeddings


@pytest.fixture(scope="session")
def session_test_proteins():
    """
    Session-wide test protein data.
    
    Returns:
        dict: Standardized test protein data
    """
    if FIXTURES_AVAILABLE:
        proteins = {}
        for protein_type in ['minimal', 'multi_chain']:
            proteins[protein_type] = TestDataFactory.create_protein_data(protein_type)
        return proteins
    else:
        return {
            'minimal': {
                'content': """HEADER    TEST PROTEIN                            01-JAN-00   TEST            
ATOM      1  N   ALA A   1      20.154  16.967  12.784  1.00 10.00           N  
ATOM      2  CA  ALA A   1      19.030  16.101  12.345  1.00 10.00           C  
ATOM      3  C   ALA A   1      18.899  14.793  13.116  1.00 10.00           C  
ATOM      4  O   ALA A   1      19.505  14.548  14.163  1.00 10.00           O  
END""",
                'chains': ['A'],
                'atoms': 4,
                'type': 'minimal'
            }
        }


@pytest.fixture(scope="session")
def session_mcs_test_pairs():
    """
    Session-wide MCS test pairs for parametrized testing.
    
    Returns:
        list: List of (mol1_smiles, mol2_smiles, description) tuples
    """
    if FIXTURES_AVAILABLE:
        return TestDataFactory.create_mcs_test_pairs()
    else:
        return [
            ('CCO', 'CCC', 'Similar alkanes'),
            ('CCO', 'CCOC', 'Alcohol vs ether'),
            ('c1ccccc1', 'c1ccccc1O', 'Benzene vs phenol')
        ]


@pytest.fixture(scope="session")
def temp_dir():
    """Create temporary directory for test files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_smiles():
    """Sample SMILES strings for testing"""
    return [
        "CCO",  # ethanol
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
    ]


@pytest.fixture
def sample_mol():
    """Sample RDKit molecule"""
    return Chem.MolFromSmiles("CCO")


@pytest.fixture
def mock_streamlit():
    """Mock Streamlit session state for UI tests"""
    if not STREAMLIT_AVAILABLE:
        # Create a mock session state when streamlit is not available
        mock_state = {}
        yield mock_state
        mock_state.clear()
        return

    if "session_state" not in st.__dict__:
        st.session_state = {}
    yield st.session_state
    st.session_state.clear()


@pytest.fixture
def output_dir(temp_dir):
    """Output directory for test results"""
    output = temp_dir / "output"
    output.mkdir(exist_ok=True)
    return output


@pytest.fixture(scope="function")
def centralized_output_dir(tmp_path):
    """
    Centralized output directory fixture using standardized structure.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        Path to centralized test output directory
    """
    from templ_pipeline.core.workspace_manager import DirectoryManager
    
    # Create centralized output structure for tests
    output_root = tmp_path / "output" 
    output_root.mkdir(exist_ok=True)
    
    manager = DirectoryManager(
        base_name="test_run",
        run_id="test",
        auto_cleanup=True,
        centralized_output=True,
        output_root=str(output_root)
    )
    
    return manager.directory


@pytest.fixture(scope="function")
def test_directory_manager(tmp_path):
    """
    Standardized directory manager for tests.
    
    Args:
        tmp_path: pytest temporary path fixture
        
    Returns:
        DirectoryManager configured for testing
    """
    from templ_pipeline.core.workspace_manager import DirectoryManager
    
    return DirectoryManager(
        base_name="test",
        auto_cleanup=True,
        centralized_output=True,
        output_root=str(tmp_path / "output")
    )


@pytest.fixture
def test_data_dir():
    """Test data directory"""
    return Path(__file__).parent.parent / "data"


def pytest_configure(config):
    """Configure pytest with custom markers and skip logic"""
    config.addinivalue_line("markers", "ui: mark test as requiring UI dependencies")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "fast: mark test as fast running")
    config.addinivalue_line("markers", "cached: mark test as using cached fixtures")
    
    # Initialize fixture cache if available
    if CACHING_AVAILABLE:
        fixture_manager.cache.clear()  # Start with clean cache


def pytest_collection_modifyitems(config, items):
    """Skip UI tests if streamlit is not available"""
    if STREAMLIT_AVAILABLE:
        return

    skip_ui = pytest.mark.skip(reason="streamlit not available")
    for item in items:
        if "ui" in item.keywords:
            item.add_marker(skip_ui)


# Cached fixtures for expensive operations
if CACHING_AVAILABLE:
    
    @cached_fixture(scope="session")
    def cached_rdkit_modules():
        """Cached RDKit modules with enhanced loading."""
        import time
        start_time = time.time()
        
        try:
            from rdkit import Chem, AllChem, Draw
            from rdkit import RDLogger
            RDLogger.DisableLog("rdApp.*")
            
            # Pre-load common functionality
            # Create some test molecules to warm up the RDKit cache
            test_smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
            for smiles in test_smiles:
                mol = Chem.MolFromSmiles(smiles)
                if mol:
                    AllChem.Compute2DCoords(mol)
            
            load_time = time.time() - start_time
            print(f"RDKit modules loaded and cached in {load_time:.3f}s")
            
            return Chem, AllChem, Draw
        except ImportError:
            pytest.skip("RDKit not available")
    
    @cached_fixture(scope="session")
    def cached_protein_data():
        """Cached protein test data."""
        import time
        start_time = time.time()
        
        # Simulate expensive protein loading
        protein_data = {
            '1abc': {
                'sequence': 'MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEEFGLAPFLPDQIHFVHSQELLSRYPDLDAKGRERAIAKDLGAVFLVGIGGKLSDGHRHDVRAPDYDDWUQTPQWNSPSYQPQLYITAQQTQRAADLGDGWKWSDLFLGPGMSEQHLAQQGKQGKGQ',
                'pdb_id': '1abc',
                'chains': ['A', 'B'],
                'embedding': [0.1] * 1280
            },
            '2xyz': {
                'sequence': 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG',
                'pdb_id': '2xyz',
                'chains': ['A'],
                'embedding': [0.2] * 1280
            }
        }
        
        load_time = time.time() - start_time
        print(f"Protein data loaded and cached in {load_time:.3f}s")
        
        return protein_data
    
    @cached_fixture(scope="session")
    def cached_ligand_data():
        """Cached ligand test data."""
        import time
        start_time = time.time()
        
        ligand_data = {
            'simple_molecules': [
                {'smiles': 'CCO', 'name': 'ethanol'},
                {'smiles': 'CC(=O)O', 'name': 'acetic_acid'},
                {'smiles': 'c1ccccc1', 'name': 'benzene'}
            ],
            'complex_molecules': [
                {'smiles': 'CN1C=NC2=C1C(=O)N(C(=O)N2C)C', 'name': 'caffeine'},
                {'smiles': 'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O', 'name': 'ibuprofen'}
            ]
        }
        
        load_time = time.time() - start_time
        print(f"Ligand data loaded and cached in {load_time:.3f}s")
        
        return ligand_data
    
    @cached_fixture(scope="session")
    def cached_benchmark_data():
        """Cached benchmark test data."""
        import time
        start_time = time.time()
        
        # Simulate expensive benchmark data generation
        benchmark_data = {
            'targets': [f'target_{i}' for i in range(100)],
            'ligands': [f'ligand_{i}' for i in range(500)],
            'embeddings': {f'target_{i}': [i/100] * 1280 for i in range(100)},
            'ground_truth': {f'target_{i}': [f'ligand_{j}' for j in range(i, i+5)] for i in range(0, 100, 5)}
        }
        
        load_time = time.time() - start_time
        print(f"Benchmark data loaded and cached in {load_time:.3f}s")
        
        return benchmark_data


def pytest_sessionfinish(session, exitstatus):
    """Report cache statistics and cleanup at session end."""
    if CACHING_AVAILABLE:
        stats = fixture_manager.get_cache_stats()
        
        if stats['memory_hits'] + stats['file_hits'] + stats['misses'] > 0:
            print("\n" + "="*60)
            print("FIXTURE CACHE PERFORMANCE")
            print("="*60)
            print(f"Hit Rate: {stats['hit_rate']:.1%}")
            print(f"Memory Hits: {stats['memory_hits']}")
            print(f"File Hits: {stats['file_hits']}")
            print(f"Cache Misses: {stats['misses']}")
            print(f"Memory Usage: {stats['memory_usage_mb']:.1f} MB")
            print("="*60)
        
        # Cleanup
        fixture_manager.cleanup_session_fixtures()
