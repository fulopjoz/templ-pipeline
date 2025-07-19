# TEMPL Timesplit Benchmark Optimization Plan

## ROOT CAUSE ANALYSIS

### Why Polaris Works (Small Scale)
- **Dataset**: ~1000 molecules max (SARS: 903, MERS: 19, test: 819)
- **Architecture**: Direct SDF loading, small molecule lists passed to workers
- **Memory**: Results accumulation works due to small scale

### Why Timesplit Has Progressive OOM (Large Scale)
- **Dataset**: Thousands of PDB targets from full molecular databases
- **Critical Issue**: Each worker loads entire 6GB molecular database independently
- **Memory Pattern**: 8 workers × 6GB cache = 48GB+ just for caches
- **Result**: OOM at ~50 targets due to exponential memory growth

### Key Code Problem
```python
# runner.py line 224 - THE CULPRIT
self._molecule_cache = load_sdf_molecules_cached(
    path, cache=None, memory_limit_gb=6.0  # PER WORKER!
)
```

## OPTIMIZATION STRATEGY

### Phase 1: Immediate Memory Fixes

#### 1.1 Shared Cache Pre-loading
```python
# New architecture: Load once, share everywhere
class SharedMolecularCache:
    _instance = None
    _cache_data = None
    
    @classmethod 
    def get_cache(cls, data_dir):
        if cls._cache_data is None:
            cls._cache_data = load_molecular_database_once(data_dir)
        return cls._cache_data

# Modified runner.py
def run_templ_pipeline_for_benchmark(...):
    # Use shared cache instead of loading per worker
    shared_cache = SharedMolecularCache.get_cache(data_dir)
    # Pass cache reference, don't reload
```

#### 1.2 Memory-Mapped Molecular Database
```python
# Use memory-mapped files for molecular data
class MemoryMappedMolCache:
    def __init__(self, data_dir):
        self.mmap_file = self._create_mmap_cache(data_dir)
    
    def get_molecule(self, pdb_id):
        # Load only specific molecule, not entire database
        return self.mmap_file.get(pdb_id)
```

#### 1.3 Worker Pool Recycling
```python
# Recycle workers after N targets to prevent accumulation
def _process_targets_with_recycling(targets, config, n_workers):
    RECYCLE_AFTER = 25  # Recycle workers every 25 targets
    
    for chunk in chunks(targets, RECYCLE_AFTER):
        with ProcessPoolExecutor(n_workers) as executor:
            # Process chunk
            futures = [executor.submit(...) for target in chunk]
            # Automatic cleanup when context exits
```

### Phase 2: Streaming Architecture

#### 2.1 Lazy Molecular Loading
```python
class LazyMoleculeLoader:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self._index = self._build_molecule_index()  # Small index only
    
    def get_ligand_data(self, pdb_id):
        # Load single molecule on demand
        return self._load_single_molecule(pdb_id)
```

#### 2.2 Target-Specific Data Loading  
```python
def process_single_target_optimized(target_pdb, config):
    # Load only data needed for this specific target
    ligand_data = LazyMoleculeLoader.get_ligand_data(target_pdb)
    protein_file = get_protein_file_cached(target_pdb)
    
    # Process with minimal memory footprint
    result = run_minimal_pipeline(ligand_data, protein_file, config)
    
    # Immediate cleanup
    del ligand_data, protein_file
    return result
```

#### 2.3 Streaming Results (Already Implemented)
```python
# timesplit.py already does this correctly:
with output_jsonl.open("a", encoding="utf-8") as fh:
    json.dump(result, fh)  # Stream to disk immediately
    fh.write('\n')
# No result accumulation in memory ✓
```

### Phase 3: Memory Management Enhancements

#### 3.1 Global Memory Monitoring
```python
class GlobalMemoryMonitor:
    def __init__(self, max_memory_gb=32):
        self.max_memory_gb = max_memory_gb
        
    def check_memory_before_task(self):
        current_usage = psutil.virtual_memory().percent
        if current_usage > 85:
            self.trigger_aggressive_cleanup()
            
    def trigger_aggressive_cleanup(self):
        # Force cleanup across all processes
        gc.collect()
        clear_all_caches()
```

#### 3.2 Memory Barriers Between Chunks
```python
def memory_barrier(context):
    """Enhanced memory barrier with cross-process cleanup"""
    # Clear local caches
    clear_worker_caches()
    
    # Force garbage collection
    for _ in range(3):
        gc.collect()
        time.sleep(0.1)
    
    # Check memory status
    memory_after = psutil.virtual_memory().percent
    logger.info(f"Memory barrier ({context}): {memory_after:.1f}%")
```

#### 3.3 Process Pool Size Optimization
```python
def get_optimal_worker_count(dataset_size):
    """Dynamic worker scaling based on memory and dataset size"""
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    if dataset_size > 1000:  # Large timesplit dataset
        # Conservative: 4GB per worker for large datasets
        max_workers = max(1, int(available_memory_gb / 4))
        return min(max_workers, 6)  # Cap at 6 for large datasets
    else:  # Small polaris dataset
        return min(8, available_memory_gb // 2)  # Existing logic
```

### Phase 4: Data Structure Optimizations

#### 4.1 Lightweight Molecule References
```python
@dataclass
class MoleculeReference:
    """Lightweight reference instead of full molecule object"""
    pdb_id: str
    smiles: str
    file_path: str
    offset: int  # File offset for direct loading
    
    def load_full_molecule(self):
        # Load actual RDKit molecule only when needed
        return load_molecule_at_offset(self.file_path, self.offset)
```

#### 4.2 Memory-Efficient Pipeline State
```python
class LightweightTEMPLPipeline:
    """Memory-optimized pipeline for batch processing"""
    
    def __init__(self):
        # Don't preload large embeddings/caches
        self.embedding_manager = None
        
    def process_target(self, target_pdb, ligand_smiles):
        # Load components only when needed
        if self.embedding_manager is None:
            self.embedding_manager = self._lazy_load_embeddings()
            
        # Process and immediately cleanup
        result = self._run_pipeline_minimal(target_pdb, ligand_smiles)
        
        # Clear intermediate state
        self._clear_intermediate_state()
        
        return result
```

## IMPLEMENTATION PRIORITY

### Critical (Fix OOM immediately)
1. **Shared Cache Pre-loading** - Eliminate per-worker 6GB cache loading
2. **Worker Pool Recycling** - Prevent memory accumulation across batches  
3. **Memory-Mapped Database** - Replace full cache loading with on-demand access

### High Priority (Performance optimization)
4. **Lazy Molecular Loading** - Load only needed molecules per target
5. **Global Memory Monitoring** - Prevent system-wide OOM
6. **Enhanced Memory Barriers** - Better cleanup between chunks

### Medium Priority (Scale optimization)  
7. **Dynamic Worker Scaling** - Adapt worker count to available memory
8. **Lightweight References** - Reduce memory footprint of molecule objects
9. **Memory-Efficient Pipeline** - Optimize TEMPLPipeline state management

## EXPECTED RESULTS

### Memory Usage
- **Before**: 6GB × 8 workers = 48GB+ (OOM at 50 targets)
- **After**: Shared 6GB cache + worker overhead = ~12GB total
- **Scaling**: Linear memory growth instead of exponential

### Performance  
- **Throughput**: Process thousands of targets without OOM
- **Speed**: Faster due to reduced cache loading overhead
- **Reliability**: Consistent memory usage across entire dataset

### Architecture Benefits
- **Polaris Compatibility**: Keep existing small-scale patterns  
- **Timesplit Scalability**: Handle large-scale datasets efficiently
- **Future-Proof**: Foundation for even larger benchmarks

## VALIDATION PLAN

1. **Memory Profiling**: Track memory usage during optimization phases
2. **Benchmark Comparison**: Compare optimized vs original timesplit performance  
3. **Stress Testing**: Run on progressively larger target sets
4. **Regression Testing**: Ensure polaris benchmarks still work correctly

This optimization plan transforms timesplit from an OOM-prone system to a scalable, memory-efficient benchmark that can handle datasets orders of magnitude larger than the current limit.
