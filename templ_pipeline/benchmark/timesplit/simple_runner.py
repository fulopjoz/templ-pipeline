#!/usr/bin/env python3
"""
Simplified time-split benchmark runner for TEMPL pipeline.

This module implements a streamlined time-split benchmarking approach that:
- Uses subprocess calls to `templ run` for memory isolation
- Leverages existing DatasetSplits class for split management
- Follows the proven pattern from run_custom_split_benchmark.py
- Maintains proper time-split data hygiene
- Uses enhanced shared data manager with SharedMemory to prevent memory explosion
"""

import json
import logging
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from tqdm import tqdm

from templ_pipeline.benchmark.runner import LazyMoleculeLoader
from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator

logger = logging.getLogger(__name__)


class SimpleTimeSplitRunner:
    """
    Simplified time-split benchmark runner using subprocess calls to CLI.
    
    This approach provides better memory isolation and follows the proven
    pattern from run_custom_split_benchmark.py while leveraging the existing
    CLI infrastructure and enhanced shared data management.
    """
    
    def __init__(self, data_dir: Optional[str] = None, results_dir: Optional[str] = None, 
                 memory_efficient: bool = True, use_shared_data: bool = True):
        """
        Initialize the simple time-split runner.
        
        Args:
            data_dir: Directory containing benchmark data and splits
            results_dir: Directory for benchmark results
            memory_efficient: Use memory-efficient molecule loading (prevents memory explosion)
            use_shared_data: Use enhanced shared data manager for embeddings and ligands
        """
        self.data_dir = Path(data_dir) if data_dir else self._find_data_directory()
        self.results_dir = Path(results_dir) if results_dir else Path("simple_timesplit_results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize splits directory and molecule loader
        self.splits_dir = self.data_dir / "splits"
        
        # Enhanced shared data management
        self.use_shared_data = use_shared_data
        self.shared_cache_files = {}
        self.shm_manager = None
        self.shared_embedding_cache = None
        
        # Set up shared data for memory efficiency
        if use_shared_data:
            self._setup_shared_data()
            logger.info("Initialized shared data for memory-efficient benchmarking")
        
        # Use memory-efficient molecule loader to prevent explosion
        if memory_efficient:
            # Use lazy loader for memory efficiency
            self.molecule_loader = LazyMoleculeLoader(str(self.data_dir))
        else:
            # Fallback to original loader (not recommended for large datasets)
            from templ_pipeline.benchmark.runner import LazyMoleculeLoader as OriginalLoader
            self.molecule_loader = OriginalLoader(str(self.data_dir))
        
        # Load split PDB IDs
        self._load_splits()
        
        # Pre-compute template sets for memory efficiency
        self._precompute_template_sets()
        
        # Set timestamp for compatibility
        from datetime import datetime
        self.benchmark_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Initialize sophisticated error and skip tracking systems
        self._initialize_tracking_systems()
        
        logger.info(f"Simple TimeSplit runner initialized:")
        logger.info(f"  Data directory: {self.data_dir}")
        logger.info(f"  Results directory: {self.results_dir}")
        logger.info(f"  Memory efficient: {memory_efficient}")
        logger.info(f"  Enhanced shared data: {use_shared_data}")
        logger.info(f"  Error tracking: {'✓' if self.error_tracker else '✗'}")
        logger.info(f"  Skip tracking: {'✓' if self.skip_tracker else '✗'}")

    def _setup_shared_data(self):
        """Set up shared data for memory-efficient benchmarking."""
        try:
            # Create shared embedding cache to prevent memory explosion
            from templ_pipeline.core.utils import create_shared_embedding_cache
            
            # Find embedding file
            embedding_path = self.data_dir / "embeddings" / "templ_protein_embeddings_v1.0.0.npz"
            if not embedding_path.exists():
                logger.warning(f"Embedding file not found: {embedding_path}")
                return
            
            logger.info(f"Creating shared embedding cache from: {embedding_path}")
            
            # Create shared embedding cache
            cache_name = create_shared_embedding_cache(str(embedding_path))
            self.shared_embedding_cache = cache_name
            
            logger.info(f"✓ Created shared embedding cache: {cache_name}")
            
        except Exception as e:
            logger.warning(f"Failed to setup shared data: {e}")
            self.shared_embedding_cache = None

    def _initialize_tracking_systems(self):
        """Initialize sophisticated error and skip tracking systems."""
        try:
            # Initialize error tracking system
            from templ_pipeline.benchmark.error_tracking import BenchmarkErrorTracker
            self.error_tracker = BenchmarkErrorTracker(self.results_dir)
            logger.info("✓ Error tracking system initialized")
        except ImportError:
            logger.warning("Error tracking module not available - using basic error handling")
            self.error_tracker = None
        except Exception as e:
            logger.warning(f"Failed to initialize error tracking: {e}")
            self.error_tracker = None
        
        try:
            # Initialize skip tracking system
            from templ_pipeline.benchmark.skip_tracker import BenchmarkSkipTracker
            self.skip_tracker = BenchmarkSkipTracker(self.results_dir)
            self.skip_tracker.load_existing_skips()  # Load any existing skip records
            logger.info("✓ Skip tracking system initialized")
        except ImportError:
            logger.warning("Skip tracking module not available - using basic skip handling")
            self.skip_tracker = None
        except Exception as e:
            logger.warning(f"Failed to initialize skip tracking: {e}")
            self.skip_tracker = None

    def _find_data_directory(self) -> Path:
        """Find the data directory with timesplit files."""
        potential_paths = [
            Path(__file__).resolve().parent.parent.parent / "data",
            Path.cwd() / "data",
            Path("data"),
        ]
        
        for path in potential_paths:
            if path.exists() and (path / "splits").exists():
                return path
        
        return Path("data")  # Fallback

    def _load_splits(self):
        """Load dataset splits from timesplit files."""
        self.split_files = {
            "train": self.splits_dir / "timesplit_train",
            "val": self.splits_dir / "timesplit_val", 
            "test": self.splits_dir / "timesplit_test"
        }
        
        # Load split PDB IDs
        self.train_pdbs = self._load_pdb_list(self.split_files["train"])
        self.val_pdbs = self._load_pdb_list(self.split_files["val"])
        self.test_pdbs = self._load_pdb_list(self.split_files["test"])
        
        logger.info(f"Loaded splits:")
        logger.info(f"  Train: {len(self.train_pdbs)} PDBs")
        logger.info(f"  Val: {len(self.val_pdbs)} PDBs")
        logger.info(f"  Test: {len(self.test_pdbs)} PDBs")

    def _load_pdb_list(self, filepath: Path) -> Set[str]:
        """Load PDB IDs from a split file."""
        pdb_ids = set()
        if not filepath.exists():
            logger.warning(f"Split file not found: {filepath}")
            return pdb_ids
            
        with open(filepath, 'r') as f:
            for line in f:
                pdb_id = line.strip().lower()
                if pdb_id:
                    pdb_ids.add(pdb_id)
        
        return pdb_ids

    def get_split_pdbs(self, split_name: str) -> Set[str]:
        """Get PDB IDs for a specific split."""
        if split_name == "train":
            return self.train_pdbs
        elif split_name == "val":
            return self.val_pdbs
        elif split_name == "test":
            return self.test_pdbs
        elif split_name == "test_example":
            # Use our test dataset
            return {"1iky", "5eqy"}
        else:
            raise ValueError(f"Unknown split: {split_name}")

    def determine_split(self, pdb_id: str) -> str:
        """Determine which split a PDB ID belongs to."""
        pdb_id = pdb_id.lower()
        
        if pdb_id in self.train_pdbs:
            return "train"
        elif pdb_id in self.val_pdbs:
            return "val"
        elif pdb_id in self.test_pdbs:
            return "test"
        elif pdb_id in {"1iky", "5eqy"}:
            return "test_example"
        else:
            raise ValueError(f"PDB {pdb_id} not found in any split")

    def get_allowed_templates_for_split(self, target_split: str, target_pdb: str) -> Set[str]:
        """
        Get allowed template PDB IDs for a target split based on time-split rules.
        
        Time-split rules:
        - Test: can use train + val templates (no future information)
        - Val: can only use train templates (no future information)  
        - Train: uses leave-one-out (other train molecules as templates)
        
        Args:
            target_split: Split of the target molecule ('train', 'val', 'test')
            target_pdb: PDB ID of the target (for leave-one-out exclusion)
            
        Returns:
            Set of PDB IDs allowed as templates
        """
        # Get the pre-computed allowed template set for this split
        if target_split == "test":
            allowed_templates = self._test_allowed_templates.copy()
        elif target_split == "val":
            allowed_templates = self._val_allowed_templates.copy()
        elif target_split == "train":
            allowed_templates = self._train_allowed_templates.copy()
        elif target_split == "test_example":
            # For test_example, use the same set as templates (simple case)
            allowed_templates = {"1iky", "5eqy"}
        else:
            raise ValueError(f"Unknown target split: {target_split}")
        
        # Only apply leave-one-out for train split (since val/test are disjoint from their template sets)
        if target_split == "train":
            allowed_templates.discard(target_pdb.upper())
            logger.debug(f"Applied LOO for train target {target_pdb}: excluded from {len(self._train_allowed_templates)} templates")
        else:
            logger.debug(f"No LOO needed for {target_split} target {target_pdb}: target not in template set")
        
        return allowed_templates

    def validate_cli_command(self, cmd: List[str]) -> bool:
        """
        Validate CLI command structure before execution.
        
        Args:
            cmd: Command list to validate
            
        Returns:
            True if command structure is valid
        """
        try:
            # Check basic structure: ["templ", "--output-dir", "path", "run", ...]
            if len(cmd) < 4:
                logger.error(f"Command too short: {cmd}")
                return False
                
            if cmd[0] != "templ":
                logger.error(f"Command should start with 'templ', got: {cmd[0]}")
                return False
                
            # Find the subcommand position
            subcommand_pos = None
            for i, arg in enumerate(cmd):
                if arg in ["run", "embed", "find-templates", "generate-poses", "benchmark"]:
                    subcommand_pos = i
                    break
                    
            if subcommand_pos is None:
                logger.error(f"No valid subcommand found in: {cmd}")
                return False
                
            # Validate that global parameters (like --output-dir) come before subcommand
            global_params = ["--output-dir", "--log-level", "--verbosity", "--seed"]
            for i in range(1, subcommand_pos):
                arg = cmd[i]
                if arg.startswith("--") and arg not in global_params:
                    logger.warning(f"Parameter {arg} might be a subcommand parameter placed before subcommand")
                    
            logger.debug(f"CLI command validation passed: {' '.join(cmd[:6])}...")
            return True
            
        except Exception as e:
            logger.error(f"CLI command validation failed: {e}")
            return False

    def classify_processing_stage(self, error_msg: str, success: bool = False) -> Tuple[str, bool]:
        """
        Classify error into processing stage and determine if it affects success rate.
        
        Processing Stages:
        1. pre_pipeline_excluded: Data availability issues (missing files, invalid data)
        2. pipeline_filtered: Molecule validation/quality filters (large peptides, etc.)
        3. pipeline_attempted: Actual algorithm processing (timeouts, pose failures, etc.)
        
        Args:
            error_msg: Error message to classify
            success: Whether the processing was successful
            
        Returns:
            Tuple of (processing_stage, affects_pipeline_success_rate)
        """
        if success:
            return "pipeline_attempted", True
            
        if not error_msg:
            return "pipeline_attempted", True
            
        error_msg_lower = error_msg.lower()
        
        # Pre-pipeline exclusions (data availability/quality issues)
        # These should NOT affect pipeline success rates
        pre_pipeline_indicators = [
            "could not load ligand smiles",
            "ligand data not found",
            "ligand smiles data not found",
            "protein file not found",
            "pdb file not found",
            "file not found",
            "embedding file not found",
            "template file not found",
            "crystal structure missing",
            "crystal ligand not found",
            "invalid smiles",
            "invalid molecule structure",
            "cli command validation failed",
            "command validation failed"
        ]
        
        for indicator in pre_pipeline_indicators:
            if indicator in error_msg_lower:
                return "pre_pipeline_excluded", False
                
        # Pipeline filtering exclusions (validation rules, quality filters)  
        # These should NOT affect pipeline success rates
        pipeline_filter_indicators = [
            "large peptide",
            "rhenium complex",
            "complex polysaccharide", 
            "validation failed",
            "molecule validation failed",
            "sanitization failed",
            "molecule sanitization failed",
            "invalid molecule",
            "poor quality crystal",
            "skipped"
        ]
        
        for indicator in pipeline_filter_indicators:
            if indicator in error_msg_lower:
                return "pipeline_filtered", False
                
        # Pipeline execution failures (algorithm processing issues)
        # These SHOULD affect pipeline success rates
        pipeline_execution_indicators = [
            "timeout",
            "pose generation failed", 
            "rmsd calculation failed",
            "conformer generation failed",
            "alignment failed",
            "molecular alignment failed",
            "mcs calculation failed",
            "mcs failed",
            "embedding failed",
            "template processing failed",
            "cli returned",
            "subprocess failed",
            "pipeline error",
            "no poses generated",
            "memory error",
            "force field failed",
            "optimization failed",
            "geometry validation failed",
            "connectivity failed"
        ]
        
        for indicator in pipeline_execution_indicators:
            if indicator in error_msg_lower:
                return "pipeline_attempted", True
                
        # Default: treat unknown errors as pipeline execution failures
        # This ensures we don't accidentally exclude real algorithm failures
        logger.warning(f"Unknown error type for stage classification: {error_msg[:100]}...")
        return "pipeline_attempted", True

    def _analyze_cli_success(self, stdout: str, target_pdb: str) -> Tuple[str, bool]:
        """
        Analyze CLI success cases to determine the actual processing stage.
        
        Uses the enhanced CLI JSON output with pipeline_stage field to accurately
        classify processing stages, ensuring correct success rate calculations.
        
        Args:
            stdout: CLI stdout output
            target_pdb: Target PDB ID for logging
            
        Returns:
            Tuple of (processing_stage, affects_pipeline_success_rate)
        """
        try:
            # Extract CLI JSON result from stdout
            json_start = stdout.find("TEMPL_JSON_RESULT:")
            if json_start != -1:
                json_start += len("TEMPL_JSON_RESULT:")
                json_end = stdout.find("\n", json_start)
                if json_end == -1:
                    json_end = len(stdout)
                
                json_str = stdout[json_start:json_end].strip()
                cli_result = json.loads(json_str)
                
                # PRIORITY 1: Use new pipeline_stage field if available (preferred)
                pipeline_stage = cli_result.get("pipeline_stage")
                made_it_to_mcs = cli_result.get("made_it_to_mcs", False)
                
                if pipeline_stage:
                    # Use the accurate pipeline stage from enhanced CLI
                    affects_success_rate = (pipeline_stage == "pipeline_attempted")
                    logger.debug(f"{target_pdb}: CLI reports pipeline_stage='{pipeline_stage}', made_it_to_mcs={made_it_to_mcs}")
                    return pipeline_stage, affects_success_rate
                
                # FALLBACK: Legacy logic for older CLI versions
                cli_success = cli_result.get("success", False)
                total_templates_in_db = cli_result.get("total_templates_in_database", 0)
                
                if not cli_success and total_templates_in_db == 0:
                    # Database is empty - data availability issue
                    logger.debug(f"{target_pdb}: CLI reports database_empty (0 templates) [LEGACY LOGIC]")
                    return "pre_pipeline_excluded", False
                elif not cli_success:
                    # CLI failed for other reasons - algorithm issue  
                    logger.debug(f"{target_pdb}: CLI reports failure with {total_templates_in_db} templates [LEGACY LOGIC]")
                    return "pipeline_attempted", True
                else:
                    # CLI succeeded - normal pipeline processing
                    logger.debug(f"{target_pdb}: CLI succeeded with {total_templates_in_db} templates [LEGACY LOGIC]")
                    return "pipeline_attempted", True
                    
        except (ValueError, KeyError, AttributeError, json.JSONDecodeError) as e:
            logger.debug(f"Failed to parse CLI JSON output for {target_pdb}: {e}")
        
        # Fallback: assume successful pipeline processing
        return "pipeline_attempted", True

    def run_single_target_subprocess(self, 
                                   target_pdb: str,
                                   allowed_templates: Set[str],
                                   n_conformers: int = 200,
                                   timeout: int = 600) -> Dict:
        """
        Run benchmark for a single target using subprocess call to CLI.
        
        Args:
            target_pdb: Target PDB ID
            allowed_templates: Set of allowed template PDB IDs
            n_conformers: Number of conformers to generate
            timeout: Timeout in seconds
            
        Returns:
            Dictionary with benchmark results
        """
        start_time = time.time()
        
        try:
            # Load ligand SMILES for this PDB
            ligand_smiles, _ = self.molecule_loader.get_ligand_data(target_pdb)
            if not ligand_smiles:
                error_msg = f"Could not load ligand SMILES for {target_pdb}"
                processing_stage, affects_success_rate = self.classify_processing_stage(error_msg, False)
                
                # Use sophisticated tracking systems if available
                if self.error_tracker:
                    self.error_tracker.record_target_failure(target_pdb, error_msg, {
                        "processing_stage": processing_stage,
                        "affects_success_rate": affects_success_rate,
                        "component": "ligand_loading"
                    })
                
                if self.skip_tracker and processing_stage in ["pre_pipeline_excluded", "pipeline_filtered"]:
                    self.skip_tracker.track_skip(
                        target_pdb, 
                        "ligand_data_missing",
                        error_msg
                    )
                
                return {
                    "success": False,
                    "target_pdb": target_pdb,
                    "error": error_msg,
                    "processing_stage": processing_stage,
                    "affects_pipeline_success_rate": affects_success_rate,
                    "runtime_total": time.time() - start_time,
                    "rmsd_values": {},  # Empty RMSD values for SMILES loading error
                }
            
            # Build CLI command with enhanced shared data parameters
            cmd = [
                "templ", 
                "--output-dir", str(self.results_dir / "poses" / target_pdb.lower()),
                "run",
                "--protein-pdb-id", target_pdb,
                "--ligand-smiles", ligand_smiles,
                "--num-conformers", str(n_conformers),
                "--allowed-pdb-ids", ",".join(sorted(allowed_templates)),
            ]
            
            # Add shared embedding cache parameter if available
            if self.shared_embedding_cache:
                cmd.extend(["--shared-embedding-cache", self.shared_embedding_cache])
            
            # Validate CLI command structure
            if not self.validate_cli_command(cmd):
                error_msg = f"CLI command validation failed for {target_pdb}"
                processing_stage, affects_success_rate = self.classify_processing_stage(error_msg, False)
                
                # Use sophisticated tracking systems if available
                if self.error_tracker:
                    self.error_tracker.record_target_failure(target_pdb, error_msg, {
                        "processing_stage": processing_stage,
                        "affects_success_rate": affects_success_rate,
                        "component": "cli_validation"
                    })
                
                return {
                    "success": False,
                    "target_pdb": target_pdb,
                    "error": error_msg,
                    "processing_stage": processing_stage,
                    "affects_pipeline_success_rate": affects_success_rate,
                    "runtime_total": time.time() - start_time,
                    "rmsd_values": {},  # Empty RMSD values for validation error
                }
            
            logger.info(f"Running: {' '.join(cmd[:6])} ... (with {len(allowed_templates)} allowed templates)")
            
            # Run subprocess with timeout
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(self.data_dir.parent)  # Run from project root
            )
            
            runtime_total = time.time() - start_time
            
            if result.returncode == 0:
                # Parse JSON output from CLI stdout for RMSD data
                rmsd_values = {}
                try:
                    # Look for the JSON result line in stdout
                    for line in result.stdout.split('\n'):
                        if line.startswith("TEMPL_JSON_RESULT:"):
                            json_str = line[len("TEMPL_JSON_RESULT:"):]
                            json_data = json.loads(json_str)
                            rmsd_values = json_data.get("rmsd_values", {})
                            break
                except Exception as e:
                    logger.debug(f"Failed to parse JSON output for {target_pdb}: {e}")
                
                # Analyze CLI JSON output to determine processing stage
                # CLI exits with code 0 even for database_empty cases
                processing_stage, affects_success_rate = self._analyze_cli_success(result.stdout, target_pdb)
                
                # Record successful processing in tracking systems
                if self.error_tracker:
                    self.error_tracker.record_target_success(target_pdb)
                
                return {
                    "success": True,
                    "target_pdb": target_pdb,
                    "processing_stage": processing_stage,
                    "affects_pipeline_success_rate": affects_success_rate,
                    "allowed_templates_count": len(allowed_templates),
                    "runtime_total": runtime_total,
                    "stdout": result.stdout,
                    "rmsd_values": rmsd_values,  # Add RMSD data for summary generation
                }
            else:
                # CLI execution failed - analyze stderr to determine processing stage
                error_msg = f"CLI returned {result.returncode}"
                if result.stderr:
                    error_msg += f": {result.stderr.strip()}"
                
                processing_stage, affects_success_rate = self.classify_processing_stage(error_msg, False)
                
                # Use sophisticated tracking systems if available
                if self.error_tracker:
                    self.error_tracker.record_target_failure(target_pdb, error_msg, {
                        "processing_stage": processing_stage,
                        "affects_success_rate": affects_success_rate,
                        "component": "cli_execution",
                        "return_code": result.returncode,
                        "stderr": result.stderr[:200] if result.stderr else None  # Truncate for storage
                    })
                
                if self.skip_tracker and processing_stage in ["pre_pipeline_excluded", "pipeline_filtered"]:
                    self.skip_tracker.track_skip(
                        target_pdb,
                        "cli_execution_failed",
                        error_msg
                    )
                
                return {
                    "success": False,
                    "target_pdb": target_pdb,
                    "error": error_msg,
                    "processing_stage": processing_stage,
                    "affects_pipeline_success_rate": affects_success_rate,
                    "stderr": result.stderr,
                    "runtime_total": runtime_total,
                    "rmsd_values": {},  # Empty RMSD values for failed runs
                }
                
        except subprocess.TimeoutExpired:
            # Timeout is a pipeline execution failure - should affect success rate
            error_msg = f"Timeout after {timeout}s"
            processing_stage, affects_success_rate = self.classify_processing_stage(error_msg, False)
            
            # Use sophisticated tracking systems for timeout tracking
            if self.error_tracker:
                self.error_tracker.record_target_failure(target_pdb, error_msg, {
                    "processing_stage": processing_stage,
                    "affects_success_rate": affects_success_rate,
                    "component": "subprocess_timeout",
                    "timeout_duration": timeout
                })
            
            return {
                "success": False,
                "target_pdb": target_pdb,
                "error": error_msg,
                "processing_stage": processing_stage,
                "affects_pipeline_success_rate": affects_success_rate,
                "runtime_total": time.time() - start_time,
                "rmsd_values": {},  # Empty RMSD values for timeout
            }
        except Exception as e:
            # Generic exception - classify based on error message
            error_msg = str(e)
            processing_stage, affects_success_rate = self.classify_processing_stage(error_msg, False)
            
            # Use sophisticated tracking systems for generic exceptions
            if self.error_tracker:
                self.error_tracker.record_target_failure(target_pdb, error_msg, {
                    "processing_stage": processing_stage,
                    "affects_success_rate": affects_success_rate,
                    "component": "generic_exception",
                    "exception_type": type(e).__name__
                })
            
            if self.skip_tracker and processing_stage in ["pre_pipeline_excluded", "pipeline_filtered"]:
                self.skip_tracker.track_skip(
                    target_pdb,
                    "pipeline_error",
                    error_msg
                )
            
            return {
                "success": False,
                "target_pdb": target_pdb,
                "error": error_msg,
                "processing_stage": processing_stage,
                "affects_pipeline_success_rate": affects_success_rate,
                "runtime_total": time.time() - start_time,
                "rmsd_values": {},  # Empty RMSD values for exceptions
            }

    def run_split_benchmark(self,
                           split_name: str,
                           n_workers: int = 4,
                           n_conformers: int = 200,
                           max_pdbs: Optional[int] = None,
                           timeout: int = 600) -> Dict:
        """
        Run benchmark for a complete split using subprocess parallelization.
        
        Args:
            split_name: Name of split to run ('train', 'val', 'test')
            n_workers: Number of parallel workers
            n_conformers: Number of conformers per molecule
            max_pdbs: Maximum number of PDBs to process (for testing)
            timeout: Timeout per molecule in seconds
            
        Returns:
            Dictionary with complete benchmark results
        """
        # Load target PDBs for this split
        target_pdbs = list(self.get_split_pdbs(split_name))
        
        if max_pdbs and max_pdbs > 0:
            target_pdbs = target_pdbs[:max_pdbs]
            logger.info(f"Limited to {len(target_pdbs)} PDBs for testing")
        
        logger.info(f"Running {split_name} split with {len(target_pdbs)} targets using {n_workers} workers")
        
        # Create output file for streaming results
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_jsonl = self.results_dir / f"results_{split_name}_{timestamp}.jsonl"
        
        # Initialize result tracking with processing stage awareness
        processed_count = 0
        success_count = 0
        failed_count = 0
        
        # Stage-aware tracking for correct success rate calculation
        pre_pipeline_excluded_count = 0
        pipeline_filtered_count = 0
        pipeline_attempted_count = 0
        pipeline_success_count = 0
        
        # Process targets in batches to avoid memory accumulation
        batch_size = min(100, len(target_pdbs))  # Process in batches of 100 or less
        total_batches = (len(target_pdbs) + batch_size - 1) // batch_size
        
        logger.info(f"Processing {len(target_pdbs)} PDBs in {total_batches} batches of {batch_size}")
        
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(target_pdbs))
            batch_pdbs = target_pdbs[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} ({len(batch_pdbs)} PDBs)")
            
            # Process batch with parallel execution
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                # Submit jobs for this batch
                future_to_pdb = {}
                
                for target_pdb in batch_pdbs:
                    # Determine target split and get allowed templates
                    target_split = self.determine_split(target_pdb)
                    allowed_templates = self.get_allowed_templates_for_split(target_split, target_pdb)
                    
                    logger.debug(f"Submitting {target_pdb} ({target_split}) with {len(allowed_templates)} templates")
                    
                    future = executor.submit(
                        self.run_single_target_subprocess,
                        target_pdb=target_pdb,
                        allowed_templates=allowed_templates,
                        n_conformers=n_conformers,
                        timeout=timeout
                    )
                    future_to_pdb[future] = target_pdb
                
                # Collect results for this batch
                desc = f"{split_name.title()} Split (Batch {batch_idx + 1}/{total_batches})"
                progress_bar = tqdm(total=len(future_to_pdb), desc=desc, ncols=100)
                
                for future in as_completed(future_to_pdb):
                    target_pdb = future_to_pdb[future]
                    
                    try:
                        result = future.result(timeout=timeout + 30)
                        
                        # Stream result to file
                        with open(output_jsonl, 'a') as f:
                            json.dump(result, f)
                            f.write('\n')
                        
                        # Update counters with processing stage awareness
                        processed_count += 1
                        
                        # Track processing stage for correct success rate calculation
                        processing_stage = result.get("processing_stage", "pipeline_attempted")
                        affects_success_rate = result.get("affects_pipeline_success_rate", True)
                        
                        if processing_stage == "pre_pipeline_excluded":
                            pre_pipeline_excluded_count += 1
                        elif processing_stage == "pipeline_filtered":
                            pipeline_filtered_count += 1
                        elif processing_stage == "pipeline_attempted":
                            pipeline_attempted_count += 1
                            if result.get("success"):
                                pipeline_success_count += 1
                        
                        # Legacy counters for compatibility
                        if result.get("success"):
                            success_count += 1
                        else:
                            failed_count += 1
                            
                    except Exception as e:
                        logger.error(f"Future failed for {target_pdb}: {e}")
                        # Future execution failure - classify as pipeline execution failure
                        error_msg = f"Future execution failed: {str(e)}"
                        processing_stage, affects_success_rate = self.classify_processing_stage(error_msg, False)
                        error_result = {
                            "success": False,
                            "target_pdb": target_pdb,
                            "error": error_msg,
                            "processing_stage": processing_stage,
                            "affects_pipeline_success_rate": affects_success_rate,
                        }
                        
                        with open(output_jsonl, 'a') as f:
                            json.dump(error_result, f)
                            f.write('\n')
                        
                        # Update counters for error result with processing stage awareness
                        processed_count += 1
                        failed_count += 1
                        
                        # Track processing stage for error result
                        processing_stage = error_result.get("processing_stage", "pipeline_attempted")
                        if processing_stage == "pre_pipeline_excluded":
                            pre_pipeline_excluded_count += 1
                        elif processing_stage == "pipeline_filtered":
                            pipeline_filtered_count += 1
                        elif processing_stage == "pipeline_attempted":
                            pipeline_attempted_count += 1
                    
                    # Update progress with pipeline success rate (main metric)
                    pipeline_success_rate = (pipeline_success_count / pipeline_attempted_count * 100) if pipeline_attempted_count > 0 else 0
                    overall_success_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
                    progress_bar.set_postfix({
                        "pipeline": f"{pipeline_success_rate:.1f}%",
                        "overall": f"{overall_success_rate:.1f}%",
                        "errors": failed_count
                    })
                    progress_bar.update(1)
                
                progress_bar.close()
                
                # Clear batch data to free memory
                future_to_pdb.clear()
                
                # Force garbage collection after each batch
                import gc
                gc.collect()
                
                logger.info(f"Batch {batch_idx + 1} completed. Total progress: {processed_count}/{len(target_pdbs)}")
        
        # Calculate stage-aware success rates
        pipeline_success_rate = (pipeline_success_count / pipeline_attempted_count * 100) if pipeline_attempted_count > 0 else 0
        data_completeness_rate = ((processed_count - pre_pipeline_excluded_count) / processed_count * 100) if processed_count > 0 else 0
        molecule_acceptance_rate = (pipeline_attempted_count / (processed_count - pre_pipeline_excluded_count) * 100) if (processed_count - pre_pipeline_excluded_count) > 0 else 0
        overall_completion_rate = (success_count / processed_count * 100) if processed_count > 0 else 0
        
        # Generate enhanced summary with stage-aware metrics
        summary = {
            "split": split_name,
            "total_targets": len(target_pdbs),
            "processed": processed_count,
            "successful": success_count,
            "failed": failed_count,
            
            # Multi-tier success rate metrics (MAIN IMPROVEMENT)
            "success_rates": {
                "pipeline_success_rate": pipeline_success_rate,      # Main metric: algorithm performance
                "data_completeness_rate": data_completeness_rate,    # Data quality metric
                "molecule_acceptance_rate": molecule_acceptance_rate, # Filtering effectiveness
                "overall_completion_rate": overall_completion_rate   # End-to-end rate
            },
            
            # Processing stage breakdown
            "processing_breakdown": {
                "total_targets": len(target_pdbs),
                "pre_pipeline_excluded": pre_pipeline_excluded_count,   # Missing files, data issues
                "pipeline_filtered": pipeline_filtered_count,           # Validation rules, quality filters
                "pipeline_attempted": pipeline_attempted_count,         # Actually processed by algorithm
                "pipeline_successful": pipeline_success_count,         # Completed with RMSD values
                "pipeline_failed": pipeline_attempted_count - pipeline_success_count  # Timeouts, pose failures
            },
            
            # Legacy fields for backward compatibility
            "success_rate": pipeline_success_rate,  # Now shows pipeline success rate (main metric)
            
            "results_file": str(output_jsonl),
            "timestamp": timestamp,
            "benchmark_info": {
                "name": "simple_timesplit_benchmark",
                "split": split_name,
                "n_conformers": n_conformers,
                "n_workers": n_workers,
                "timeout": timeout,
                "enhanced_shared_data_used": self.use_shared_data,
                "stage_aware_reporting": True  # Flag indicating enhanced reporting
            }
        }
        
        # Save summary
        summary_file = self.results_dir / f"summary_{split_name}_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Split {split_name} completed:")
        logger.info(f"  Total targets: {len(target_pdbs)}")
        logger.info(f"  Processing breakdown:")
        logger.info(f"    Pre-pipeline excluded: {pre_pipeline_excluded_count}")
        logger.info(f"    Pipeline filtered: {pipeline_filtered_count}")
        logger.info(f"    Pipeline attempted: {pipeline_attempted_count}")
        logger.info(f"    Pipeline successful: {pipeline_success_count}")
        logger.info(f"  Success rates:")
        logger.info(f"    Pipeline success rate: {pipeline_success_rate:.1f}% (main metric)")
        logger.info(f"    Data completeness rate: {data_completeness_rate:.1f}%")
        logger.info(f"    Overall completion rate: {overall_completion_rate:.1f}%")
        logger.info(f"  Results: {output_jsonl}")
        
        # Generate tracking system summaries if available
        self._generate_tracking_summaries(split_name)
        
        # Generate detailed 2A/5A success rate summaries using BenchmarkSummaryGenerator
        self._generate_detailed_summaries(output_jsonl, split_name, timestamp)
        
        return summary
    
    def _generate_tracking_summaries(self, split_name: str):
        """Generate and save tracking system summaries."""
        try:
            if self.error_tracker:
                error_summary = self.error_tracker.get_summary()
                logger.info(f"Error tracking summary:")
                logger.info(f"  Total errors recorded: {error_summary['total_errors']}")
                if error_summary['error_breakdown']:
                    logger.info(f"  Error breakdown:")
                    for error_type, count in error_summary['error_breakdown'].items():
                        logger.info(f"    {error_type}: {count}")
            
            if self.skip_tracker:
                skip_stats = self.skip_tracker.get_formatted_skip_statistics()
                if skip_stats['total_skipped'] > 0:
                    logger.info(f"Skip tracking summary:")
                    logger.info(f"  {skip_stats['formatted_summary']}")
                    
                    # Generate and save detailed skip summary
                    skip_summary_file = self.skip_tracker.save_summary(f"skip_summary_{split_name}_{self.benchmark_timestamp}.json")
                    logger.info(f"  Detailed skip summary: {skip_summary_file}")
                else:
                    logger.info("Skip tracking: No molecules were skipped")
                    
        except Exception as e:
            logger.warning(f"Failed to generate tracking summaries: {e}")
    
    def _generate_detailed_summaries(self, results_jsonl: Path, split_name: str, timestamp: str):
        """Generate detailed 2A/5A success rate summaries using BenchmarkSummaryGenerator."""
        try:
            logger.info(f"Generating detailed 2A/5A success rate summaries for {split_name} split...")
            
            # Load JSONL results
            results_data = []
            with open(results_jsonl, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        results_data.append(json.loads(line))
            
            if not results_data:
                logger.warning(f"No results data found in {results_jsonl}")
                return
            
            # Initialize BenchmarkSummaryGenerator
            summary_generator = BenchmarkSummaryGenerator()
            
            # Generate unified summary with 2A/5A success rates
            unified_summary = summary_generator.generate_unified_summary(
                results_data=results_data, 
                benchmark_type="timesplit",
                output_format="dict"
            )
            
            # Create summaries directory if it doesn't exist
            summaries_dir = self.results_dir / "summaries"
            summaries_dir.mkdir(exist_ok=True)
            
            # Save detailed JSON summary with 2A/5A success rates
            detailed_summary_file = summaries_dir / f"detailed_summary_{split_name}_{timestamp}.json"
            with open(detailed_summary_file, 'w') as f:
                json.dump(unified_summary, f, indent=2, default=str)
            
            # Generate and save CSV summary if pandas is available
            try:
                csv_summary = summary_generator.generate_unified_summary(
                    results_data=results_data, 
                    benchmark_type="timesplit",
                    output_format="pandas"
                )
                if csv_summary is not None and hasattr(csv_summary, 'to_csv'):
                    csv_file = summaries_dir / f"benchmark_summary_{split_name}_{timestamp}.csv"
                    csv_summary.to_csv(csv_file, index=False)
                    logger.info(f"✓ CSV summary saved: {csv_file}")
            except Exception as e:
                logger.warning(f"Could not generate CSV summary: {e}")
            
            # Log success rates from the detailed summary
            if isinstance(unified_summary, dict) and "summary" in unified_summary:
                for row in unified_summary["summary"]:
                    success_rate_2A = row.get("Success_Rate_2A", "N/A")
                    success_rate_5A = row.get("Success_Rate_5A", "N/A") 
                    pipeline_attempted = row.get("Pipeline_Attempted", "N/A")
                    logger.info(f"✓ 2A Success Rate: {success_rate_2A} (from {pipeline_attempted} attempted)")
                    logger.info(f"✓ 5A Success Rate: {success_rate_5A} (from {pipeline_attempted} attempted)")
            
            logger.info(f"✓ Detailed summaries with 2A/5A success rates saved to: {summaries_dir}")
            logger.info(f"  - JSON summary: {detailed_summary_file}")
            
        except Exception as e:
            logger.error(f"Failed to generate detailed summaries: {e}")
            import traceback
            traceback.print_exc()
    
    def cleanup(self):
        """Clean up enhanced shared data and temporary files."""
        logger.info("Starting SimpleTimeSplitRunner cleanup...")
        
        try:
            # Clean up shared embedding cache
            if self.shared_embedding_cache:
                from templ_pipeline.core.utils import cleanup_shared_embedding_cache
                logger.info(f"Cleaning up shared embedding cache: {self.shared_embedding_cache}")
                cleanup_shared_embedding_cache(self.shared_embedding_cache)
                self.shared_embedding_cache = None
                logger.info("✓ Shared embedding cache cleaned up")
            else:
                logger.debug("No shared embedding cache to clean up")
            
            
            # Clean up shared memory manager
            if self.shm_manager:
                logger.info("Cleaning up shared memory manager...")
                self.shm_manager.cleanup_all()
                self.shm_manager = None
                logger.info("✓ Shared memory manager cleaned up")
                
            # Additional cleanup for any remaining shared memory objects
            try:
                import multiprocessing.resource_tracker as rt
                
                # Force cleanup of any remaining shared memory objects
                try:
                    cleanup_callbacks = getattr(rt, '_CLEANUP_CALLBACKS', None)
                    if cleanup_callbacks:
                        for callback in cleanup_callbacks:
                            try:
                                callback()
                            except Exception:
                                pass
                except (AttributeError, TypeError):
                    # _CLEANUP_CALLBACKS may not exist or be accessible in all Python versions
                    pass
                            
            except Exception as e:
                logger.debug(f"Additional shared memory cleanup failed: {e}")
                
            logger.info("✓ SimpleTimeSplitRunner cleanup completed")
            
        except Exception as e:
            logger.warning(f"Failed to cleanup SimpleTimeSplitRunner: {e}")
            
        finally:
            # Reset shared data flags
            self.use_shared_data = False
            self.shared_cache_files = {}

    def _precompute_template_sets(self):
        """Pre-compute template sets to avoid repeated set creation during benchmark."""
        logger.info("Pre-computing template sets for memory efficiency...")
        
        # Pre-compute base template sets
        self._train_templates = self.get_split_pdbs("train")
        self._val_templates = self.get_split_pdbs("val")
        self._test_templates = self.get_split_pdbs("test")
        
        # Pre-compute allowed template sets for each split
        # Test: can use train + val templates (no future information)
        self._test_allowed_templates = self._train_templates.union(self._val_templates)
        # Val: can only use train templates (no future information)
        self._val_allowed_templates = self._train_templates.copy()
        # Train: uses leave-one-out (other train molecules as templates)
        self._train_allowed_templates = self._train_templates.copy()
        
        logger.info(f"Pre-computed template sets:")
        logger.info(f"  Test allowed templates: {len(self._test_allowed_templates)} (train+val)")
        logger.info(f"  Val allowed templates: {len(self._val_allowed_templates)} (train only)")
        logger.info(f"  Train allowed templates: {len(self._train_allowed_templates)} (train only, LOO applied)")

    def create_test_dataset(self):
        """Create a test dataset using available example files."""
        logger.info("Creating test dataset using available example files...")
        
        # Use the example files we know exist
        test_pdbs = {"1iky", "5eqy"}  # These have protein files in data/example/
        
        # Create temporary split files for testing
        test_split_file = self.splits_dir / "test_example"
        with open(test_split_file, 'w') as f:
            for pdb_id in test_pdbs:
                f.write(f"{pdb_id}\n")
        
        # Update the test_pdbs to use our test dataset
        self.test_pdbs = test_pdbs
        self._test_templates = test_pdbs
        self._test_allowed_templates = test_pdbs.copy()
        
        logger.info(f"Created test dataset with {len(test_pdbs)} PDBs: {test_pdbs}")
        return test_pdbs


def main():
    """CLI entry point for simple timesplit runner."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple TimeSplit Benchmark Runner")
    parser.add_argument("split", choices=["train", "val", "test"], help="Split to run")
    parser.add_argument("--data-dir", type=str, help="Data directory")
    parser.add_argument("--results-dir", type=str, help="Results directory")
    parser.add_argument("--n-workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--n-conformers", type=int, default=200, help="Number of conformers")
    parser.add_argument("--max-pdbs", type=int, help="Max PDBs to process (for testing)")
    parser.add_argument("--timeout", type=int, default=600, help="Timeout per target")
    parser.add_argument("--use-enhanced-shared-data", action="store_true", help="Use enhanced shared data manager")
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Create runner and run benchmark
    runner = SimpleTimeSplitRunner(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        use_shared_data=args.use_enhanced_shared_data
    )
    
    try:
        summary = runner.run_split_benchmark(
            split_name=args.split,
            n_workers=args.n_workers,
            n_conformers=args.n_conformers,
            max_pdbs=args.max_pdbs,
            timeout=args.timeout
        )
        
        print(f"Benchmark completed with {summary['success_rate']:.1f}% success rate")
        
    finally:
        # Clean up enhanced shared data
        runner.cleanup()


if __name__ == "__main__":
    main()