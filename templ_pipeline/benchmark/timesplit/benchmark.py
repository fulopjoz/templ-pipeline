#!/usr/bin/env python3
"""
Time-split benchmark main entry point for TEMPL pipeline.

This module provides the main CLI entry point for time-split benchmarking,
with proper workspace organization and integration with the unified
benchmark infrastructure.

Usage:
    From CLI: templ benchmark time-split [options]
    From code: from templ_pipeline.benchmark.timesplit import run_timesplit_benchmark
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Import the runner and infrastructure
from .timesplit_runner import TimeSplitBenchmarkRunner
from templ_pipeline.core.hardware import get_suggested_worker_config

logger = logging.getLogger(__name__)


def setup_workspace_directory(workspace_dir: Path) -> Dict[str, Path]:
    """
    Set up organized workspace directory structure matching polaris.
    
    Args:
        workspace_dir: Base workspace directory
        
    Returns:
        Dictionary of organized subdirectories
    """
    # Create subdirectory structure
    subdirs = {
        "raw_results": workspace_dir / "raw_results" / "timesplit",
        "summaries": workspace_dir / "summaries",
        "logs": workspace_dir / "logs",
        "poses": workspace_dir / "raw_results" / "timesplit" / "poses",
    }
    
    # Create all directories
    for dir_path in subdirs.values():
        dir_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Workspace organized at: {workspace_dir}")
    for name, path in subdirs.items():
        logger.info(f"  {name}: {path}")
    
    return subdirs


def run_timesplit_benchmark(
    splits_to_run: List[str] = None,
    n_workers: int = None,
    n_conformers: int = 200,
    template_knn: int = 100,
    max_pdbs: Optional[int] = None,
    data_dir: str = None,
    results_dir: str = None,
    poses_output_dir: str = None,
    similarity_threshold: float = None,
    timeout: int = 600,
    quiet: bool = False,
    # Ablation study parameters
    unconstrained: bool = False,
    align_metric: str = "combo",
    enable_optimization: bool = False,
    no_realign: bool = False,
    # Advanced options
    per_worker_ram_gb: float = 4.0,
) -> Dict:
    """
    Main entry point for time-split benchmark execution.
    
    Args:
        splits_to_run: List of splits to evaluate ['train', 'val', 'test']
        n_workers: Number of parallel workers
        n_conformers: Number of conformers per molecule
        template_knn: Number of template neighbors
        max_pdbs: Maximum PDBs per split (for testing)
        data_dir: Data directory path
        results_dir: Results output directory
        poses_output_dir: Poses output directory
        similarity_threshold: Similarity threshold (overrides KNN)
        timeout: Timeout per molecule in seconds
        quiet: Suppress progress output
        unconstrained: Skip MCS and constrained embedding
        align_metric: Alignment metric ('shape', 'color', 'combo')
        enable_optimization: Enable force field optimization
        no_realign: Disable pose realignment
        per_worker_ram_gb: RAM limit per worker process
        
    Returns:
        Dictionary with benchmark results and summary
    """
    # Set defaults
    if splits_to_run is None:
        splits_to_run = ["test"]  # Default to test split only
    
    if n_workers is None:
        hardware_config = get_suggested_worker_config()
        n_workers = hardware_config["n_workers"]
    
    # Set up results directory
    if results_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"timesplit_results_{timestamp}"
    
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Starting time-split benchmark:")
    logger.info(f"  Splits: {', '.join(splits_to_run)}")
    logger.info(f"  Workers: {n_workers}")
    logger.info(f"  Conformers: {n_conformers}")
    logger.info(f"  Results: {results_path}")
    
    # Initialize simplified benchmark runner
    from templ_pipeline.benchmark.timesplit.simple_runner import SimpleTimeSplitRunner
    logger.info("Creating SimpleTimeSplitRunner with enhanced shared data...")
    runner = SimpleTimeSplitRunner(
        data_dir=data_dir,
        results_dir=str(results_path),
        memory_efficient=True,  # Prevent memory explosion
        use_shared_data=True    # Use shared data manager for memory efficiency
    )
    logger.info("✓ SimpleTimeSplitRunner created successfully")
    
    # Track overall results
    all_results = {
        "benchmark_info": {
            "name": "templ_timesplit_benchmark",
            "timestamp": datetime.now().isoformat(),
            "splits_evaluated": splits_to_run,
            "parameters": {
                "n_workers": n_workers,
                "n_conformers": n_conformers,
                "template_knn": template_knn,
                "max_pdbs": max_pdbs,
                "similarity_threshold": similarity_threshold,
                "timeout": timeout,
                "unconstrained": unconstrained,
                "align_metric": align_metric,
                "enable_optimization": enable_optimization,
                "no_realign": no_realign,
            }
        },
        "split_results": {},
        "overall_summary": {}
    }
    
    try:
        # Run benchmarks for each split
        for split_name in splits_to_run:
            logger.info(f"\n{'='*60}")
            logger.info(f"EVALUATING {split_name.upper()} SPLIT")
            logger.info(f"{'='*60}")
            
            try:
                split_results = runner.run_split_benchmark(
                    split_name=split_name,
                    n_workers=n_workers,
                    n_conformers=n_conformers,
                    max_pdbs=max_pdbs,
                    timeout=timeout,
                )
                
                all_results["split_results"][split_name] = split_results
                logger.info(f"✓ {split_name} split completed successfully")
                
            except Exception as e:
                logger.error(f"✗ {split_name} split failed: {e}")
                all_results["split_results"][split_name] = {
                    "success": False,
                    "error": str(e),
                    "split": split_name,
                }
        
        # Generate overall summary
        total_processed = sum(
            result.get("processed", 0) 
            for result in all_results["split_results"].values()
        )
        total_successful = sum(
            result.get("successful", 0) 
            for result in all_results["split_results"].values()
        )
        
        all_results["overall_summary"] = {
            "total_splits_run": len(splits_to_run),
            "total_molecules_processed": total_processed,
            "total_successful": total_successful,
            "overall_success_rate": (total_successful / total_processed * 100) if total_processed > 0 else 0,
            "splits_completed": [
                split for split, result in all_results["split_results"].items()
                if result.get("processed", 0) > 0
            ]
        }
        
        # Save complete results
        results_file = results_path / f"complete_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        all_results["results_file"] = str(results_file)
        all_results["success"] = True
        
        logger.info(f"\n{'='*60}")
        logger.info("BENCHMARK COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Total processed: {total_processed}")
        logger.info(f"Total successful: {total_successful}")
        logger.info(f"Overall success rate: {all_results['overall_summary']['overall_success_rate']:.1f}%")
        logger.info(f"Results saved to: {results_file}")
        
        return all_results
        
    finally:
        # Ensure proper cleanup of shared memory resources
        try:
            logger.info("Cleaning up shared memory resources...")
            runner.cleanup()
            logger.info("✓ Shared memory cleanup completed")
        except Exception as e:
            logger.warning(f"Shared memory cleanup failed: {e}")
            
        # Additional cleanup for any remaining shared memory objects
        try:
            import multiprocessing.resource_tracker as rt
            
            # Force cleanup of any remaining shared memory objects
            if hasattr(rt, '_CLEANUP_CALLBACKS'):
                for callback in rt._CLEANUP_CALLBACKS:
                    try:
                        callback()
                    except Exception:
                        pass
                        
        except Exception as e:
            logger.debug(f"Additional cleanup failed: {e}")


def build_parser() -> argparse.ArgumentParser:
    """Build argument parser for time-split benchmark CLI."""
    parser = argparse.ArgumentParser(
        description="Time-split benchmark for TEMPL pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Get hardware defaults
    hardware_config = get_suggested_worker_config()
    
    # Basic benchmark options
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["train", "val", "test"],
        default=["test"],
        help="Which splits to evaluate",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=hardware_config["n_workers"],
        help=f"Number of parallel workers (auto-detected: {hardware_config['n_workers']})",
    )
    parser.add_argument(
        "--n-conformers",
        type=int,
        default=200,
        help="Number of conformers per molecule",
    )
    parser.add_argument(
        "--template-knn",
        type=int,
        default=100,
        help="Number of template neighbors",
    )
    parser.add_argument(
        "--max-pdbs",
        type=int,
        default=None,
        help="Maximum PDBs per split (for testing)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Similarity threshold (overrides KNN)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Timeout per molecule in seconds",
    )
    
    # Input/output options
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory path",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=None,
        help="Results output directory",
    )
    parser.add_argument(
        "--poses-dir",
        type=str,
        default=None,
        help="Poses output directory",
    )
    
    # Split-specific options
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Evaluate only training set",
    )
    parser.add_argument(
        "--val-only",
        action="store_true", 
        help="Evaluate only validation set",
    )
    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Evaluate only test set",
    )
    
    # Quick mode
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: 10 conformers, 10 templates, first 10 molecules",
    )
    
    # Ablation study options
    parser.add_argument(
        "--unconstrained",
        action="store_true",
        help="Skip MCS and constrained embedding",
    )
    parser.add_argument(
        "--align-metric",
        choices=["shape", "color", "combo"],
        default="combo",
        help="Alignment metric for pose scoring",
    )
    parser.add_argument(
        "--enable-optimization",
        action="store_true",
        help="Enable force field optimization",
    )
    parser.add_argument(
        "--no-realign",
        action="store_true",
        help="Disable pose realignment",
    )
    
    # Advanced options
    parser.add_argument(
        "--per-worker-ram-gb",
        type=float,
        default=4.0,
        help="RAM limit per worker process (GB)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Set logging level",
    )
    
    return parser


def main(argv: List[str] = None) -> int:
    """
    Main CLI entry point for time-split benchmark.
    
    Args:
        argv: Command line arguments (for testing)
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    parser = build_parser()
    args = parser.parse_args(argv or sys.argv[1:])
    
    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    
    # Determine splits to run
    if args.train_only:
        splits_to_run = ["train"]
    elif args.val_only:
        splits_to_run = ["val"]
    elif args.test_only:
        splits_to_run = ["test"]
    else:
        splits_to_run = args.splits
    
    # Apply quick mode overrides
    if args.quick:
        args.n_conformers = 10
        args.template_knn = 10
        args.max_pdbs = 10
        args.n_workers = min(4, args.n_workers)
        logger.info("Quick mode: 10 conformers, 10 templates, 10 molecules max")
    
    try:
        result = run_timesplit_benchmark(
            splits_to_run=splits_to_run,
            n_workers=args.n_workers,
            n_conformers=args.n_conformers,
            template_knn=args.template_knn,
            max_pdbs=args.max_pdbs,
            data_dir=args.data_dir,
            results_dir=args.results_dir,
            poses_output_dir=args.poses_dir,
            similarity_threshold=args.similarity_threshold,
            timeout=args.timeout,
            quiet=args.quiet,
            unconstrained=args.unconstrained,
            align_metric=args.align_metric,
            enable_optimization=args.enable_optimization,
            no_realign=args.no_realign,
            per_worker_ram_gb=args.per_worker_ram_gb,
        )
        
        # Generate summary files after completion
        try:
            from templ_pipeline.benchmark.summary_generator import BenchmarkSummaryGenerator
            
            if result.get("results_file") and Path(result["results_file"]).exists():
                generator = BenchmarkSummaryGenerator()
                
                # Load results for summary generation
                with open(result["results_file"], 'r') as f:
                    results_data = json.load(f)
                
                # Extract individual results for summary processing
                individual_results = []
                for split_name, split_data in results_data.get("split_results", {}).items():
                    results_file = split_data.get("results_file")
                    if results_file and Path(results_file).exists():
                        # Load JSONL results
                        with open(results_file, 'r') as f:
                            for line in f:
                                if line.strip():
                                    individual_results.append(json.loads(line))
                
                if individual_results:
                    summary = generator.generate_unified_summary(individual_results, "timesplit")
                    
                    # Save to summaries directory
                    results_dir = Path(result["results_file"]).parent
                    summaries_dir = results_dir / "summaries"
                    summaries_dir.mkdir(exist_ok=True)
                    
                    saved_files = generator.save_summary_files(summary, summaries_dir)
                    logger.info("✓ Summary files generated:")
                    for fmt, path in saved_files.items():
                        logger.info(f"  {fmt.upper()}: {path}")
        
        except Exception as e:
            logger.warning(f"Failed to generate summary files: {e}")
        
        return 0 if result.get("success") else 1
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())