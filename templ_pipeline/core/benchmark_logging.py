"""
Centralized logging configuration for TEMPL benchmarks.

This module provides clean logging configuration that routes all log messages
to workspace-specific files while keeping the terminal output clean with only
progress bars visible to the user.

Features:
- File-only logging configuration for benchmarks
- Progress bar isolation from log messages
- Workspace-aware log file organization
- Context manager for clean setup/teardown
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager


class BenchmarkLoggingConfig:
    """Configuration for benchmark logging setup."""
    
    def __init__(
        self,
        workspace_dir: Path,
        benchmark_name: str,
        log_level: str = "INFO",
        suppress_console: bool = True,
        preserve_progress_bars: bool = True
    ):
        """
        Initialize benchmark logging configuration.
        
        Args:
            workspace_dir: Workspace directory for log files
            benchmark_name: Name of the benchmark (e.g., 'polaris', 'timesplit')
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            suppress_console: Whether to suppress console output completely
            preserve_progress_bars: Whether to preserve progress bar output
        """
        self.workspace_dir = Path(workspace_dir)
        self.benchmark_name = benchmark_name
        self.log_level = getattr(logging, log_level.upper())
        self.suppress_console = suppress_console
        self.preserve_progress_bars = preserve_progress_bars
        
        # Create logs directory
        self.logs_dir = self.workspace_dir / "logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Define log file paths
        self.main_log_file = self.logs_dir / f"{benchmark_name}_benchmark.log"
        self.error_log_file = self.logs_dir / f"{benchmark_name}_benchmark_errors.log"
        
        # Store original handlers for restoration
        self.original_handlers = {}
        self.original_levels = {}
        
    def setup_file_logging(self) -> Dict[str, str]:
        """
        Set up file-only logging configuration.
        
        Returns:
            Dictionary with log file paths
        """
        # Get root logger
        root_logger = logging.getLogger()
        
        # Store original configuration
        self.original_handlers = root_logger.handlers[:]
        self.original_levels = {
            'root': root_logger.level,
            'templ_pipeline': logging.getLogger('templ_pipeline').level,
            'templ-cli': logging.getLogger('templ-cli').level
        }
        
        # Clear existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Set up file formatter
        file_formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Main log file handler (INFO+)
        main_file_handler = logging.FileHandler(self.main_log_file, mode='w')
        main_file_handler.setLevel(logging.INFO)
        main_file_handler.setFormatter(file_formatter)
        root_logger.addHandler(main_file_handler)
        
        # Error log file handler (WARNING+)
        error_file_handler = logging.FileHandler(self.error_log_file, mode='w')
        error_file_handler.setLevel(logging.WARNING)
        error_file_handler.setFormatter(file_formatter)
        root_logger.addHandler(error_file_handler)
        
        # Set root logger level
        root_logger.setLevel(self.log_level)
        
        # Configure specific loggers
        benchmark_logger = logging.getLogger(f'templ_pipeline.benchmark')
        benchmark_logger.setLevel(self.log_level)
        
        cli_logger = logging.getLogger('templ-cli')
        cli_logger.setLevel(self.log_level)
        
        # Suppress console output if requested
        if self.suppress_console:
            # Add null handler to specific loggers that might print to console
            for logger_name in [
                'templ_pipeline.core.mcs',
                'templ_pipeline.core.scoring',
                'templ_pipeline.core.pipeline',
                'templ_pipeline.benchmark',
                'templ_pipeline.core.embedding',
                'templ_pipeline.core.templates',
                'templ_pipeline.core.utils',
                'templ_pipeline.core.hardware',
                'templ_pipeline.core.workspace_manager'
            ]:
                logger = logging.getLogger(logger_name)
                # Remove all existing handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                # Add null handler only
                logger.addHandler(logging.NullHandler())
                # Disable propagation to prevent parent logger handling
                logger.propagate = False
        
        # Log successful setup
        setup_msg = f"Benchmark logging configured for {self.benchmark_name}"
        root_logger.info(setup_msg)
        root_logger.info(f"Main log: {self.main_log_file}")
        root_logger.info(f"Error log: {self.error_log_file}")
        
        return {
            'main_log': str(self.main_log_file),
            'error_log': str(self.error_log_file),
            'logs_dir': str(self.logs_dir)
        }
    
    def restore_original_logging(self):
        """Restore original logging configuration."""
        root_logger = logging.getLogger()
        
        # Clear current handlers
        for handler in root_logger.handlers[:]:
            handler.close()
            root_logger.removeHandler(handler)
        
        # Restore original handlers
        for handler in self.original_handlers:
            root_logger.addHandler(handler)
        
        # Restore original levels
        for logger_name, level in self.original_levels.items():
            if logger_name == 'root':
                root_logger.setLevel(level)
            else:
                logging.getLogger(logger_name).setLevel(level)


@contextmanager
def benchmark_logging_context(
    workspace_dir: Path,
    benchmark_name: str,
    log_level: str = "INFO",
    suppress_console: bool = True
):
    """
    Context manager for benchmark logging configuration.
    
    Args:
        workspace_dir: Workspace directory for log files
        benchmark_name: Name of the benchmark
        log_level: Logging level
        suppress_console: Whether to suppress console output
        
    Yields:
        Dictionary with log file paths
    """
    config = BenchmarkLoggingConfig(
        workspace_dir=workspace_dir,
        benchmark_name=benchmark_name,
        log_level=log_level,
        suppress_console=suppress_console
    )
    
    try:
        log_info = config.setup_file_logging()
        yield log_info
    finally:
        config.restore_original_logging()


def configure_tqdm_for_benchmark():
    """
    Configure tqdm for clean progress bar display in benchmarks.
    
    Returns:
        Dictionary with tqdm configuration
    """
    return {
        'bar_format': '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        'ncols': 100,
        'leave': True,
        'disable': False,
        'file': sys.stdout  # Ensure progress bars go to stdout
    }


def suppress_worker_logging():
    """
    Suppress logging from worker processes to prevent console pollution.
    
    This function should be called at the start of worker processes
    to prevent log messages from appearing in the main process output.
    """
    # Get root logger
    root_logger = logging.getLogger()
    
    # Remove all handlers to prevent output
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add null handler to prevent log messages
    null_handler = logging.NullHandler()
    root_logger.addHandler(null_handler)
    
    # Set high level to minimize processing
    root_logger.setLevel(logging.CRITICAL)
    
    # Also suppress specific loggers that might be noisy
    for logger_name in [
        'templ_pipeline.core.mcs',
        'templ_pipeline.core.scoring',
        'templ_pipeline.core.pipeline',
        'templ_pipeline.core.embedding',
        'templ_pipeline.core.templates',
        'templ_pipeline.benchmark',
        'templ_pipeline.core.utils',
        'templ_pipeline.core.hardware',
        'templ_pipeline.core.workspace_manager',
        'templ-cli'
    ]:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 1)  # Set to above CRITICAL
        # Remove all handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        # Add null handler
        logger.addHandler(logging.NullHandler())
        # Disable propagation to prevent parent logger handling
        logger.propagate = False
    
    # Also suppress any remaining print statements by redirecting stdout/stderr
    import sys
    import os
    
    # Redirect stdout and stderr to devnull for worker processes
    try:
        devnull = open(os.devnull, 'w')
        # Don't redirect stdout/stderr completely as this might break tqdm
        # sys.stdout = devnull  
        # sys.stderr = devnull
    except:
        pass


def create_benchmark_logger(benchmark_name: str) -> logging.Logger:
    """
    Create a benchmark-specific logger.
    
    Args:
        benchmark_name: Name of the benchmark
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(f'templ_pipeline.benchmark.{benchmark_name}')
    return logger


# Progress bar configuration matching the user's requested format
BENCHMARK_PROGRESS_CONFIG = {
    'polaris': {
        'bar_format': '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        'ncols': 100,
        'leave': True,
        'file': sys.stdout,
        'disable': False
    },
    'timesplit': {
        'bar_format': '{percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]',
        'ncols': 100,
        'leave': True,
        'file': sys.stdout,
        'disable': False
    }
}


def get_progress_bar_config(benchmark_name: str) -> Dict[str, Any]:
    """
    Get progress bar configuration for a specific benchmark.
    
    Args:
        benchmark_name: Name of the benchmark
        
    Returns:
        Progress bar configuration dictionary
    """
    return BENCHMARK_PROGRESS_CONFIG.get(
        benchmark_name, 
        BENCHMARK_PROGRESS_CONFIG['polaris']
    )