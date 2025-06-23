from importlib import import_module

__all__ = ["run"]

def run(argv=None):
    """Entry-point to run the Polaris benchmark via templ CLI wrappers."""
    runner = import_module("templ_pipeline.benchmark.polaris.polaris_runner")
    return runner.main(argv) 