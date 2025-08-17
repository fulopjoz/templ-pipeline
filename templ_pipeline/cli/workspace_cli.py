# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
TEMPL Pipeline - CLI Workspace Management

Command-line utilities for managing workspaces, cleaning up files,
and monitoring storage usage across CLI runs.
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from templ_pipeline.core.unified_workspace_manager import (
        UnifiedWorkspaceManager,
        WorkspaceConfig,
        cleanup_old_workspaces,
    )

    WORKSPACE_AVAILABLE = True
except ImportError:
    WORKSPACE_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def list_workspaces(workspace_root: str = "workspace") -> List[Dict[str, Any]]:
    """
    List all workspace directories.

    Args:
        workspace_root: Root workspace directory

    Returns:
        List of workspace information
    """
    workspace_path = Path(workspace_root)
    if not workspace_path.exists():
        return []

    workspaces = []
    for run_dir in workspace_path.glob("run_*"):
        if not run_dir.is_dir():
            continue

        try:
            # Calculate directory size
            total_size = sum(
                f.stat().st_size for f in run_dir.rglob("*") if f.is_file()
            )

            # Count files by type
            temp_files = (
                len(list((run_dir / "temp").rglob("*")))
                if (run_dir / "temp").exists()
                else 0
            )
            output_files = (
                len(list((run_dir / "output").rglob("*")))
                if (run_dir / "output").exists()
                else 0
            )

            # Get creation time
            creation_time = run_dir.stat().st_ctime

            workspaces.append(
                {
                    "run_id": run_dir.name.replace("run_", ""),
                    "path": str(run_dir),
                    "size_mb": total_size / (1024 * 1024),
                    "temp_files": temp_files,
                    "output_files": output_files,
                    "created": datetime.fromtimestamp(creation_time).isoformat(),
                    "age_hours": (datetime.now().timestamp() - creation_time) / 3600,
                }
            )

        except Exception as e:
            logger.warning(f"Could not analyze workspace {run_dir}: {e}")
            continue

    return sorted(workspaces, key=lambda x: x["created"], reverse=True)


def display_workspace_summary(workspace_root: str = "workspace"):
    """Display summary of all workspaces"""
    workspaces = list_workspaces(workspace_root)

    if not workspaces:
        print(f"No workspaces found in {workspace_root}")
        return

    print(f"\n📁 Workspace Summary ({workspace_root})")
    print("=" * 60)

    total_size = sum(w["size_mb"] for w in workspaces)
    total_temp_files = sum(w["temp_files"] for w in workspaces)
    total_output_files = sum(w["output_files"] for w in workspaces)

    print(f"Total workspaces: {len(workspaces)}")
    print(f"Total size: {total_size:.1f} MB")
    print(f"Total temp files: {total_temp_files}")
    print(f"Total output files: {total_output_files}")
    print()

    # Display individual workspaces
    print("Individual Workspaces:")
    print("-" * 60)
    for workspace in workspaces:
        print(f"🗂️  {workspace['run_id']}")
        print(f"    Size: {workspace['size_mb']:.1f} MB")
        print(
            f"    Files: {workspace['temp_files']} temp, {workspace['output_files']} output"
        )
        print(f"    Age: {workspace['age_hours']:.1f} hours")
        print(f"    Created: {workspace['created']}")
        print()


def cleanup_workspaces(
    workspace_root: str = "workspace",
    max_age_days: int = 7,
    keep_successful: bool = True,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """
    Cleanup old workspaces.

    Args:
        workspace_root: Root workspace directory
        max_age_days: Maximum age in days
        keep_successful: Whether to keep workspaces with output files
        dry_run: If True, show what would be cleaned without actually doing it

    Returns:
        Cleanup statistics
    """
    if not WORKSPACE_AVAILABLE:
        return {"error": "Unified workspace manager not available"}

    print(f"\n🧹 Workspace Cleanup ({'DRY RUN' if dry_run else 'LIVE'})")
    print("=" * 60)
    print(f"Target: {workspace_root}")
    print(f"Max age: {max_age_days} days")
    print(f"Keep successful: {keep_successful}")
    print()

    if dry_run:
        # Simulate cleanup by listing what would be removed
        import time

        workspace_path = Path(workspace_root)
        if not workspace_path.exists():
            return {"error": "Workspace root does not exist"}

        cutoff_time = time.time() - (max_age_days * 24 * 3600)
        candidates = []

        for run_dir in workspace_path.glob("run_*"):
            if not run_dir.is_dir():
                continue

            try:
                dir_mtime = run_dir.stat().st_mtime
                if dir_mtime < cutoff_time:
                    # Check if we should keep successful runs
                    if keep_successful and (run_dir / "output").exists():
                        output_files = list((run_dir / "output").glob("*"))
                        if output_files:
                            continue  # Skip this directory

                    # Calculate size
                    dir_size = sum(
                        f.stat().st_size for f in run_dir.rglob("*") if f.is_file()
                    )

                    candidates.append(
                        {
                            "path": str(run_dir),
                            "size_mb": dir_size / (1024 * 1024),
                            "age_days": (time.time() - dir_mtime) / (24 * 3600),
                        }
                    )

            except Exception as e:
                logger.warning(f"Could not analyze {run_dir}: {e}")

        print(f"Would remove {len(candidates)} workspaces:")
        total_size = 0
        for candidate in candidates:
            print(
                f"  - {Path(candidate['path']).name}: {candidate['size_mb']:.1f} MB (age: {candidate['age_days']:.1f} days)"
            )
            total_size += candidate["size_mb"]

        print(f"\nTotal space to free: {total_size:.1f} MB")

        return {
            "dry_run": True,
            "workspaces_to_clean": len(candidates),
            "mb_to_free": total_size,
        }

    else:
        # Actual cleanup
        stats = cleanup_old_workspaces(
            workspace_root=workspace_root,
            max_age_days=max_age_days,
            keep_successful=keep_successful,
        )

        if "error" in stats:
            print(f"❌ Cleanup failed: {stats['error']}")
        else:
            cleaned = stats.get("workspaces_cleaned", 0)
            freed_mb = stats.get("mb_freed", 0)
            print(f"✅ Cleanup completed:")
            print(f"   Workspaces removed: {cleaned}")
            print(f"   Space freed: {freed_mb:.1f} MB")

        return stats


def create_test_workspace(run_id: str = None) -> str:
    """Create a test workspace for demonstration"""
    if not WORKSPACE_AVAILABLE:
        print("❌ Unified workspace manager not available")
        return ""

    config = WorkspaceConfig(base_dir="workspace", auto_cleanup=False)

    with UnifiedWorkspaceManager(run_id=run_id, config=config) as workspace:
        print(f"\n🔧 Created test workspace: {workspace.run_dir}")

        # Create some test files
        test_content = "This is a test file for workspace demonstration"

        # Save test files in different categories
        temp_file = workspace.get_temp_file("test", ".txt", "processing")
        Path(temp_file).write_text(test_content)

        output_file = workspace.save_output("test_output.txt", test_content)

        metadata = {
            "test": True,
            "created": datetime.now().isoformat(),
            "purpose": "CLI workspace demonstration",
        }
        metadata_file = workspace.save_metadata("test_output", metadata)

        print(f"   Temp file: {temp_file}")
        print(f"   Output file: {output_file}")
        print(f"   Metadata file: {metadata_file}")

        # Display workspace summary
        summary = workspace.get_workspace_summary()
        print(f"\n📊 Workspace Summary:")
        print(f"   Files: {summary['file_counts']}")
        print(f"   Size: {summary['total_size_mb']:.1f} MB")

        return workspace.run_id


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="TEMPL Pipeline Workspace Management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                           # List all workspaces
  %(prog)s cleanup --max-age 7            # Clean workspaces older than 7 days
  %(prog)s cleanup --dry-run              # Show what would be cleaned
  %(prog)s create-test                    # Create test workspace
  %(prog)s summary                        # Show workspace summary
        """,
    )

    parser.add_argument(
        "command",
        choices=["list", "cleanup", "create-test", "summary"],
        help="Command to execute",
    )

    parser.add_argument(
        "--workspace-root",
        default="workspace",
        help="Root workspace directory (default: workspace)",
    )

    parser.add_argument(
        "--max-age",
        type=int,
        default=7,
        help="Maximum age in days for cleanup (default: 7)",
    )

    parser.add_argument(
        "--keep-successful",
        action="store_true",
        default=True,
        help="Keep workspaces with output files during cleanup",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually doing it",
    )

    parser.add_argument("--run-id", help="Custom run ID for test workspace creation")

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)",
    )

    parser.add_argument(
        "--json", action="store_true", help="Output results in JSON format"
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging(args.log_level)

    # Check if workspace manager is available
    if not WORKSPACE_AVAILABLE:
        print("❌ Error: Unified workspace manager not available")
        print("Make sure templ_pipeline.core.unified_workspace_manager is installed")
        sys.exit(1)

    # Execute command
    try:
        if args.command == "list":
            workspaces = list_workspaces(args.workspace_root)
            if args.json:
                print(json.dumps(workspaces, indent=2))
            else:
                if workspaces:
                    print(
                        f"\n📁 Found {len(workspaces)} workspaces in {args.workspace_root}:"
                    )
                    for w in workspaces:
                        print(
                            f"  {w['run_id']} - {w['size_mb']:.1f}MB - {w['age_hours']:.1f}h old"
                        )
                else:
                    print(f"No workspaces found in {args.workspace_root}")

        elif args.command == "summary":
            display_workspace_summary(args.workspace_root)

        elif args.command == "cleanup":
            stats = cleanup_workspaces(
                workspace_root=args.workspace_root,
                max_age_days=args.max_age,
                keep_successful=args.keep_successful,
                dry_run=args.dry_run,
            )
            if args.json:
                print(json.dumps(stats, indent=2))

        elif args.command == "create-test":
            run_id = create_test_workspace(args.run_id)
            if args.json:
                print(json.dumps({"run_id": run_id}))

    except KeyboardInterrupt:
        print("\n⚠️  Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Command failed: {e}")
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
