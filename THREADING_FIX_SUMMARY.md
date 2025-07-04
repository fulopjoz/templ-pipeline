# Threading Fix Summary

## Problem
The `_get_executor_for_context` function in `templ_pipeline/core/scoring.py` was creating ThreadPoolExecutor instances without proper resource management, leading to potential "can't start new thread" errors under high load scenarios.

## Solution
Modified the threading system to integrate with the existing ThreadResourceManager for proper thread resource management and monitoring.

## Files Modified

### 1. `/templ_pipeline/core/scoring.py`

**Function: `_get_executor_for_context`** (lines 973-1012)
- **Added**: Integration with ThreadResourceManager to get safe worker counts
- **Added**: Thread status logging for debugging
- **Added**: Robust fallback handling for when ThreadResourceManager is not available
- **Changed**: Uses `get_safe_worker_count(n_workers, task_type="scoring")` instead of raw n_workers
- **Changed**: Added comprehensive error handling with multiple fallback levels

**Function: `select_best`** (lines 1130-1159)
- **Added**: Optional thread monitoring around batch processing
- **Added**: Fallback handling for when thread monitoring is not available
- **Changed**: Enhanced error handling for parallel processing failures

### 2. `/templ_pipeline/core/thread_manager.py`

**Imports section** (lines 8-18)
- **Added**: Optional psutil import to prevent hanging if psutil is not available
- **Added**: HAS_PSUTIL flag to handle cases where psutil is missing

**Function: `_get_system_thread_limit`** (lines 54-59)
- **Changed**: Made psutil usage conditional based on HAS_PSUTIL flag
- **Changed**: Improved error handling for psutil operations

## Key Improvements

1. **Thread Resource Management**: The scoring system now respects system thread limits and adapts worker counts based on available resources.

2. **Graceful Degradation**: If ThreadResourceManager is not available, the system falls back to conservative worker counts (max 4 workers).

3. **Enhanced Monitoring**: Added thread status logging to help debug threading issues.

4. **Robust Error Handling**: Multiple layers of fallback ensure the system continues to work even if advanced threading features fail.

5. **Memory Efficiency**: Maintains existing batch processing and memory cleanup while adding thread safety.

## Behavior Changes

- **Worker Count Adaptation**: The system now automatically reduces worker counts based on available system resources
- **Conservative Fallbacks**: If threading resources are limited, the system uses fewer workers rather than failing
- **Better Logging**: Thread status is logged for debugging threading issues
- **Safer Defaults**: Default worker counts are capped at safer levels (2-4 workers) when resource management fails

## Testing

The fix includes comprehensive fallback mechanisms, so it should work even in environments where:
- psutil is not available
- ThreadResourceManager cannot be imported
- System resources are limited
- High threading pressure exists

## Usage

The changes are transparent to existing code - all existing calls to `_get_executor_for_context` and `select_best` will automatically benefit from the improved thread resource management.