#!/bin/bash
set -e

# TEMPL Pipeline Build Performance Benchmark Script
# Compares build performance with and without optimizations
# Measures multi-core utilization and build times

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
HARBOR_USERNAME=${1:-$USER}
VERSION="benchmark-$(date +%Y%m%d-%H%M%S)"
REGISTRY="cerit.io"
BENCHMARK_RESULTS_DIR="./benchmark_results"

print_header() {
    echo -e "\n${PURPLE}=========================================="
    echo -e "$1"
    echo -e "==========================================${NC}\n"
}

print_step() {
    echo -e "${BLUE}[BENCHMARK]${NC} $1"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_info() {
    echo -e "${CYAN}ℹ️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# System information
get_system_info() {
    print_step "Gathering system information..."
    
    echo "System Information:" > "$BENCHMARK_RESULTS_DIR/system_info.txt"
    echo "==================" >> "$BENCHMARK_RESULTS_DIR/system_info.txt"
    echo "CPU: $(nproc) cores" >> "$BENCHMARK_RESULTS_DIR/system_info.txt"
    echo "Memory: $(free -h | awk '/^Mem:/ {print $2}')" >> "$BENCHMARK_RESULTS_DIR/system_info.txt"
    echo "Docker Version: $(docker --version)" >> "$BENCHMARK_RESULTS_DIR/system_info.txt"
    echo "BuildKit Status: $(docker buildx version 2>/dev/null || echo 'Not available')" >> "$BENCHMARK_RESULTS_DIR/system_info.txt"
    echo "Date: $(date)" >> "$BENCHMARK_RESULTS_DIR/system_info.txt"
    echo "" >> "$BENCHMARK_RESULTS_DIR/system_info.txt"
    
    print_success "System info collected"
}

# Monitor CPU and memory usage during build
monitor_resources() {
    local test_name="$1"
    local pid="$2"
    local output_file="$BENCHMARK_RESULTS_DIR/resources_${test_name}.log"
    
    echo "Timestamp,CPU_Usage,Memory_Usage,Load_Average" > "$output_file"
    
    while kill -0 "$pid" 2>/dev/null; do
        cpu_usage=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//')
        mem_usage=$(free | awk '/^Mem:/ {printf "%.1f", ($3/$2)*100}')
        load_avg=$(uptime | awk -F'load average:' '{print $2}' | awk '{print $1}' | sed 's/,//')
        timestamp=$(date '+%Y-%m-%d %H:%M:%S')
        
        echo "$timestamp,$cpu_usage,$mem_usage,$load_avg" >> "$output_file"
        sleep 2
    done
}

# Benchmark standard Docker build
benchmark_standard_build() {
    print_step "Benchmarking standard Docker build..."
    
    local start_time=$(date +%s)
    
    # Start resource monitoring in background
    docker build -f deploy/docker/Dockerfile -t "test-standard:$VERSION" . >/dev/null 2>&1 &
    local build_pid=$!
    
    monitor_resources "standard" "$build_pid" &
    local monitor_pid=$!
    
    # Wait for build to complete
    wait "$build_pid"
    local build_result=$?
    
    # Stop monitoring
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "standard_build_duration_seconds: $duration" >> "$BENCHMARK_RESULTS_DIR/timing_results.txt"
    echo "standard_build_success: $([[ $build_result -eq 0 ]] && echo "true" || echo "false")" >> "$BENCHMARK_RESULTS_DIR/timing_results.txt"
    
    if [[ $build_result -eq 0 ]]; then
        print_success "Standard build completed in ${duration}s"
    else
        print_error "Standard build failed"
    fi
    
    # Cleanup
    docker rmi "test-standard:$VERSION" 2>/dev/null || true
}

# Benchmark optimized BuildKit build
benchmark_buildkit_build() {
    print_step "Benchmarking optimized BuildKit build..."
    
    # Enable BuildKit
    export DOCKER_BUILDKIT=1
    
    # Create builder if needed
    if ! docker buildx inspect templ-benchmark >/dev/null 2>&1; then
        docker buildx create --name templ-benchmark --use --driver-opt=network=host
    else
        docker buildx use templ-benchmark
    fi
    
    local start_time=$(date +%s)
    
    # Start BuildKit build with cache
    docker buildx build \
        --platform linux/amd64 \
        -f deploy/docker/Dockerfile \
        -t "test-buildkit:$VERSION" \
        --cache-from type=registry,ref="${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:buildcache" \
        --cache-to type=registry,ref="${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:buildcache",mode=max \
        --load \
        . >/dev/null 2>&1 &
    local build_pid=$!
    
    monitor_resources "buildkit" "$build_pid" &
    local monitor_pid=$!
    
    # Wait for build to complete
    wait "$build_pid"
    local build_result=$?
    
    # Stop monitoring
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "buildkit_build_duration_seconds: $duration" >> "$BENCHMARK_RESULTS_DIR/timing_results.txt"
    echo "buildkit_build_success: $([[ $build_result -eq 0 ]] && echo "true" || echo "false")" >> "$BENCHMARK_RESULTS_DIR/timing_results.txt"
    
    if [[ $build_result -eq 0 ]]; then
        print_success "BuildKit build completed in ${duration}s"
    else
        print_error "BuildKit build failed"
    fi
    
    # Cleanup
    docker rmi "test-buildkit:$VERSION" 2>/dev/null || true
}

# Benchmark multi-platform build
benchmark_multiplatform_build() {
    print_step "Benchmarking multi-platform BuildKit build..."
    
    local start_time=$(date +%s)
    
    # Multi-platform build (without push for benchmark)
    docker buildx build \
        --platform linux/amd64,linux/arm64 \
        -f deploy/docker/Dockerfile \
        -t "test-multiplatform:$VERSION" \
        --cache-from type=registry,ref="${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:buildcache" \
        --cache-to type=registry,ref="${REGISTRY}/${HARBOR_USERNAME}/templ-pipeline:buildcache",mode=max \
        . >/dev/null 2>&1 &
    local build_pid=$!
    
    monitor_resources "multiplatform" "$build_pid" &
    local monitor_pid=$!
    
    # Wait for build to complete
    wait "$build_pid"
    local build_result=$?
    
    # Stop monitoring
    kill "$monitor_pid" 2>/dev/null || true
    wait "$monitor_pid" 2>/dev/null || true
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    echo "multiplatform_build_duration_seconds: $duration" >> "$BENCHMARK_RESULTS_DIR/timing_results.txt"
    echo "multiplatform_build_success: $([[ $build_result -eq 0 ]] && echo "true" || echo "false")" >> "$BENCHMARK_RESULTS_DIR/timing_results.txt"
    
    if [[ $build_result -eq 0 ]]; then
        print_success "Multi-platform build completed in ${duration}s"
    else
        print_error "Multi-platform build failed"
    fi
}

# Generate performance report
generate_report() {
    print_step "Generating performance report..."
    
    local report_file="$BENCHMARK_RESULTS_DIR/performance_report.md"
    
    cat > "$report_file" << 'EOF'
# TEMPL Pipeline Build Performance Report

## System Information
EOF
    
    cat "$BENCHMARK_RESULTS_DIR/system_info.txt" >> "$report_file"
    
    cat >> "$report_file" << 'EOF'

## Build Performance Results

| Build Type | Duration (seconds) | Success | Performance Gain |
|------------|-------------------|---------|------------------|
EOF
    
    # Parse timing results
    if [[ -f "$BENCHMARK_RESULTS_DIR/timing_results.txt" ]]; then
        local standard_time=$(grep "standard_build_duration_seconds:" "$BENCHMARK_RESULTS_DIR/timing_results.txt" | cut -d: -f2 | tr -d ' ')
        local buildkit_time=$(grep "buildkit_build_duration_seconds:" "$BENCHMARK_RESULTS_DIR/timing_results.txt" | cut -d: -f2 | tr -d ' ')
        local multiplatform_time=$(grep "multiplatform_build_duration_seconds:" "$BENCHMARK_RESULTS_DIR/timing_results.txt" | cut -d: -f2 | tr -d ' ')
        
        local standard_success=$(grep "standard_build_success:" "$BENCHMARK_RESULTS_DIR/timing_results.txt" | cut -d: -f2 | tr -d ' ')
        local buildkit_success=$(grep "buildkit_build_success:" "$BENCHMARK_RESULTS_DIR/timing_results.txt" | cut -d: -f2 | tr -d ' ')
        local multiplatform_success=$(grep "multiplatform_build_success:" "$BENCHMARK_RESULTS_DIR/timing_results.txt" | cut -d: -f2 | tr -d ' ')
        
        # Calculate performance gains
        local buildkit_gain=""
        local multiplatform_gain=""
        
        if [[ -n "$standard_time" && -n "$buildkit_time" && "$standard_time" -gt 0 ]]; then
            buildkit_gain=$(echo "scale=1; (($standard_time - $buildkit_time) * 100) / $standard_time" | bc -l 2>/dev/null || echo "N/A")
            buildkit_gain="${buildkit_gain}%"
        fi
        
        if [[ -n "$standard_time" && -n "$multiplatform_time" ]]; then
            # For multi-platform, compare against 2x standard time (sequential build for 2 platforms)
            local sequential_multiplatform=$((standard_time * 2))
            if [[ "$multiplatform_time" -lt "$sequential_multiplatform" ]]; then
                multiplatform_gain=$(echo "scale=1; (($sequential_multiplatform - $multiplatform_time) * 100) / $sequential_multiplatform" | bc -l 2>/dev/null || echo "N/A")
                multiplatform_gain="${multiplatform_gain}%"
            else
                multiplatform_gain="0%"
            fi
        fi
        
        echo "| Standard Docker | ${standard_time:-N/A} | ${standard_success:-N/A} | Baseline |" >> "$report_file"
        echo "| BuildKit Optimized | ${buildkit_time:-N/A} | ${buildkit_success:-N/A} | ${buildkit_gain:-N/A} |" >> "$report_file"
        echo "| Multi-Platform | ${multiplatform_time:-N/A} | ${multiplatform_success:-N/A} | ${multiplatform_gain:-N/A} |" >> "$report_file"
    fi
    
    cat >> "$report_file" << 'EOF'

## Resource Usage Analysis

### CPU Usage
See individual resource logs for detailed CPU utilization during builds.

### Memory Usage
Memory usage patterns are recorded in the resource log files.

## Recommendations

Based on the benchmark results:

1. **For single-platform builds**: Use BuildKit optimization for improved performance
2. **For multi-platform builds**: Use concurrent BuildKit builds for significant time savings
3. **For CI/CD**: Implement registry cache to maximize build efficiency
4. **For development**: Use single-platform builds with cache optimization

## Files Generated

- `system_info.txt`: System specifications
- `timing_results.txt`: Raw timing data
- `resources_*.log`: Resource usage during builds
- `performance_report.md`: This summary report

EOF
    
    print_success "Performance report generated: $report_file"
}

# Main benchmark function
main() {
    print_header "TEMPL Pipeline Build Performance Benchmark"
    
    # Validate parameters
    if [[ -z "$HARBOR_USERNAME" ]]; then
        print_error "Harbor username is required"
        echo "Usage: $0 <harbor-username>"
        exit 1
    fi
    
    # Create results directory
    mkdir -p "$BENCHMARK_RESULTS_DIR"
    echo "# TEMPL Pipeline Build Benchmark Results" > "$BENCHMARK_RESULTS_DIR/timing_results.txt"
    echo "# Generated: $(date)" >> "$BENCHMARK_RESULTS_DIR/timing_results.txt"
    echo "" >> "$BENCHMARK_RESULTS_DIR/timing_results.txt"
    
    # Run benchmarks
    get_system_info
    
    print_info "Starting build performance benchmarks..."
    print_info "This may take 15-30 minutes depending on system performance"
    
    # Clear Docker cache for fair comparison
    print_step "Clearing Docker cache for fair comparison..."
    docker system prune -f >/dev/null 2>&1 || true
    
    # Run benchmark tests
    benchmark_standard_build
    
    # Small delay between tests
    sleep 5
    
    benchmark_buildkit_build
    
    # Small delay between tests
    sleep 5
    
    benchmark_multiplatform_build
    
    # Generate final report
    generate_report
    
    print_header "Benchmark Complete!"
    print_success "Results saved to: $BENCHMARK_RESULTS_DIR/"
    print_info "View the performance report: $BENCHMARK_RESULTS_DIR/performance_report.md"
    
    # Show quick summary
    if [[ -f "$BENCHMARK_RESULTS_DIR/timing_results.txt" ]]; then
        echo ""
        print_info "Quick Summary:"
        grep "_duration_seconds:" "$BENCHMARK_RESULTS_DIR/timing_results.txt" | while read line; do
            test_name=$(echo "$line" | cut -d_ -f1)
            duration=$(echo "$line" | cut -d: -f2 | tr -d ' ')
            echo "  $test_name: ${duration}s"
        done
    fi
}

# Execute main function
main "$@"