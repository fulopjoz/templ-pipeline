"""
Automated QA reporting dashboard for comprehensive quality metrics.
Provides real-time visualization and analysis of testing and quality data.
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import subprocess
import sys
import os
from collections import defaultdict
import statistics


class QADashboard:
    """Comprehensive QA dashboard with real-time metrics and reporting."""
    
    def __init__(self, output_dir: str = "qa-dashboard"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metrics = {}
        self.historical_data = []
        
    def collect_all_metrics(self) -> Dict:
        """Collect comprehensive QA metrics from all sources."""
        print("Collecting QA metrics...")
        
        metrics = {
            'timestamp': time.time(),
            'date': datetime.now().isoformat(),
            'test_metrics': self._collect_test_metrics(),
            'coverage_metrics': self._collect_coverage_metrics(),
            'performance_metrics': self._collect_performance_metrics(),
            'flaky_test_metrics': self._collect_flaky_test_metrics(),
            'fixture_cache_metrics': self._collect_fixture_cache_metrics(),
            'quality_score': 0  # Will be calculated
        }
        
        # Calculate overall quality score
        metrics['quality_score'] = self._calculate_quality_score(metrics)
        
        return metrics
    
    def _collect_test_metrics(self) -> Dict:
        """Collect basic test execution metrics."""
        try:
            # Run test discovery to get test counts
            result = subprocess.run([
                sys.executable, "-m", "pytest", "--collect-only", "-q"
            ], capture_output=True, text=True, timeout=30)
            
            test_count = 0
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    if "test session starts" in line.lower():
                        continue
                    if " collected" in line:
                        parts = line.split()
                        for i, part in enumerate(parts):
                            if part == "collected":
                                try:
                                    test_count = int(parts[i-1])
                                    break
                                except (ValueError, IndexError):
                                    pass
            
            return {
                'total_tests': test_count,
                'discovery_time': time.time(),
                'discovery_status': 'success' if result.returncode == 0 else 'failed'
            }
        except Exception as e:
            return {
                'total_tests': 0,
                'discovery_time': time.time(),
                'discovery_status': 'error',
                'error': str(e)
            }
    
    def _collect_coverage_metrics(self) -> Dict:
        """Collect coverage analysis metrics."""
        try:
            from .test_coverage_analysis import coverage_analyzer
            
            # Check if coverage data exists
            coverage_file = coverage_analyzer.output_dir / "coverage-analysis.json"
            if coverage_file.exists():
                with open(coverage_file, 'r') as f:
                    coverage_data = json.load(f)
                
                overall = coverage_data.get('overall_metrics', {})
                return {
                    'line_coverage': overall.get('line_coverage', 0),
                    'branch_coverage': overall.get('branch_coverage', 0),
                    'quality_score': coverage_data.get('coverage_quality_score', 0),
                    'files_analyzed': overall.get('files_analyzed', 0),
                    'critical_path_coverage': len(coverage_data.get('critical_path_coverage', {})),
                    'low_coverage_files': len(coverage_data.get('low_coverage_files', [])),
                    'last_analysis': coverage_data.get('timestamp', time.time())
                }
            else:
                return {
                    'line_coverage': 0,
                    'branch_coverage': 0,
                    'quality_score': 0,
                    'status': 'no_data'
                }
        except Exception as e:
            return {
                'line_coverage': 0,
                'branch_coverage': 0,
                'quality_score': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _collect_performance_metrics(self) -> Dict:
        """Collect performance and timing metrics."""
        try:
            from .test_timing_analysis import TestTimingAnalyzer
            from .test_profiler import TestProfiler
            
            # Timing analysis
            timing_analyzer = TestTimingAnalyzer()
            timing_data = timing_analyzer.load_historical_data()
            
            # Profiler data
            profiler = TestProfiler()
            profile_report = profiler.generate_performance_report()
            
            if timing_data and 'summary' in timing_data:
                summary = timing_data['summary']
                return {
                    'total_test_time': summary.get('total_duration', 0),
                    'average_test_time': summary.get('average_duration', 0),
                    'slowest_test_time': summary.get('max_duration', 0),
                    'fast_tests': len(timing_data.get('categorized_tests', {}).get('fast', [])),
                    'medium_tests': len(timing_data.get('categorized_tests', {}).get('medium', [])),
                    'slow_tests': len(timing_data.get('categorized_tests', {}).get('slow', [])),
                    'profile_data_available': 'error' not in profile_report
                }
            else:
                return {
                    'total_test_time': 0,
                    'average_test_time': 0,
                    'status': 'no_data'
                }
        except Exception as e:
            return {
                'total_test_time': 0,
                'average_test_time': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _collect_flaky_test_metrics(self) -> Dict:
        """Collect flaky test detection metrics."""
        try:
            from .test_flaky_detection import flaky_detector
            
            report = flaky_detector.generate_flaky_report()
            
            if 'summary' in report:
                summary = report['summary']
                return {
                    'total_tests_analyzed': summary.get('total_tests_analyzed', 0),
                    'flaky_tests_count': summary.get('flaky_tests_count', 0),
                    'flaky_percentage': summary.get('flaky_percentage', 0),
                    'most_flaky_test': summary.get('most_flaky', None),
                    'highest_failure_rate': summary.get('highest_failure_rate', 0),
                    'flaky_tests': report.get('flaky_tests', [])[:5]  # Top 5
                }
            else:
                return {
                    'total_tests_analyzed': 0,
                    'flaky_tests_count': 0,
                    'status': 'no_data'
                }
        except Exception as e:
            return {
                'total_tests_analyzed': 0,
                'flaky_tests_count': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _collect_fixture_cache_metrics(self) -> Dict:
        """Collect fixture caching performance metrics."""
        try:
            from .test_fixture_caching import fixture_manager
            
            stats = fixture_manager.get_cache_stats()
            
            return {
                'cache_hit_rate': stats.get('hit_rate', 0),
                'memory_hits': stats.get('memory_hits', 0),
                'file_hits': stats.get('file_hits', 0),
                'cache_misses': stats.get('misses', 0),
                'memory_usage_mb': stats.get('memory_usage_mb', 0),
                'memory_cache_size': stats.get('memory_cache_size', 0),
                'file_cache_size': stats.get('file_cache_size', 0)
            }
        except Exception as e:
            return {
                'cache_hit_rate': 0,
                'status': 'error',
                'error': str(e)
            }
    
    def _calculate_quality_score(self, metrics: Dict) -> float:
        """Calculate overall quality score from all metrics."""
        score = 0.0
        weight_total = 0.0
        
        # Coverage contribution (30%)
        coverage = metrics.get('coverage_metrics', {})
        if 'line_coverage' in coverage:
            line_cov = coverage['line_coverage']
            branch_cov = coverage.get('branch_coverage', 0)
            coverage_score = (line_cov * 0.7 + branch_cov * 0.3)
            score += coverage_score * 0.3
            weight_total += 0.3
        
        # Performance contribution (25%)
        performance = metrics.get('performance_metrics', {})
        if 'average_test_time' in performance:
            avg_time = performance['average_test_time']
            # Score based on test speed (lower is better)
            if avg_time > 0:
                perf_score = max(0, 100 - (avg_time * 10))  # Penalty for slow tests
                score += perf_score * 0.25
                weight_total += 0.25
        
        # Flaky test contribution (20%)
        flaky = metrics.get('flaky_test_metrics', {})
        if 'flaky_percentage' in flaky:
            flaky_pct = flaky['flaky_percentage']
            flaky_score = max(0, 100 - (flaky_pct * 5))  # Penalty for flaky tests
            score += flaky_score * 0.2
            weight_total += 0.2
        
        # Test discovery contribution (15%)
        test_metrics = metrics.get('test_metrics', {})
        if test_metrics.get('discovery_status') == 'success':
            score += 100 * 0.15
            weight_total += 0.15
        
        # Cache performance contribution (10%)
        cache = metrics.get('fixture_cache_metrics', {})
        if 'cache_hit_rate' in cache:
            cache_score = cache['cache_hit_rate'] * 100
            score += cache_score * 0.1
            weight_total += 0.1
        
        return score / weight_total if weight_total > 0 else 0
    
    def generate_html_dashboard(self, metrics: Dict) -> str:
        """Generate HTML dashboard with all metrics."""
        # Extract metrics with defaults first
        quality_score = metrics.get('quality_score', 0)
        coverage = metrics.get('coverage_metrics', {})
        performance = metrics.get('performance_metrics', {})
        flaky = metrics.get('flaky_test_metrics', {})
        cache = metrics.get('fixture_cache_metrics', {})
        test_metrics = metrics.get('test_metrics', {})
        
        # Calculate derived values
        reliability_score = 100 - flaky.get('flaky_percentage', 0)
        cache_hit_rate = cache.get('cache_hit_rate', 0)
        
        # Generate recommendations
        recommendations = self._generate_dashboard_recommendations(metrics)
        recommendations_html = ''.join(f'<li>{rec}</li>' for rec in recommendations)
        
        # Helper function for CSS classes
        def get_score_class(score):
            if score >= 90:
                return "score-excellent", "progress-excellent"
            elif score >= 70:
                return "score-good", "progress-good"
            else:
                return "score-poor", "progress-poor"
        
        quality_score_class, quality_progress_class = get_score_class(quality_score)
        coverage_score_class, coverage_progress_class = get_score_class(coverage.get('line_coverage', 0))
        reliability_score_class, reliability_progress_class = get_score_class(reliability_score)
        cache_score_class, cache_progress_class = get_score_class(cache_hit_rate * 100)
        
        discovery_status = test_metrics.get('discovery_status', 'unknown')
        discovery_status_class = {
            'success': 'status-success',
            'failed': 'status-warning',
            'error': 'status-error'
        }.get(discovery_status, 'status-error')
        
        # Create HTML content using string replacement instead of .format()
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TEMPL Pipeline QA Dashboard</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .dashboard {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; text-align: center; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); border-left: 4px solid #667eea; }}
        .metric-card h3 {{ margin: 0 0 15px 0; color: #333; }}
        .metric-value {{ font-size: 2em; font-weight: bold; margin-bottom: 10px; }}
        .metric-description {{ color: #666; font-size: 0.9em; }}
        .quality-score {{ font-size: 3em; font-weight: bold; text-align: center; }}
        .score-excellent {{ color: #27ae60; }}
        .score-good {{ color: #f39c12; }}
        .score-poor {{ color: #e74c3c; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #e0e0e0; border-radius: 10px; overflow: hidden; margin: 10px 0; }}
        .progress-fill {{ height: 100%; transition: width 0.3s ease; }}
        .progress-excellent {{ background-color: #27ae60; }}
        .progress-good {{ background-color: #f39c12; }}
        .progress-poor {{ background-color: #e74c3c; }}
        .detail-section {{ background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin-bottom: 20px; }}
        .timestamp {{ text-align: center; color: #666; font-size: 0.9em; }}
        .status-indicator {{ display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 5px; }}
        .status-success {{ background-color: #27ae60; }}
        .status-warning {{ background-color: #f39c12; }}
        .status-error {{ background-color: #e74c3c; }}
    </style>
</head>
<body>
    <div class="dashboard">
        <div class="header">
            <h1>TEMPL Pipeline QA Dashboard</h1>
            <p>Comprehensive Quality Assurance Metrics</p>
        </div>
        
        <div class="metrics-grid">
            <div class="metric-card">
                <h3>Overall Quality Score</h3>
                <div class="metric-value quality-score {quality_score_class}">{quality_score:.1f}/100</div>
                <div class="progress-bar">
                    <div class="progress-fill {quality_progress_class}" style="width: {quality_score}%"></div>
                </div>
                <div class="metric-description">Composite score based on all QA metrics</div>
            </div>
            
            <div class="metric-card">
                <h3>Test Coverage</h3>
                <div class="metric-value">{coverage.get('line_coverage', 0):.1f}%</div>
                <div class="progress-bar">
                    <div class="progress-fill {coverage_progress_class}" style="width: {coverage.get('line_coverage', 0)}%"></div>
                </div>
                <div class="metric-description">
                    Line: {coverage.get('line_coverage', 0):.1f}% | Branch: {coverage.get('branch_coverage', 0):.1f}%<br>
                    Files analyzed: {coverage.get('files_analyzed', 0)}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Test Performance</h3>
                <div class="metric-value">{performance.get('average_test_time', 0):.2f}s</div>
                <div class="metric-description">
                    Average test execution time<br>
                    Total: {performance.get('total_test_time', 0):.1f}s | Fast: {performance.get('fast_tests', 0)} | Medium: {performance.get('medium_tests', 0)} | Slow: {performance.get('slow_tests', 0)}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Test Reliability</h3>
                <div class="metric-value">{reliability_score:.1f}%</div>
                <div class="progress-bar">
                    <div class="progress-fill {reliability_progress_class}" style="width: {reliability_score}%"></div>
                </div>
                <div class="metric-description">
                    Flaky tests: {flaky.get('flaky_tests_count', 0)} ({flaky.get('flaky_percentage', 0):.1f}%)<br>
                    Total tests analyzed: {flaky.get('total_tests_analyzed', 0)}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Cache Performance</h3>
                <div class="metric-value">{cache_hit_rate:.1%}</div>
                <div class="progress-bar">
                    <div class="progress-fill {cache_progress_class}" style="width: {cache_hit_rate * 100}%"></div>
                </div>
                <div class="metric-description">
                    Memory hits: {cache.get('memory_hits', 0)} | File hits: {cache.get('file_hits', 0)}<br>
                    Memory usage: {cache.get('memory_usage_mb', 0):.1f}MB
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Test Discovery</h3>
                <div class="metric-value">
                    <span class="status-indicator {discovery_status_class}"></span>
                    {test_metrics.get('total_tests', 0)} tests
                </div>
                <div class="metric-description">
                    Status: {discovery_status}<br>
                    Last updated: {datetime.fromtimestamp(test_metrics.get('discovery_time', time.time())).strftime('%Y-%m-%d %H:%M:%S')}
                </div>
            </div>
        </div>
        
        <div class="detail-section">
            <h3>Quality Trends</h3>
            <p>Quality trend analysis would appear here with historical data.</p>
        </div>
        
        <div class="detail-section">
            <h3>Recommendations</h3>
            <ul>
                {recommendations_html}
            </ul>
        </div>
        
        <div class="timestamp">
            Last updated: {datetime.fromtimestamp(metrics['timestamp']).strftime('%Y-%m-%d %H:%M:%S')}
        </div>
    </div>
</body>
</html>"""
        
        return html_content
    
    def _generate_dashboard_recommendations(self, metrics: Dict) -> List[str]:
        """Generate actionable recommendations based on metrics."""
        recommendations = []
        
        coverage = metrics.get('coverage_metrics', {})
        performance = metrics.get('performance_metrics', {})
        flaky = metrics.get('flaky_test_metrics', {})
        
        # Coverage recommendations
        line_coverage = coverage.get('line_coverage', 0)
        if line_coverage < 80:
            recommendations.append(f"Increase test coverage from {line_coverage:.1f}% to at least 80%")
        
        branch_coverage = coverage.get('branch_coverage', 0)
        if branch_coverage < 70:
            recommendations.append(f"Improve branch coverage from {branch_coverage:.1f}% to at least 70%")
        
        # Performance recommendations
        avg_time = performance.get('average_test_time', 0)
        if avg_time > 5:
            recommendations.append(f"Optimize test performance - average execution time is {avg_time:.1f}s")
        
        slow_tests = performance.get('slow_tests', 0)
        if slow_tests > 5:
            recommendations.append(f"Consider optimizing {slow_tests} slow tests or running them separately")
        
        # Flaky test recommendations
        flaky_count = flaky.get('flaky_tests_count', 0)
        if flaky_count > 0:
            recommendations.append(f"Investigate and fix {flaky_count} flaky tests")
        
        # General recommendations
        if metrics.get('quality_score', 0) < 80:
            recommendations.append("Overall quality score is below 80% - focus on improving coverage and reliability")
        
        if not recommendations:
            recommendations.append("Quality metrics look good! Continue maintaining high standards.")
        
        return recommendations
    
    def save_dashboard(self, metrics: Dict) -> str:
        """Save the dashboard to HTML file."""
        html_content = self.generate_html_dashboard(metrics)
        dashboard_file = self.output_dir / "qa-dashboard.html"
        
        with open(dashboard_file, 'w') as f:
            f.write(html_content)
        
        # Also save metrics as JSON
        metrics_file = self.output_dir / "qa-metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return str(dashboard_file)
    
    def run_full_analysis(self) -> str:
        """Run complete QA analysis and generate dashboard."""
        print("Running comprehensive QA analysis...")
        
        metrics = self.collect_all_metrics()
        dashboard_file = self.save_dashboard(metrics)
        
        print(f"QA Dashboard generated: {dashboard_file}")
        print(f"Overall Quality Score: {metrics['quality_score']:.1f}/100")
        
        return dashboard_file


def main():
    """Main entry point for QA dashboard generation."""
    dashboard = QADashboard()
    dashboard_file = dashboard.run_full_analysis()
    
    print(f"\nQA Dashboard available at: file://{Path(dashboard_file).absolute()}")
    
    # Print summary to console
    metrics_file = dashboard.output_dir / "qa-metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        
        print("\n" + "="*60)
        print("QA DASHBOARD SUMMARY")
        print("="*60)
        print(f"Overall Quality Score: {metrics['quality_score']:.1f}/100")
        
        coverage = metrics.get('coverage_metrics', {})
        print(f"Coverage: {coverage.get('line_coverage', 0):.1f}% line, {coverage.get('branch_coverage', 0):.1f}% branch")
        
        performance = metrics.get('performance_metrics', {})
        print(f"Performance: {performance.get('average_test_time', 0):.2f}s average test time")
        
        flaky = metrics.get('flaky_test_metrics', {})
        print(f"Reliability: {flaky.get('flaky_tests_count', 0)} flaky tests ({flaky.get('flaky_percentage', 0):.1f}%)")
        
        print("="*60)


if __name__ == "__main__":
    main()