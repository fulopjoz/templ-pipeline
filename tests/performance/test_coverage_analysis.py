"""
Advanced coverage analysis with branch coverage and detailed reporting.
Provides comprehensive coverage metrics and analysis tools.
"""

import ast
import json
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import coverage
import pytest
from collections import defaultdict
import subprocess
import sys


class CoverageAnalyzer:
    """Advanced coverage analysis with branch coverage and quality metrics."""
    
    def __init__(self, source_dir: str = "templ_pipeline", 
                 output_dir: str = "coverage-analysis"):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.coverage_data = {}
        self.branch_data = {}
        self.critical_paths = set()
        
    def analyze_critical_paths(self) -> Set[str]:
        """Identify critical code paths that require high coverage."""
        critical_patterns = [
            "pipeline.py",
            "mcs.py", 
            "embedding.py",
            "scoring.py",
            "main.py",
            "__init__.py"
        ]
        
        critical_files = set()
        
        for pattern in critical_patterns:
            for file_path in self.source_dir.rglob(f"*{pattern}"):
                if file_path.is_file() and file_path.suffix == ".py":
                    critical_files.add(str(file_path.relative_to(self.source_dir)))
        
        self.critical_paths = critical_files
        return critical_files
    
    def run_coverage_analysis(self, test_args: List[str] = None) -> Dict:
        """Run comprehensive coverage analysis."""
        test_args = test_args or ["tests/"]
        
        # Initialize coverage with branch coverage enabled
        cov = coverage.Coverage(
            source=[str(self.source_dir)],
            branch=True,
            config_file=True,
            data_file=str(self.output_dir / ".coverage")
        )
        
        cov.start()
        
        # Run tests
        exit_code = subprocess.run([
            sys.executable, "-m", "pytest", 
            "--tb=short", 
            "-q"
        ] + test_args, 
        cwd=Path.cwd(),
        capture_output=False
        ).returncode
        
        cov.stop()
        cov.save()
        
        # Generate reports
        self._generate_detailed_reports(cov)
        
        return {
            "exit_code": exit_code,
            "coverage_file": str(self.output_dir / ".coverage"),
            "reports_dir": str(self.output_dir)
        }
    
    def _generate_detailed_reports(self, cov: coverage.Coverage):
        """Generate detailed coverage reports."""
        # Standard HTML report
        html_dir = self.output_dir / "html"
        cov.html_report(directory=str(html_dir))
        
        # XML report for CI
        xml_file = self.output_dir / "coverage.xml"
        cov.xml_report(outfile=str(xml_file))
        
        # JSON report for analysis
        json_file = self.output_dir / "coverage.json"
        with open(json_file, 'w') as f:
            json.dump(self._extract_coverage_data(cov), f, indent=2)
        
        # Generate custom analysis
        analysis = self.analyze_coverage_quality(cov)
        analysis_file = self.output_dir / "coverage-analysis.json"
        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Generate summary report
        self._generate_summary_report(analysis)
    
    def _extract_coverage_data(self, cov: coverage.Coverage) -> Dict:
        """Extract detailed coverage data."""
        data = {}
        
        for filename in cov.get_data().measured_files():
            try:
                rel_path = str(Path(filename).relative_to(self.source_dir))
            except ValueError:
                continue
            
            analysis = cov.analyze(filename)
            
            data[rel_path] = {
                'statements': analysis.statements,
                'missing': analysis.missing,
                'excluded': analysis.excluded,
                'branches': getattr(analysis, 'branches', []),
                'partial_branches': getattr(analysis, 'partial_branches', []),
                'missing_branches': getattr(analysis, 'missing_branches', []),
                'coverage_percent': (len(analysis.statements) - len(analysis.missing)) / len(analysis.statements) * 100 if analysis.statements else 0
            }
        
        return data
    
    def analyze_coverage_quality(self, cov: coverage.Coverage) -> Dict:
        """Analyze coverage quality and generate metrics."""
        self.analyze_critical_paths()
        coverage_data = self._extract_coverage_data(cov)
        
        # Overall metrics
        total_statements = sum(len(data['statements']) for data in coverage_data.values())
        total_missing = sum(len(data['missing']) for data in coverage_data.values())
        total_branches = sum(len(data['branches']) for data in coverage_data.values())
        total_missing_branches = sum(len(data['missing_branches']) for data in coverage_data.values())
        
        overall_line_coverage = (total_statements - total_missing) / total_statements * 100 if total_statements > 0 else 0
        overall_branch_coverage = (total_branches - total_missing_branches) / total_branches * 100 if total_branches > 0 else 0
        
        # Critical path analysis
        critical_coverage = {}
        for file_path in self.critical_paths:
            if file_path in coverage_data:
                data = coverage_data[file_path]
                critical_coverage[file_path] = {
                    'line_coverage': data['coverage_percent'],
                    'has_branches': len(data['branches']) > 0,
                    'branch_coverage': (len(data['branches']) - len(data['missing_branches'])) / len(data['branches']) * 100 if data['branches'] else 100,
                    'missing_lines': len(data['missing']),
                    'missing_branches': len(data['missing_branches'])
                }
        
        # Identify problem areas
        low_coverage_files = []
        for file_path, data in coverage_data.items():
            if data['coverage_percent'] < 80:  # Less than 80% coverage
                low_coverage_files.append({
                    'file': file_path,
                    'coverage': data['coverage_percent'],
                    'missing_lines': len(data['missing']),
                    'missing_branches': len(data['missing_branches']),
                    'is_critical': file_path in self.critical_paths
                })
        
        low_coverage_files.sort(key=lambda x: x['coverage'])
        
        # Coverage trends (would need historical data)
        coverage_trends = self._analyze_coverage_trends()
        
        analysis = {
            'timestamp': time.time(),
            'overall_metrics': {
                'line_coverage': overall_line_coverage,
                'branch_coverage': overall_branch_coverage,
                'total_statements': total_statements,
                'total_missing': total_missing,
                'total_branches': total_branches,
                'total_missing_branches': total_missing_branches,
                'files_analyzed': len(coverage_data)
            },
            'critical_path_coverage': critical_coverage,
            'low_coverage_files': low_coverage_files,
            'coverage_quality_score': self._calculate_quality_score(
                overall_line_coverage, overall_branch_coverage, critical_coverage
            ),
            'recommendations': self._generate_coverage_recommendations(
                overall_line_coverage, overall_branch_coverage, 
                low_coverage_files, critical_coverage
            ),
            'coverage_trends': coverage_trends,
            'detailed_data': coverage_data
        }
        
        return analysis
    
    def _calculate_quality_score(self, line_coverage: float, 
                               branch_coverage: float, 
                               critical_coverage: Dict) -> float:
        """Calculate overall coverage quality score (0-100)."""
        # Base score from overall coverage
        base_score = (line_coverage * 0.6 + branch_coverage * 0.4)
        
        # Critical path penalty
        critical_penalty = 0
        if critical_coverage:
            critical_line_avg = sum(data['line_coverage'] for data in critical_coverage.values()) / len(critical_coverage)
            critical_branch_avg = sum(data['branch_coverage'] for data in critical_coverage.values()) / len(critical_coverage)
            
            if critical_line_avg < 90:
                critical_penalty += (90 - critical_line_avg) * 0.5
            if critical_branch_avg < 85:
                critical_penalty += (85 - critical_branch_avg) * 0.3
        
        return max(0, min(100, base_score - critical_penalty))
    
    def _generate_coverage_recommendations(self, line_coverage: float,
                                         branch_coverage: float,
                                         low_coverage_files: List[Dict],
                                         critical_coverage: Dict) -> List[str]:
        """Generate actionable coverage recommendations."""
        recommendations = []
        
        if line_coverage < 80:
            recommendations.append(f"Increase overall line coverage from {line_coverage:.1f}% to at least 80%")
        
        if branch_coverage < 70:
            recommendations.append(f"Increase branch coverage from {branch_coverage:.1f}% to at least 70%")
        
        # Critical path recommendations
        low_critical = [f for f, data in critical_coverage.items() if data['line_coverage'] < 90]
        if low_critical:
            recommendations.append(f"Prioritize coverage for critical files: {', '.join(low_critical[:3])}")
        
        # File-specific recommendations
        worst_files = low_coverage_files[:5]
        if worst_files:
            recommendations.append(f"Focus on low-coverage files: {', '.join(f['file'] for f in worst_files)}")
        
        # Branch coverage recommendations
        low_branch_files = [f for f, data in critical_coverage.items() if data['branch_coverage'] < 80 and data['has_branches']]
        if low_branch_files:
            recommendations.append(f"Add branch coverage tests for: {', '.join(low_branch_files[:3])}")
        
        return recommendations
    
    def _analyze_coverage_trends(self) -> Dict:
        """Analyze coverage trends over time."""
        # This would analyze historical coverage data
        # For now, return placeholder structure
        return {
            'trend_available': False,
            'message': 'Historical data needed for trend analysis',
            'suggestion': 'Run coverage analysis regularly to build trend data'
        }
    
    def _generate_summary_report(self, analysis: Dict):
        """Generate human-readable summary report."""
        summary_file = self.output_dir / "coverage-summary.md"
        
        overall = analysis['overall_metrics']
        quality_score = analysis['coverage_quality_score']
        
        content = f"""# Coverage Analysis Summary
        
## Overall Metrics
- **Line Coverage**: {overall['line_coverage']:.1f}%
- **Branch Coverage**: {overall['branch_coverage']:.1f}%
- **Quality Score**: {quality_score:.1f}/100
- **Files Analyzed**: {overall['files_analyzed']}

## Critical Path Coverage
"""
        
        for file_path, data in analysis['critical_path_coverage'].items():
            content += f"- **{file_path}**: {data['line_coverage']:.1f}% lines, {data['branch_coverage']:.1f}% branches\n"
        
        content += "\n## Low Coverage Files\n"
        for file_data in analysis['low_coverage_files'][:10]:
            critical_marker = " (CRITICAL)" if file_data['is_critical'] else ""
            content += f"- **{file_data['file']}**: {file_data['coverage']:.1f}%{critical_marker}\n"
        
        content += "\n## Recommendations\n"
        for rec in analysis['recommendations']:
            content += f"- {rec}\n"
        
        with open(summary_file, 'w') as f:
            f.write(content)
    
    def generate_coverage_badge(self, analysis: Dict) -> str:
        """Generate coverage badge data."""
        line_coverage = analysis['overall_metrics']['line_coverage']
        
        if line_coverage >= 90:
            color = "brightgreen"
        elif line_coverage >= 80:
            color = "green"
        elif line_coverage >= 70:
            color = "yellow"
        elif line_coverage >= 60:
            color = "orange"
        else:
            color = "red"
        
        badge_data = {
            "schemaVersion": 1,
            "label": "coverage",
            "message": f"{line_coverage:.1f}%",
            "color": color
        }
        
        badge_file = self.output_dir / "coverage-badge.json"
        with open(badge_file, 'w') as f:
            json.dump(badge_data, f)
        
        return str(badge_file)


class CoverageMonitor:
    """Monitor coverage during test execution."""
    
    def __init__(self):
        self.analyzer = CoverageAnalyzer()
        
    def monitor_test_coverage(self, test_function):
        """Decorator to monitor coverage for individual tests."""
        def wrapper(*args, **kwargs):
            # This would integrate with the test execution
            # to provide per-test coverage metrics
            return test_function(*args, **kwargs)
        return wrapper


# Global coverage analyzer
coverage_analyzer = CoverageAnalyzer()


def pytest_configure(config):
    """Configure coverage monitoring."""
    config.addinivalue_line(
        "markers",
        "coverage_critical: Mark test as covering critical code paths"
    )


def pytest_sessionfinish(session, exitstatus):
    """Generate coverage analysis at session end."""
    if os.getenv('PYTEST_COVERAGE_ANALYSIS'):
        print("\n" + "="*80)
        print("GENERATING COVERAGE ANALYSIS...")
        print("="*80)
        
        try:
            result = coverage_analyzer.run_coverage_analysis()
            analysis_file = coverage_analyzer.output_dir / "coverage-analysis.json"
            
            if analysis_file.exists():
                with open(analysis_file, 'r') as f:
                    analysis = json.load(f)
                
                overall = analysis['overall_metrics']
                print(f"Line Coverage: {overall['line_coverage']:.1f}%")
                print(f"Branch Coverage: {overall['branch_coverage']:.1f}%")
                print(f"Quality Score: {analysis['coverage_quality_score']:.1f}/100")
                
                if analysis['recommendations']:
                    print("\nTop Recommendations:")
                    for rec in analysis['recommendations'][:3]:
                        print(f"  • {rec}")
                
                print(f"\nDetailed reports available in: {coverage_analyzer.output_dir}")
                print("="*80)
        
        except Exception as e:
            print(f"Coverage analysis failed: {e}")


@pytest.fixture
def coverage_analyzer_fixture():
    """Fixture to provide coverage analysis capabilities."""
    return coverage_analyzer


def test_coverage_analysis_functionality():
    """Test the coverage analysis functionality."""
    analyzer = CoverageAnalyzer(".", "test-coverage-output")
    
    # Test critical path identification
    critical_paths = analyzer.analyze_critical_paths()
    assert isinstance(critical_paths, set)
    
    # Test quality score calculation
    score = analyzer._calculate_quality_score(85.0, 75.0, {})
    assert 0 <= score <= 100
    
    # Test recommendations generation
    recommendations = analyzer._generate_coverage_recommendations(
        75.0, 65.0, [], {}
    )
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0


@pytest.mark.coverage_critical
def test_coverage_monitor():
    """Test coverage monitoring capabilities."""
    monitor = CoverageMonitor()
    
    @monitor.monitor_test_coverage
    def sample_function():
        return "test result"
    
    result = sample_function()
    assert result == "test result"


if __name__ == "__main__":
    # Run coverage analysis
    analyzer = CoverageAnalyzer()
    print("Running coverage analysis...")
    result = analyzer.run_coverage_analysis()
    print(f"Analysis complete. Results in: {result['reports_dir']}")