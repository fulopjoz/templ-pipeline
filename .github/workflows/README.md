# GitHub Actions Workflows

This directory contains the GitHub Actions workflows for the TEMPL Pipeline project.

## Workflow Overview

### Primary Workflows

1. **`ci.yml`** - Main CI pipeline (recommended)
   - Runs tests across Python 3.9-3.12
   - Includes linting (black, isort, flake8, mypy)
   - Security checks (bandit, safety)
   - Coverage reporting to Codecov
   - Artifact uploads for test results

### Specialized Workflows

3. **`sonarcloud.yml`** - Code quality analysis
   - Runs SonarCloud analysis
   - Generates coverage reports
   - Quality gate enforcement

4. **`cffconvert.yml`** - Citation file management
   - Validates and converts citation.cff
   - Generates multiple citation formats
   - Runs on citation file changes

5. **`livetests.yml`** - Manual test execution
   - Workflow dispatch for manual testing
   - Integration, performance, and UI test options
   - Useful for debugging and validation

### Legacy Workflows (Deprecated)

- **`tests.yml`** - Replaced by `ci.yml`
  - Kept for backward compatibility
  - Will be removed in future releases

## Best Practices Implemented

### Performance Optimizations
- **Dependency Caching**: All workflows use pip caching with hash-based keys
- **Fail-Fast Strategy**: Matrix builds fail fast to save resources
- **Artifact Management**: Test results and coverage reports are preserved
- **Parallel Execution**: Jobs run in parallel where possible

### Security & Quality
- **Security Scanning**: Bandit and Safety checks for vulnerabilities
- **Code Quality**: Black, isort, flake8, and mypy for code standards
- **Coverage Tracking**: Comprehensive coverage reporting
- **Artifact Retention**: Configurable retention periods

### Maintainability
- **Consistent Structure**: All workflows follow the same pattern
- **Environment Variables**: Centralized configuration
- **Documentation**: Clear job names and descriptions
- **Error Handling**: Graceful failure handling with artifacts

## Configuration

### Required Secrets
- `CODECOV_TOKEN`: For coverage reporting
- `SONAR_TOKEN`: For SonarCloud analysis

### Branch Strategy
- **master**: Production-ready code
- **dev**: Development branch
- **feature branches**: Individual features

## Usage

### Automatic Triggers
- Push to master/dev: Runs CI, SonarCloud
- Pull requests: Runs CI and SonarCloud
- Citation changes: Runs cffconvert

### Manual Triggers
- Live tests: Use workflow dispatch for specific test types

## Troubleshooting

### Common Issues
1. **Dependency Installation Failures**: Check Python version compatibility
2. **Test Failures**: Review test artifacts for detailed logs
3. **Coverage Issues**: Verify Codecov token configuration

### Debugging
- All workflows generate artifacts for inspection
- Test results include JUnit XML reports
- Coverage reports are available for analysis
- Security reports provide vulnerability details

## Future Improvements

1. **Performance**: Consider using GitHub Actions cache for larger dependencies
2. **Security**: Add container scanning for Docker images
3. **Monitoring**: Implement workflow metrics and alerting
4. **Automation**: Add automated dependency updates
