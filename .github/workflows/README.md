# GitHub Actions Workflows

This directory contains GitHub Actions workflows for the TEMPL Pipeline project.

## Workflows

### CI (`ci.yml`)

The main CI workflow that runs on pushes to `master` and `dev` branches, and on pull requests to `master`.

#### Jobs

1. **Test** - Runs tests across Python 3.9, 3.10, 3.11, and 3.12
2. **SonarCloud Analysis** - Performs code quality analysis (only on pushes to master/dev)
3. **Lint** - Runs code formatting and linting checks
4. **Security** - Runs security vulnerability scans

#### Key Features

- **Parallel Execution**: All Python versions run tests in parallel
- **Fault Tolerance**: Jobs continue even if individual tests fail (`fail-fast: false`)
- **Artifact Management**: Unique artifact names per Python version to prevent conflicts
- **Coverage Merging**: Combines coverage reports from all Python versions for SonarCloud
- **Fallback Handling**: Creates empty reports when tools fail to generate output

#### Artifact Naming Convention

- `coverage-reports-{python-version}` - Coverage reports per Python version
- `test-results-{python-version}` - Test results per Python version  
- `security-reports-{python-version}` - Security scan reports per Python version

#### Error Handling

- **Missing Files**: Creates empty XML/JSON files when tools fail to generate output
- **Upload Failures**: Uses `if: always()` to ensure artifacts are uploaded even if tests fail
- **Coverage Merging**: Handles cases where no coverage files are available

### Live Tests (`livetests.yml`)

Runs live integration tests to verify the application works end-to-end.

### Citation File Format (`cffconvert.yml`)

Validates and converts citation files for proper academic attribution.

## Troubleshooting

### Common Issues

1. **Artifact Conflicts**: Ensure unique artifact names per job/matrix combination
2. **Missing Files**: Check that tools generate expected output files
3. **Job Cancellations**: Use `fail-fast: false` to prevent cascading failures

### Debugging

- Check workflow logs for specific error messages
- Verify file paths and permissions
- Ensure all required dependencies are installed
