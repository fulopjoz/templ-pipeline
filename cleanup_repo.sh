#!/bin/bash

echo "🧹 Starting repository cleanup..."

# Verify we're in the right directory
if [[ ! -f "run_pipeline.py" ]]; then
    echo "❌ Error: Not in TEMPL pipeline root directory"
    exit 1
fi

echo "✅ Confirmed: In TEMPL pipeline directory"

# 1. Remove unused streamlit files
echo "🗑️  Removing unused streamlit files..."
rm -f templ_pipeline/ui/archive/streamlit_app.py
rm -f templ_pipeline/ui/archive/streamlit_app.py.bak

# 2. Remove outdated backups
echo "🗑️  Removing outdated backup files..."
rm -f templ_pipeline/cli/main.py.backup
rm -f .cursor/rules/isolation_rules/main.mdc.backup

# 3. Remove Python cache directories (excluding virtual env)
echo "🗑️  Removing Python cache directories..."
find . -name "__pycache__" -type d -not -path "./.templ/*" -exec rm -rf {} + 2>/dev/null

# 4. Remove testing artifacts
echo "🗑️  Removing testing artifacts..."
rm -f .coverage
rm -rf htmlcov/
rm -rf .pytest_cache/
rm -rf tests/cli/.pytest_cache/
rm -rf templ_pipeline.egg-info/

# 5. Remove timestamped output directories (keep current output/)
echo "🗑️  Removing old test output directories..."
rm -rf output_2025*

# 6. Remove incomplete duplicate directory
echo "🗑️  Removing incomplete duplicate directory..."
rm -rf templ_pipeline/templ_pipeline/

echo "✨ Cleanup complete!"
echo "📊 Repository size before/after:"
du -sh . 2>/dev/null || echo "Size calculation unavailable"

echo ""
echo "🔍 Verification - these commands should work normally:"
echo "   python run_pipeline.py --help"
echo "   python run_streamlit_app.py (should launch app.py)"
echo "   python -m pytest tests/ (should run tests)"

