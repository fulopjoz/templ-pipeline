# Memory Bank: Tasks - Repository Cleanup & Git Optimization

**Current Status:** 📋 **PLANNING IN PROGRESS**  
**Task Type:** Level 2 - Simple Enhancement  
**Priority:** HIGH  
**Started:** 2025-06-25

## Task Overview

**Problem:** Repository contains redundant files and outdated configurations after successful optimization. Large data files needed for benchmarking are currently tracked by Git, causing GitHub repository bloat.

**Goal:** Clean up redundant/outdated files while keeping large data locally for benchmarking but properly excluding it from Git tracking.

## Revised Strategy: Smart Cleanup + Git Optimization

### 🎯 Key Principles
1. **Keep large data locally** for benchmarking purposes
2. **Exclude large data from Git** via improved .gitignore
3. **Remove truly redundant files** that are no longer needed
4. **Maintain clean GitHub repository** while preserving local functionality

## Current Repository Analysis

### 📊 File Categories

#### ✅ KEEP (Essential Files)
**Core Application:**
- `templ_pipeline/` - Main application code
- `requirements.txt` - Python dependencies  
- `pyproject.toml` - Project configuration
- `run_streamlit_app.py` - Application entry point
- `run_pipeline.py` - Pipeline execution

**Optimized Deployment (Current Best Practice):**
- `data-minimal/` - Optimized data for web app (98MB)
- `Dockerfile.minimal` - Optimized Docker build
- `templ-app-minimal.yaml` - DigitalOcean configuration
- `deploy-minimal.sh` - Deployment automation

**Development Tools:**
- `.git/` - Version control
- `memory-bank/` - Task tracking and documentation
- `README.md` - Documentation

#### 🏠 KEEP LOCALLY BUT EXCLUDE FROM GIT
**Large Data (for benchmarking):**
- `data/` - Keep locally, improve .gitignore exclusion
  - `data/PDBBind/` (11GB) - Needed for benchmarking
  - `data/embeddings/` (172MB) - Full embeddings for research
  - `data/ligands/` - Full ligand database
  - `data/polaris/` - Benchmarking data
  - Other benchmark/research data

**Large Output Files:**
- `output/` - Benchmark results and analysis
- `templ_benchmark_results_polaris/` - Benchmark outputs

#### ❌ DELETE (Truly Redundant/Outdated)

**Outdated Deployment Files (~50MB):**
- `Dockerfile` - Original version (superseded by Dockerfile.minimal)
- `Dockerfile.optimized` - Intermediate version (superseded)
- `app.yaml` - Original config (superseded by templ-app-minimal.yaml)
- `templ-app.yaml` - Original DigitalOcean config (superseded)
- `templ-app-optimized.yaml` - Intermediate config (superseded)
- `deploy-optimized.sh` - Intermediate deployment script (superseded)

**Temporary/Test Files:**
- `test_minimal_streamlit.py` - Temporary test file
- `test_minimal_app.py` - Temporary test file  
- `debug_app.py` - Debug script no longer needed
- `debug_plan.md` - Debug documentation
- `cleanup_repo.sh` - Previous cleanup script
- `test-docker-local.sh` - Temporary test script

**Build Artifacts (can be regenerated):**
- `templ_pipeline.egg-info/` - Build artifacts
- `.coverage` - Test coverage data
- `htmlcov/` - HTML coverage reports
- `.pytest_cache/` - Pytest cache

**Outdated Documentation:**
- `DEPLOYMENT_EXECUTION_SUMMARY.md` - Superseded by memory-bank/
- `DEPLOYMENT_GUIDE.md` - Superseded by memory-bank/
- `DEPLOYMENT_STATUS.md` - Superseded by memory-bank/
- `project_brief.txt` - Large file superseded by memory-bank/

## Implementation Plan

### Phase 1: Git Optimization (20 minutes)

#### 1.1 Improve .gitignore
```bash
# Add comprehensive data exclusions
echo "" >> .gitignore
echo "# Large data files (keep locally for benchmarking)" >> .gitignore
echo "data/" >> .gitignore
echo "!data/README.md" >> .gitignore
echo "" >> .gitignore
echo "# Large output files" >> .gitignore
echo "output/" >> .gitignore
echo "templ_benchmark_results_polaris/" >> .gitignore
echo "" >> .gitignore
echo "# Temporary deployment files" >> .gitignore
echo "Dockerfile.optimized" >> .gitignore
echo "templ-app-optimized.yaml" >> .gitignore
echo "deploy-optimized.sh" >> .gitignore
```

#### 1.2 Remove large files from Git tracking
```bash
git rm -r --cached data/
git rm --cached output/ templ_benchmark_results_polaris/
git commit -m "Remove large data files from Git tracking (keep locally)"
```

### Phase 2: Remove Redundant Files (15 minutes)

#### 2.1 Remove outdated deployment files
```bash
rm -f Dockerfile Dockerfile.optimized
rm -f app.yaml templ-app.yaml templ-app-optimized.yaml  
rm -f deploy-optimized.sh
rm -f test-docker-local.sh
```

#### 2.2 Remove temporary/test files
```bash
rm -f test_minimal_*.py debug_*.py debug_plan.md
rm -f cleanup_repo.sh
```

#### 2.3 Remove build artifacts
```bash
rm -rf templ_pipeline.egg-info/
rm -f .coverage
rm -rf htmlcov/ .pytest_cache/
```

#### 2.4 Remove outdated documentation
```bash
rm -f DEPLOYMENT_*.md project_brief.txt
```

### Phase 3: Repository Structure Optimization (10 minutes)

#### 3.1 Verify .dockerignore excludes large data
- Ensure `data/` is excluded from Docker builds
- Keep `data-minimal/` included for web app builds

#### 3.2 Update README.md
- Document the data structure (data/ vs data-minimal/)
- Explain benchmarking vs web app deployment
- Update deployment instructions

#### 3.3 Clean Git history (optional)
```bash
git add .
git commit -m "Clean up redundant files and optimize repository structure"
```

### Phase 4: Validation (10 minutes)

#### 4.1 Verify essential functionality
- Test `Dockerfile.minimal` builds successfully
- Confirm `deploy-minimal.sh` works
- Verify data-minimal/ contains essential files

#### 4.2 Verify Git optimization
- Check `git status` shows clean repository
- Confirm large files are ignored but present locally
- Verify GitHub repository size will be manageable

## Success Criteria

### 🎯 Git Repository Optimization
- ✅ Large data files excluded from Git tracking
- ✅ GitHub repository size <100MB (vs GB+ before)
- ✅ Large files still available locally for benchmarking

### 🧹 File Cleanup
- ✅ Remove redundant deployment configurations
- ✅ Remove temporary/test files
- ✅ Remove outdated documentation
- ✅ Keep essential functionality intact

### 🚀 Deployment Clarity
- ✅ Single optimized deployment path (minimal)
- ✅ Clear separation: benchmarking (local) vs web app (deployed)
- ✅ Fast Docker builds with minimal context

## Expected Results

### Repository Size (GitHub)
- **Before:** Multiple GB (large data tracked)
- **After:** <100MB (essential files only)
- **Local:** Full data preserved for benchmarking

### File Organization
- **Before:** Multiple deployment configs, redundant files
- **After:** Single optimized path, clean structure
- **Benefit:** Clear development workflow

### Git Performance
- **Before:** Slow operations due to large files
- **After:** Fast Git operations, manageable repository
- **Benefit:** Better collaboration, faster clones

## Risk Assessment

### ✅ Very Low Risk
- Improving .gitignore (non-destructive)
- Removing redundant deployment files (superseded versions)
- Removing temporary/test files (can be recreated)

### ⚠️ Low Risk
- Removing large files from Git tracking (files remain locally)
- Removing build artifacts (can be regenerated)

### 🛡️ Mitigation Strategies
1. **Files remain locally** - no data loss for benchmarking
2. **Git history preserved** - can recover if needed
3. **Essential files protected** - only redundant files removed
4. **Testing at each step** - verify functionality maintained

## Implementation Checklist

### Pre-Cleanup Verification
- [ ] Confirm data-minimal/ has all essential files for web app
- [ ] Test Dockerfile.minimal builds successfully
- [ ] Verify deploy-minimal.sh works
- [ ] Check current Git status

### Cleanup Execution
- [ ] Update .gitignore with comprehensive exclusions
- [ ] Remove large files from Git tracking (keep locally)
- [ ] Delete redundant deployment files
- [ ] Remove temporary/test files
- [ ] Clean build artifacts
- [ ] Remove outdated documentation

### Post-Cleanup Validation
- [ ] Test minimal deployment still works
- [ ] Verify large data accessible locally for benchmarking
- [ ] Confirm Git repository is clean and manageable
- [ ] Update documentation as needed

## Next Steps

1. **PLAN MODE:** Complete this planning document ✅
2. **IMPLEMENT MODE:** Execute the smart cleanup plan
3. **REFLECT MODE:** Document results and optimized workflow

---

**Planning Status:** ✅ **COMPLETE**  
**Strategy:** Smart cleanup focusing on redundancy, not essential data  
**Estimated Implementation Time:** 55 minutes  
**Expected Success Rate:** 99%+ (very low risk approach)

**Key Insight:** Keep large data locally for benchmarking, exclude from Git for clean GitHub repository, remove only truly redundant files.
