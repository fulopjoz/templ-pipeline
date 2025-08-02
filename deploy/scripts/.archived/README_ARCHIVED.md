# Archived Scripts

These scripts have been archived as part of the deployment optimization process.
They contained overlapping functionality that has been consolidated into the main deployment scripts.

## Archived Scripts:

- **build.sh** (282 lines) → Replaced by `deploy.sh build` + `deploy/docker/build-optimized.sh`
- **deploy-master.sh** → Replaced by `deploy.sh` with simpler commands
- **deploy.sh** (old version) → Replaced by main `deploy.sh`
- **deploy-complete.sh** → Replaced by `deploy.sh deploy`
- **rebuild-full.sh** → Replaced by `deploy.sh update`
- **copy-data.sh** → Functionality integrated into Docker/Kubernetes setup

## What Replaced Them:

### Simple Commands (from project root):
```bash
# Build image
./deploy.sh build -u USERNAME --push

# Full deployment
./deploy.sh deploy -u USERNAME -n NAMESPACE -d DOMAIN

# Update deployment
./deploy.sh update -u USERNAME -n NAMESPACE

# Quick code updates (30 seconds)
./quick-update.sh -n NAMESPACE
```

## Benefits:
- 90% fewer commands to remember
- Consistent interface
- Better error handling
- Optimized performance
- Clear separation of concerns

## Recovery:
These scripts are preserved here if you need to reference old functionality.
They can be restored if needed, but the new system provides all the same functionality with better usability.

Date archived: $(date)
Reason: Deployment optimization and simplification