# TEMPL Pipeline: Simple Lean Deployment

Minimal always-on deployment following CERIT-SC best practices:
- CPU requests set to minimum (1 CPU) for long-term running
- CPU limits set higher (2 CPU) for burst during predictions
- No GPU — app runs CPU-only with pre-computed embeddings
- PDBBind data on PVC for template structure access

## Resource Usage

| Resource | Request (guaranteed) | Limit (burst) |
|----------|---------------------|---------------|
| CPU | 1 | 2 |
| Memory | 4Gi | 6Gi |
| GPU | none | none |
| Storage | 5Gi ephemeral | 10Gi ephemeral |
| Data PVC | 20Gi (templ-data-pvc) | shared NFS |

## Deploy

```bash
chmod +x deploy.sh
./deploy.sh                    # Deploy
./deploy.sh --dry-run          # Preview
```

## Files

| File | Purpose |
|------|---------|
| `deployment.yaml` | Main app: 1 CPU, 4Gi, no GPU, replicas=1 |
| `configmap.yaml` | App config (CPU-tuned, no CUDA) |
| `service.yaml` | ClusterIP with session affinity |
| `ingress.yaml` | TLS ingress for templ.dyn.cloud.e-infra.cz |
| `deploy.sh` | One-command deployment |

## Alternative: Scale-to-Zero

For zero idle cost (at the expense of 60-90s startup latency), see
`../scale-to-zero/` which uses a gatekeeper proxy + idle scaler CronJob.
