# TEMPL Pipeline - Architecture Diagram & Documentation

## Overview
TEMPL (Template-based Protein Ligand) is a pose prediction pipeline that leverages ligand similarity and template superposition for fast, accurate protein-ligand binding pose predictions. The system provides both CLI and web interfaces with a modular, scalable architecture.

---

## System Architecture Diagram

```mermaid
graph TB
    %% External Interfaces
    subgraph "User Interfaces"
        CLI[CLI Interface<br/>templ command]
        WEB[Web Application<br/>Streamlit UI]
        API[Python API<br/>Programmatic Access]
    end

    %% Entry Points
    subgraph "Application Entry Points"
        MAIN_CLI[run_pipeline.py<br/>CLI Entry Point]
        MAIN_WEB[run_streamlit_app.py<br/>Web Entry Point]
        SETUP[setup_templ_env.sh<br/>Environment Setup]
    end

    %% Core Pipeline
    subgraph "Core Pipeline Components"
        PIPELINE[TEMPLPipeline<br/>Main Orchestrator]
        
        subgraph "Processing Modules"
            EMBED[Embedding Manager<br/>ESM-2 Protein Embeddings]
            TEMPLATES[Template Engine<br/>K-NN Template Search]
            MCS[MCS Engine<br/>Maximum Common Substructure]
            CONFORMER[Conformer Generator<br/>Constrained ETKDG v3]
            SCORING[Scoring Engine<br/>Shape/Color/Combo Scoring]
        end
    end

    %% Data Layer
    subgraph "Data & Storage"
        PDBBIND[(PDBBind Database<br/>Protein-Ligand Complexes)]
        EMBEDDINGS[(Protein Embeddings<br/>Pre-computed ESM-2)]
        LIGANDS[(Processed Ligands<br/>Standardized SDF)]
        POLARIS[(Polaris Benchmark<br/>Validation Dataset)]
        OUTPUT[(Output Files<br/>SDF + Metadata)]
    end

    %% Supporting Services
    subgraph "Supporting Services"
        HARDWARE[Hardware Detection<br/>CPU/GPU/RAM Optimization]
        VALIDATION[Input Validation<br/>SMILES/PDB Validation]
        LOGGING[Logging System<br/>Progress Indicators]
        CONFIG[Configuration<br/>UX Adaptive Settings]
    end

    %% External Dependencies
    subgraph "External Dependencies"
        RDKIT[RDKit<br/>Chemical Informatics]
        BIOTITE[Biotite<br/>Structural Biology]
        TORCH[PyTorch<br/>ML Models]
        STREAMLIT[Streamlit<br/>Web Framework]
    end

    %% Flow Connections
    CLI --> MAIN_CLI
    WEB --> MAIN_WEB
    API --> PIPELINE
    
    MAIN_CLI --> PIPELINE
    MAIN_WEB --> PIPELINE
    
    PIPELINE --> EMBED
    PIPELINE --> TEMPLATES
    PIPELINE --> MCS
    PIPELINE --> CONFORMER
    PIPELINE --> SCORING
    
    EMBED --> EMBEDDINGS
    TEMPLATES --> PDBBIND
    TEMPLATES --> LIGANDS
    MCS --> RDKIT
    CONFORMER --> RDKIT
    SCORING --> RDKIT
    
    PIPELINE --> OUTPUT
    PIPELINE --> HARDWARE
    PIPELINE --> VALIDATION
    PIPELINE --> LOGGING
    
    %% Benchmark Connections
    POLARIS --> PIPELINE
    
    %% Styling
    classDef interface fill:#e1f5fe,stroke:#0277bd,stroke-width:2px
    classDef core fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef data fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef service fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef external fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    
    class CLI,WEB,API interface
    class PIPELINE,EMBED,TEMPLATES,MCS,CONFORMER,SCORING core
    class PDBBIND,EMBEDDINGS,LIGANDS,POLARIS,OUTPUT data
    class HARDWARE,VALIDATION,LOGGING,CONFIG service
    class RDKIT,BIOTITE,TORCH,STREAMLIT external
```

---

## Pipeline Flow Diagram

```mermaid
flowchart TD
    START([Start Pipeline]) --> INPUT{Input Type?}
    
    %% Input Processing
    INPUT -->|SMILES + Protein| VALIDATE[Validate Inputs<br/>SMILES & PDB Check]
    INPUT -->|SDF + Protein| VALIDATE
    INPUT -->|Batch Mode| VALIDATE
    
    VALIDATE --> PREP[Prepare Query<br/>Standardize Molecule]
    
    %% Protein Processing
    PREP --> PROTEIN_CHECK{Protein Embedding<br/>Available?}
    PROTEIN_CHECK -->|No| GENERATE_EMB[Generate ESM-2<br/>Embedding]
    PROTEIN_CHECK -->|Yes| LOAD_EMB[Load Cached<br/>Embedding]
    
    GENERATE_EMB --> EMBED_STORE[(Store Embedding)]
    LOAD_EMB --> TEMPLATE_SEARCH
    EMBED_STORE --> TEMPLATE_SEARCH
    
    %% Template Search
    TEMPLATE_SEARCH[K-NN Template Search<br/>Similarity Matching] --> LOAD_TEMPLATES[Load Template<br/>Molecules]
    
    %% MCS & Pose Generation
    LOAD_TEMPLATES --> MCS_CALC[Calculate MCS<br/>Query vs Templates]
    MCS_CALC --> CONSTRAINTS[Generate Constraints<br/>From MCS Alignment]
    CONSTRAINTS --> CONFORMERS[Generate Conformers<br/>Constrained ETKDG]
    
    %% Scoring & Selection
    CONFORMERS --> SCORING_PHASE[Multi-metric Scoring]
    
    subgraph "Scoring Methods"
        SHAPE[Shape Similarity<br/>USRCAT]
        COLOR[Pharmacophore<br/>Color Matching]
        COMBO[Combined Score<br/>Weighted Average]
    end
    
    SCORING_PHASE --> SHAPE
    SCORING_PHASE --> COLOR
    SCORING_PHASE --> COMBO
    
    SHAPE --> SELECT[Select Best Poses<br/>Per Scoring Method]
    COLOR --> SELECT
    COMBO --> SELECT
    
    %% Output Generation
    SELECT --> SAVE[Save Results<br/>SDF + Metadata]
    SAVE --> VISUALIZE[Generate<br/>Visualizations]
    VISUALIZE --> END([Pipeline Complete])
    
    %% Error Handling
    VALIDATE -->|Error| ERROR[Validation Error]
    GENERATE_EMB -->|Error| ERROR
    TEMPLATE_SEARCH -->|Error| ERROR
    MCS_CALC -->|Error| ERROR
    CONFORMERS -->|Error| ERROR
    
    ERROR --> CLEANUP[Cleanup Resources]
    CLEANUP --> FAIL([Pipeline Failed])
    
    %% Styling
    classDef startEnd fill:#4caf50,stroke:#2e7d32,stroke-width:2px,color:#fff
    classDef process fill:#2196f3,stroke:#1565c0,stroke-width:2px,color:#fff
    classDef decision fill:#ff9800,stroke:#ef6c00,stroke-width:2px,color:#fff
    classDef data fill:#9c27b0,stroke:#6a1b9a,stroke-width:2px,color:#fff
    classDef error fill:#f44336,stroke:#c62828,stroke-width:2px,color:#fff
    
    class START,END startEnd
    class VALIDATE,PREP,GENERATE_EMB,LOAD_EMB,TEMPLATE_SEARCH,LOAD_TEMPLATES,MCS_CALC,CONSTRAINTS,CONFORMERS,SCORING_PHASE,SELECT,SAVE,VISUALIZE,CLEANUP process
    class INPUT,PROTEIN_CHECK decision
    class EMBED_STORE data
    class ERROR,FAIL error
```

---

## DevOps Architecture & Deployment

```mermaid
graph TB
    %% Development Environment
    subgraph "Development Environment"
        DEV_LOCAL[Local Development<br/>uv + Python 3.9+]
        DEV_TEST[Testing<br/>pytest + benchmarks]
        DEV_LINT[Code Quality<br/>black + isort + flake8]
    end
    
    %% CI/CD Pipeline
    subgraph "CI/CD Pipeline"
        GIT[Git Repository<br/>Version Control]
        CI[Continuous Integration<br/>Test + Build + Package]
        CD[Continuous Deployment<br/>Environment Promotion]
    end
    
    %% Container Infrastructure
    subgraph "Containerization"
        DOCKER[Docker Container<br/>Multi-stage Build]
        REGISTRY[Container Registry<br/>Image Storage]
        ORCHESTRATION[Kubernetes/Docker Compose<br/>Container Orchestration]
    end
    
    %% Deployment Environments
    subgraph "Deployment Environments"
        DEV_ENV[Development<br/>Feature Testing]
        STAGING[Staging<br/>Integration Testing]
        PROD[Production<br/>Live System]
    end
    
    %% Infrastructure
    subgraph "Infrastructure Components"
        COMPUTE[Compute Resources<br/>CPU/GPU Instances]
        STORAGE[Persistent Storage<br/>Data + Models]
        NETWORK[Load Balancer<br/>SSL Termination]
        MONITORING[Monitoring<br/>Metrics + Logging]
    end
    
    %% Data Management
    subgraph "Data Pipeline"
        DATA_INGESTION[Data Ingestion<br/>PDBBind + Polaris]
        DATA_PROCESSING[Data Processing<br/>Preprocessing + Validation]
        DATA_VERSIONING[Data Versioning<br/>DVC + Git LFS]
        BACKUP[Backup & Recovery<br/>Automated Snapshots]
    end
    
    %% Security & Compliance
    subgraph "Security & Compliance"
        SECRETS[Secrets Management<br/>Environment Variables]
        ACCESS[Access Control<br/>RBAC + Authentication]
        AUDIT[Audit Logging<br/>Compliance Tracking]
        SCANNING[Security Scanning<br/>Vulnerability Assessment]
    end
    
    %% Connections
    DEV_LOCAL --> GIT
    DEV_TEST --> CI
    DEV_LINT --> CI
    
    GIT --> CI
    CI --> DOCKER
    DOCKER --> REGISTRY
    REGISTRY --> ORCHESTRATION
    
    CD --> DEV_ENV
    CD --> STAGING
    CD --> PROD
    
    ORCHESTRATION --> DEV_ENV
    ORCHESTRATION --> STAGING
    ORCHESTRATION --> PROD
    
    DEV_ENV --> COMPUTE
    STAGING --> COMPUTE
    PROD --> COMPUTE
    
    COMPUTE --> STORAGE
    COMPUTE --> NETWORK
    COMPUTE --> MONITORING
    
    DATA_INGESTION --> DATA_PROCESSING
    DATA_PROCESSING --> DATA_VERSIONING
    DATA_VERSIONING --> BACKUP
    
    %% Styling
    classDef dev fill:#e8f5e8,stroke:#388e3c,stroke-width:2px
    classDef cicd fill:#e3f2fd,stroke:#1976d2,stroke-width:2px
    classDef container fill:#f3e5f5,stroke:#7b1fa2,stroke-width:2px
    classDef deploy fill:#fff3e0,stroke:#f57c00,stroke-width:2px
    classDef infra fill:#fce4ec,stroke:#c2185b,stroke-width:2px
    classDef data fill:#e0f2f1,stroke:#00695c,stroke-width:2px
    classDef security fill:#ffebee,stroke:#d32f2f,stroke-width:2px
    
    class DEV_LOCAL,DEV_TEST,DEV_LINT dev
    class GIT,CI,CD cicd
    class DOCKER,REGISTRY,ORCHESTRATION container
    class DEV_ENV,STAGING,PROD deploy
    class COMPUTE,STORAGE,NETWORK,MONITORING infra
    class DATA_INGESTION,DATA_PROCESSING,DATA_VERSIONING,BACKUP data
    class SECRETS,ACCESS,AUDIT,SCANNING security
```

---

## Component Details

### Core Components

| Component | Purpose | Key Technologies | Scalability |
|-----------|---------|------------------|-------------|
| **TEMPLPipeline** | Main orchestrator | Python, asyncio | Horizontal via workers |
| **Embedding Manager** | Protein embeddings | ESM-2, PyTorch, CUDA | GPU acceleration |
| **Template Engine** | Similarity search | NumPy, scikit-learn | Vector databases |
| **MCS Engine** | Molecular alignment | RDKit, C++ bindings | Parallel processing |
| **Conformer Generator** | 3D structure gen | RDKit ETKDG | Multi-threading |
| **Scoring Engine** | Pose evaluation | USRCAT, pharmacophore | Vectorized ops |

### Interface Components

| Interface | Technology | Features | Deployment |
|-----------|------------|----------|------------|
| **CLI** | argparse, rich | Batch processing, scripting | Native executable |
| **Web App** | Streamlit | Interactive UI, visualizations | Container deployment |
| **Python API** | Direct imports | Programmatic access | Library package |

### Data Management

| Data Type | Format | Storage | Versioning |
|-----------|--------|---------|------------|
| **Protein Structures** | PDB | File system | Git LFS |
| **Ligand Database** | SDF | Compressed archives | Content hashing |
| **Embeddings** | NPZ | Binary files | Semantic versioning |
| **Benchmarks** | JSON/CSV | Version controlled | Data lineage |
| **Results** | SDF + metadata | Timestamped dirs | Automated archival |

---

## Deployment Strategies

### Development Setup
```bash
# Quick start with auto-detection
source setup_templ_env.sh

# Development with all features
source setup_templ_env.sh --dev

# Production minimal
source setup_templ_env.sh --minimal
```

### Container Deployment
```dockerfile
# Multi-stage build
FROM python:3.11-slim as builder
# ... build dependencies

FROM python:3.11-slim as runtime
# ... runtime configuration
EXPOSE 8501
CMD ["python", "run_streamlit_app.py"]
```

### Kubernetes Deployment
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: templ-pipeline
spec:
  replicas: 3
  selector:
    matchLabels:
      app: templ-pipeline
  template:
    metadata:
      labels:
        app: templ-pipeline
    spec:
      containers:
      - name: templ-pipeline
        image: templ-pipeline:latest
        ports:
        - containerPort: 8501
        env:
        - name: STREAMLIT_SERVER_HEADLESS
          value: "true"
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "16Gi"
            cpu: "8"
```

---

## DevOps Best Practices Implemented

### 1. **Infrastructure as Code**
- Automated environment setup (`setup_templ_env.sh`)
- Hardware auto-detection and optimization
- Containerized deployment configurations

### 2. **Monitoring & Observability**
- Rich progress indicators and logging
- Performance metrics collection
- Error tracking and diagnostics

### 3. **Scalability & Performance**
- Multi-processing support with worker pools
- Hardware-aware resource allocation
- Caching for embeddings and templates

### 4. **Security**
- Input validation and sanitization
- Dependency scanning via `deptry`
- Minimal container surfaces

### 5. **Testing & Quality**
- Comprehensive test suite (`pytest`)
- Code formatting (`black`, `isort`)
- Linting and type checking (`flake8`, `mypy`)

### 6. **Documentation & UX**
- Progressive CLI interface
- Contextual help system
- User experience adaptation

---

## Performance Characteristics

### Typical Processing Times
- **Embedding Generation**: 30-120 seconds (protein size dependent)
- **Template Search**: 1-5 seconds (1000 templates)
- **Pose Generation**: 10-60 seconds (100-200 conformers)
- **Total Pipeline**: 1-3 minutes per query

### Resource Requirements
- **Minimum**: 4GB RAM, 4 CPU cores
- **Recommended**: 16GB RAM, 8+ CPU cores
- **GPU Acceleration**: 4GB+ VRAM for embeddings
- **Storage**: 2-10GB for datasets

### Scalability Limits
- **Concurrent Users**: 10-50 (web interface)
- **Batch Processing**: 100s-1000s molecules
- **Template Database**: Up to 100K structures
- **Memory Usage**: 2-8GB per pipeline instance

---

## Maintenance & Operations

### Regular Tasks
- **Database Updates**: Monthly PDBBind releases
- **Model Updates**: Quarterly embedding model updates
- **Performance Tuning**: Weekly optimization reviews
- **Security Patches**: Automated dependency updates

### Monitoring Metrics
- **Pipeline Success Rate**: >95% target
- **Average Processing Time**: <3 minutes
- **System Resource Usage**: <80% capacity
- **Error Rate**: <1% of requests

### Backup & Recovery
- **Data Backup**: Daily incremental, weekly full
- **Configuration Backup**: Git-based versioning
- **Disaster Recovery**: <4 hour RTO target
- **Data Retention**: 1 year for results, 5 years for datasets

---

*This diagram provides a comprehensive overview of the TEMPL Pipeline architecture, designed for clarity and operational understanding. The system follows modern DevOps practices with emphasis on scalability, maintainability, and user experience.* 