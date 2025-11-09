# NeurIPS Papers Clustering Pipeline

A simple, SQLite-based pipeline for clustering NeurIPS papers using embeddings and soft clustering.

## Architecture

```
pipeline/
├── db.py              # Database schema & utilities
├── step1_load.py      # Load CSV → papers table
├── step2_embed.py     # Compute & store embeddings
├── step3_cluster.py   # GMM clustering + associations
├── step4_describe.py  # Generate cluster names (uses Anthropic API)
└── run.py            # Simple orchestrator
```

## Database Schema

**papers**
- `id`, `type`, `name`, `virtualsite_url`, `speakers_authors`, `abstract`
- `embedding` (BLOB): numpy array serialized as float32 bytes

**clusters**
- `id`, `name`, `description`

**cluster_associations**
- `cluster_id`, `paper_id`, `score` (soft clustering probability)

## Quick Start

```bash
# Run full pipeline
python pipeline/run.py "NeurIPS 2025 Events.csv"

# Run specific steps
python pipeline/run.py "NeurIPS 2025 Events.csv" --steps 1 2 3

# Custom clustering parameters
python pipeline/run.py "NeurIPS 2025 Events.csv" --n-clusters 10 --model all-mpnet-base-v2
```

## Individual Steps

### Step 1: Load CSV
```bash
python pipeline/step1_load.py "NeurIPS 2025 Events.csv" neurips.db --replace
```

### Step 2: Compute Embeddings
```bash
python pipeline/step2_embed.py -d neurips.db -m all-MiniLM-L6-v2
```

### Step 3: Cluster Papers
```bash
# Auto-determine optimal k using silhouette score
python pipeline/step3_cluster.py -d neurips.db

# Or specify k manually
python pipeline/step3_cluster.py -d neurips.db -k 10
```

### Step 4: Generate Cluster Descriptions
```bash
export ANTHROPIC_API_KEY="your-api-key"
python pipeline/step4_describe.py -d neurips.db -n 20
```

## Features

- **Soft Clustering**: Uses Gaussian Mixture Models (GMM) instead of hard KMeans
- **Optimal k**: Auto-determines number of clusters using silhouette score
- **Score-based associations**: Papers can belong to multiple clusters with probabilities
- **SQLite storage**: Single file, no external dependencies
- **Idempotent steps**: Can re-run individual steps safely
- **Low dependencies**: Just sklearn, sentence-transformers, anthropic, pandas, numpy

## Example Queries

```python
import sqlite3

conn = sqlite3.connect('neurips.db')

# Get top papers in a cluster
cursor = conn.execute("""
    SELECT p.name, ca.score
    FROM papers p
    JOIN cluster_associations ca ON p.id = ca.paper_id
    WHERE ca.cluster_id = 1
    ORDER BY ca.score DESC
    LIMIT 10
""")

# Get all clusters for a paper
cursor = conn.execute("""
    SELECT c.name, ca.score
    FROM clusters c
    JOIN cluster_associations ca ON c.id = ca.cluster_id
    WHERE ca.paper_id = 123
    ORDER BY ca.score DESC
""")

# Get cluster statistics
cursor = conn.execute("""
    SELECT c.id, c.name, COUNT(*) as paper_count
    FROM clusters c
    JOIN cluster_associations ca ON c.id = ca.cluster_id
    WHERE ca.score > 0.1
    GROUP BY c.id
    ORDER BY paper_count DESC
""")
```

## Dependencies

Already in `pyproject.toml`:
- `pandas`, `numpy`
- `scikit-learn` (GMM, silhouette score)
- `sentence-transformers` (embeddings)
- `anthropic` (cluster descriptions)

## Notes

- Embeddings are computed from `name + abstract` (not authors)
- Default model: `all-MiniLM-L6-v2` (384 dimensions, fast)
- GMM allows papers to belong to multiple clusters
- Cluster descriptions use top 20 papers by score
- Auto k-selection tests range [3, 15] by default
