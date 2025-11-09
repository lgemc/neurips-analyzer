#!/usr/bin/env python3
"""
Step 3: Cluster papers using Gaussian Mixture Models with soft clustering.
"""
import sys
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from db import get_connection, deserialize_embedding, clear_clusters


def load_embeddings(db_path: str = "neurips.db") -> tuple[np.ndarray, list[int]]:
    """
    Load all embeddings from database.

    Returns:
        Tuple of (embeddings array, paper IDs list)
    """
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT id, embedding FROM papers WHERE embedding IS NOT NULL")

    embeddings = []
    paper_ids = []

    for row in cursor:
        paper_ids.append(row['id'])
        # Determine dimension from first embedding
        if not embeddings:
            embedding = np.frombuffer(row['embedding'], dtype=np.float32)
            dim = len(embedding)
            embeddings.append(embedding)
        else:
            embedding = deserialize_embedding(row['embedding'], dim)
            embeddings.append(embedding)

    conn.close()

    if not embeddings:
        raise ValueError("No embeddings found in database. Run step2_embed.py first.")

    return np.array(embeddings), paper_ids


def find_optimal_k(
    embeddings: np.ndarray,
    min_k: int = 3,
    max_k: int = 15,
    random_state: int = 42
) -> tuple[int, dict]:
    """
    Find optimal number of clusters using silhouette score.

    Args:
        embeddings: Array of embeddings
        min_k: Minimum number of clusters to try
        max_k: Maximum number of clusters to try
        random_state: Random seed

    Returns:
        Tuple of (optimal k, scores dict)
    """
    print(f"Finding optimal k in range [{min_k}, {max_k}]...")

    scores = {}
    best_k = min_k
    best_score = -1

    for k in range(min_k, max_k + 1):
        print(f"  Trying k={k}...")

        # Fit GMM
        gmm = GaussianMixture(
            n_components=k,
            covariance_type='full',
            random_state=random_state,
            n_init=3,
            max_iter=100
        )
        labels = gmm.fit_predict(embeddings)

        # Calculate silhouette score
        score = silhouette_score(embeddings, labels, metric='cosine', sample_size=min(10000, len(embeddings)))
        scores[k] = score

        print(f"    Silhouette score: {score:.4f}")

        if score > best_score:
            best_score = score
            best_k = k

    print(f"\n✓ Optimal k={best_k} with silhouette score={best_score:.4f}")

    return best_k, scores


def cluster_with_gmm(
    embeddings: np.ndarray,
    n_clusters: int,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Cluster embeddings using GMM.

    Args:
        embeddings: Array of embeddings
        n_clusters: Number of clusters
        random_state: Random seed

    Returns:
        Tuple of (hard labels, soft probabilities matrix)
    """
    print(f"Clustering {len(embeddings)} papers into {n_clusters} clusters...")

    gmm = GaussianMixture(
        n_components=n_clusters,
        covariance_type='full',
        random_state=random_state,
        n_init=10,
        max_iter=200
    )

    gmm.fit(embeddings)

    # Get hard labels (most likely cluster)
    labels = gmm.predict(embeddings)

    # Get soft probabilities for each cluster
    probabilities = gmm.predict_proba(embeddings)

    return labels, probabilities


def store_clusters(
    db_path: str,
    paper_ids: list[int],
    probabilities: np.ndarray,
    min_score: float = 0.01
) -> int:
    """
    Store clusters and associations in database.

    Args:
        db_path: Database path
        paper_ids: List of paper IDs
        probabilities: Soft cluster probabilities (n_papers x n_clusters)
        min_score: Minimum probability to store association

    Returns:
        Number of clusters created
    """
    n_clusters = probabilities.shape[1]

    # Clear existing clusters
    clear_clusters(db_path)

    conn = get_connection(db_path)

    # Create cluster entries
    print(f"Creating {n_clusters} cluster entries...")
    for cluster_id in range(n_clusters):
        conn.execute(
            "INSERT INTO clusters (name, description) VALUES (?, ?)",
            (f"Cluster {cluster_id}", "")
        )
    conn.commit()

    # Store associations
    print("Storing cluster associations...")
    association_count = 0

    for paper_idx, paper_id in enumerate(paper_ids):
        # Get cluster probabilities for this paper
        probs = probabilities[paper_idx]

        # Store associations for clusters above threshold
        for cluster_idx, score in enumerate(probs):
            if score >= min_score:
                # cluster_id in DB is 1-indexed (AUTOINCREMENT starts at 1)
                conn.execute(
                    "INSERT INTO cluster_associations (cluster_id, paper_id, score) VALUES (?, ?, ?)",
                    (cluster_idx + 1, paper_id, float(score))
                )
                association_count += 1

        if (paper_idx + 1) % 1000 == 0:
            conn.commit()
            print(f"  Processed {paper_idx + 1}/{len(paper_ids)} papers...")

    conn.commit()
    conn.close()

    print(f"✓ Created {association_count} cluster associations")

    return n_clusters


def cluster_papers(
    db_path: str = "neurips.db",
    n_clusters: int = None,
    min_k: int = 3,
    max_k: int = 15,
    random_state: int = 42,
    min_score: float = 0.01
) -> int:
    """
    Main clustering function.

    Args:
        db_path: Database path
        n_clusters: Fixed number of clusters (if None, auto-determine)
        min_k: Minimum k to try when auto-determining
        max_k: Maximum k to try when auto-determining
        random_state: Random seed
        min_score: Minimum probability to store association

    Returns:
        Number of clusters created
    """
    # Load embeddings
    print("Loading embeddings from database...")
    embeddings, paper_ids = load_embeddings(db_path)
    print(f"Loaded {len(embeddings)} papers with embeddings")

    # Determine optimal k if not specified
    if n_clusters is None:
        n_clusters, scores = find_optimal_k(embeddings, min_k, max_k, random_state)
    else:
        print(f"Using fixed k={n_clusters}")

    # Cluster
    labels, probabilities = cluster_with_gmm(embeddings, n_clusters, random_state)

    # Store results
    n_created = store_clusters(db_path, paper_ids, probabilities, min_score)

    # Print cluster distribution
    print("\nCluster distribution (primary assignments):")
    unique, counts = np.unique(labels, return_counts=True)
    for cluster_id, count in zip(unique, counts):
        print(f"  Cluster {cluster_id}: {count} papers")

    return n_created


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Cluster papers using GMM")
    parser.add_argument(
        "-d", "--db",
        default="neurips.db",
        help="Database path (default: neurips.db)"
    )
    parser.add_argument(
        "-k", "--n-clusters",
        type=int,
        help="Number of clusters (if not set, auto-determine using silhouette score)"
    )
    parser.add_argument(
        "--min-k",
        type=int,
        default=3,
        help="Minimum k for auto-determination (default: 3)"
    )
    parser.add_argument(
        "--max-k",
        type=int,
        default=15,
        help="Maximum k for auto-determination (default: 15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--min-score",
        type=float,
        default=0.01,
        help="Minimum probability to store association (default: 0.01)"
    )

    args = parser.parse_args()

    try:
        n_clusters = cluster_papers(
            db_path=args.db,
            n_clusters=args.n_clusters,
            min_k=args.min_k,
            max_k=args.max_k,
            random_state=args.seed,
            min_score=args.min_score
        )
        print(f"\n✓ Successfully created {n_clusters} clusters")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
