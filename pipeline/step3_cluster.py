#!/usr/bin/env python3
"""
Step 3: Cluster papers using Gaussian Mixture Models with soft clustering.
"""
import sys
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from db import get_connection, deserialize_embedding, clear_clusters


def load_embeddings(db_path: str = "neurips.db", year: int = None) -> tuple[np.ndarray, list[int]]:
    """
    Load embeddings from database, optionally filtered by year.

    Args:
        db_path: Database path
        year: Optional year to filter by

    Returns:
        Tuple of (embeddings array, paper IDs list)
    """
    conn = get_connection(db_path)
    if year is not None:
        cursor = conn.execute("SELECT id, embedding FROM papers WHERE embedding IS NOT NULL AND year = ?", (year,))
    else:
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


def compute_cluster_vectors(
    db_path: str,
    embeddings: np.ndarray,
    paper_ids: list[int],
    probabilities: np.ndarray,
    cluster_db_ids: list[int],
    top_n: int = 100
) -> None:
    """
    Compute and store cluster representative vectors using weighted centroid.

    Args:
        db_path: Database path
        embeddings: Paper embeddings array
        paper_ids: List of paper IDs
        probabilities: Soft cluster probabilities (n_papers x n_clusters)
        cluster_db_ids: Actual database IDs for clusters
        top_n: Number of top papers to use for cluster vector (default: 100)
    """
    from db import serialize_embedding

    n_clusters = probabilities.shape[1]
    conn = get_connection(db_path)

    print(f"Computing cluster vectors from top {top_n} papers per cluster...")

    for cluster_idx in range(n_clusters):
        # Get scores for this cluster
        cluster_scores = probabilities[:, cluster_idx]

        # Get top N papers by score
        top_indices = np.argsort(cluster_scores)[::-1][:top_n]

        # Get embeddings and scores for top papers
        top_embeddings = embeddings[top_indices]
        top_scores = cluster_scores[top_indices]

        # Compute weighted centroid
        # Normalize weights to sum to 1
        weights = top_scores / top_scores.sum()
        cluster_vector = np.average(top_embeddings, weights=weights, axis=0)

        # Store cluster vector using actual database ID
        conn.execute(
            "UPDATE clusters SET embedding = ? WHERE id = ?",
            (serialize_embedding(cluster_vector), cluster_db_ids[cluster_idx])
        )

        if (cluster_idx + 1) % 5 == 0:
            print(f"  Processed {cluster_idx + 1}/{n_clusters} clusters...")

    conn.commit()
    conn.close()

    print(f"✓ Computed and stored {n_clusters} cluster vectors")


def store_clusters(
    db_path: str,
    paper_ids: list[int],
    probabilities: np.ndarray,
    year: int,
    min_score: float = 0.01
) -> tuple[int, list[int]]:
    """
    Store clusters and associations in database.

    Args:
        db_path: Database path
        paper_ids: List of paper IDs
        probabilities: Soft cluster probabilities (n_papers x n_clusters)
        year: Year for these clusters
        min_score: Minimum probability to store association

    Returns:
        Tuple of (number of clusters created, list of cluster database IDs)
    """
    n_clusters = probabilities.shape[1]

    conn = get_connection(db_path)

    # Create cluster entries and track their actual IDs
    print(f"Creating {n_clusters} cluster entries for year {year}...")
    cluster_db_ids = []
    for cluster_id in range(n_clusters):
        cursor = conn.execute(
            "INSERT INTO clusters (name, description, year) VALUES (?, ?, ?)",
            (f"Cluster {cluster_id} ({year})", "", year)
        )
        cluster_db_ids.append(cursor.lastrowid)
    conn.commit()

    print(f"Created clusters with IDs: {cluster_db_ids[0]}-{cluster_db_ids[-1]}")

    # Store associations
    print("Storing cluster associations...")
    association_count = 0

    for paper_idx, paper_id in enumerate(paper_ids):
        # Get cluster probabilities for this paper
        probs = probabilities[paper_idx]

        # Store associations for clusters above threshold
        for cluster_idx, score in enumerate(probs):
            if score >= min_score:
                # Use the actual database cluster ID
                conn.execute(
                    "INSERT INTO cluster_associations (cluster_id, paper_id, score) VALUES (?, ?, ?)",
                    (cluster_db_ids[cluster_idx], paper_id, float(score))
                )
                association_count += 1

        if (paper_idx + 1) % 1000 == 0:
            conn.commit()
            print(f"  Processed {paper_idx + 1}/{len(paper_ids)} papers...")

    conn.commit()
    conn.close()

    print(f"✓ Created {association_count} cluster associations")

    return n_clusters, cluster_db_ids


def cluster_papers(
    db_path: str = "neurips.db",
    n_clusters: int = 7,
    min_k: int = 3,
    max_k: int = 15,
    random_state: int = 42,
    min_score: float = 0.01,
    top_n_for_vector: int = 100,
    year: int = None
) -> int:
    """
    Main clustering function - clusters papers per year.

    Args:
        db_path: Database path
        n_clusters: Number of clusters per year (default: 7)
        min_k: Minimum k to try when auto-determining
        max_k: Maximum k to try when auto-determining
        random_state: Random seed
        min_score: Minimum probability to store association
        top_n_for_vector: Number of top papers for cluster vector (default: 100)
        year: Optional specific year to cluster (default: None clusters all years)

    Returns:
        Total number of clusters created
    """
    # Clear existing clusters
    clear_clusters(db_path)

    # Get available years
    conn = get_connection(db_path)
    if year is not None:
        years = [year]
    else:
        cursor = conn.execute("SELECT DISTINCT year FROM papers WHERE embedding IS NOT NULL ORDER BY year")
        years = [row['year'] for row in cursor.fetchall()]
    conn.close()

    if not years:
        raise ValueError("No papers with embeddings found")

    print(f"Clustering papers for {len(years)} year(s): {min(years)}-{max(years)}")

    total_clusters = 0

    for year_val in years:
        print(f"\n{'='*60}")
        print(f"Processing year {year_val}")
        print(f"{'='*60}")

        # Load embeddings for this year
        print(f"Loading embeddings for {year_val}...")
        embeddings, paper_ids = load_embeddings(db_path, year=year_val)
        print(f"Loaded {len(embeddings)} papers with embeddings")

        if len(embeddings) < min_k:
            print(f"Warning: Only {len(embeddings)} papers in {year_val}, skipping (need at least {min_k})")
            continue

        # Determine optimal k if not specified
        if n_clusters is None:
            year_n_clusters, scores = find_optimal_k(embeddings, min_k, max_k, random_state)
        else:
            year_n_clusters = n_clusters
            print(f"Using k={year_n_clusters} clusters")

        # Cluster
        labels, probabilities = cluster_with_gmm(embeddings, year_n_clusters, random_state)

        # Store results
        n_created, cluster_db_ids = store_clusters(db_path, paper_ids, probabilities, year_val, min_score)

        # Compute and store cluster vectors
        compute_cluster_vectors(db_path, embeddings, paper_ids, probabilities, cluster_db_ids, top_n_for_vector)

        # Print cluster distribution
        print(f"\nCluster distribution for {year_val} (primary assignments):")
        unique, counts = np.unique(labels, return_counts=True)
        for cluster_id, count in zip(unique, counts):
            print(f"  Cluster {cluster_id}: {count} papers")

        total_clusters += n_created

    return total_clusters


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
        default=7,
        help="Number of clusters (default: 7)"
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
    parser.add_argument(
        "--top-n",
        type=int,
        default=100,
        help="Number of top papers for cluster vector (default: 100)"
    )
    parser.add_argument(
        "-y", "--year",
        type=int,
        default=None,
        help="Cluster only a specific year (default: all years)"
    )

    args = parser.parse_args()

    try:
        n_clusters = cluster_papers(
            db_path=args.db,
            n_clusters=args.n_clusters,
            min_k=args.min_k,
            max_k=args.max_k,
            random_state=args.seed,
            min_score=args.min_score,
            top_n_for_vector=args.top_n,
            year=args.year
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
