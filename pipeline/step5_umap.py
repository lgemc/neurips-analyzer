#!/usr/bin/env python3
"""
Step 5: Compute UMAP 2D coordinates for visualization.
"""
import sys
import numpy as np
import umap
from db import get_connection, deserialize_embedding


def load_embeddings(db_path: str = "neurips.db") -> tuple[np.ndarray, list[int]]:
    """
    Load all embeddings from database.

    Returns:
        Tuple of (embeddings array, paper IDs list)
    """
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT id, embedding FROM papers WHERE embedding IS NOT NULL ORDER BY id")

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


def add_umap_columns(db_path: str = "neurips.db") -> None:
    """Add UMAP coordinate columns to papers table if they don't exist."""
    conn = get_connection(db_path)

    # Check if columns exist
    cursor = conn.execute("PRAGMA table_info(papers)")
    columns = [row[1] for row in cursor.fetchall()]

    if 'umap_x' not in columns:
        print("Adding umap_x column...")
        conn.execute("ALTER TABLE papers ADD COLUMN umap_x REAL")

    if 'umap_y' not in columns:
        print("Adding umap_y column...")
        conn.execute("ALTER TABLE papers ADD COLUMN umap_y REAL")

    conn.commit()
    conn.close()


def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42
) -> np.ndarray:
    """
    Compute 2D UMAP projection of embeddings.

    Args:
        embeddings: Array of embeddings (n_samples, n_features)
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        metric: Distance metric
        random_state: Random seed

    Returns:
        2D coordinates (n_samples, 2)
    """
    print(f"Computing UMAP projection with n_neighbors={n_neighbors}, min_dist={min_dist}...")

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        verbose=True
    )

    coordinates = reducer.fit_transform(embeddings)

    print(f"✓ UMAP complete. Shape: {coordinates.shape}")

    return coordinates


def store_umap_coordinates(
    db_path: str,
    paper_ids: list[int],
    coordinates: np.ndarray
) -> None:
    """
    Store UMAP coordinates in database.

    Args:
        db_path: Database path
        paper_ids: List of paper IDs
        coordinates: 2D UMAP coordinates
    """
    conn = get_connection(db_path)

    print("Storing UMAP coordinates...")

    for paper_id, (x, y) in zip(paper_ids, coordinates):
        conn.execute(
            "UPDATE papers SET umap_x = ?, umap_y = ? WHERE id = ?",
            (float(x), float(y), paper_id)
        )

    conn.commit()
    conn.close()

    print(f"✓ Stored coordinates for {len(paper_ids)} papers")


def generate_umap(
    db_path: str = "neurips.db",
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42
) -> None:
    """
    Main UMAP generation function.

    Args:
        db_path: Database path
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        metric: Distance metric
        random_state: Random seed
    """
    # Add columns if needed
    add_umap_columns(db_path)

    # Load embeddings
    print("Loading embeddings from database...")
    embeddings, paper_ids = load_embeddings(db_path)
    print(f"Loaded {len(embeddings)} papers with embeddings")

    # Compute UMAP
    coordinates = compute_umap(embeddings, n_neighbors, min_dist, metric, random_state)

    # Store coordinates
    store_umap_coordinates(db_path, paper_ids, coordinates)

    print("\n✓ UMAP coordinates generated successfully")


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute UMAP 2D coordinates for visualization")
    parser.add_argument(
        "-d", "--db",
        default="neurips.db",
        help="Database path (default: neurips.db)"
    )
    parser.add_argument(
        "--n-neighbors",
        type=int,
        default=15,
        help="Number of neighbors for UMAP (default: 15)"
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.1,
        help="Minimum distance for UMAP (default: 0.1)"
    )
    parser.add_argument(
        "--metric",
        default="cosine",
        help="Distance metric (default: cosine)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    try:
        generate_umap(
            db_path=args.db,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.seed
        )
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
