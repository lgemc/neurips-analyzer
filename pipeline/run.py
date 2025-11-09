#!/usr/bin/env python3
"""
Simple pipeline orchestrator - runs all steps in order.
"""
import sys
import argparse
from pathlib import Path

# Import step functions
from step1_load import load_csv, load_folder
from step2_embed import compute_embeddings
from step3_cluster import cluster_papers
from step4_describe import update_cluster_descriptions
from step5_umap import generate_umap


def run_pipeline(
    csv_path: str,
    db_path: str = "neurips.db",
    steps: list[int] = None,
    replace: bool = False,
    model: str = "all-MiniLM-L6-v2",
    n_clusters: int = 7,
    min_k: int = 3,
    max_k: int = 15,
    top_n: int = 20,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    metric: str = 'cosine',
    random_state: int = 42
) -> bool:
    """
    Run the complete pipeline or specific steps.

    Args:
        csv_path: Path to input CSV file or folder
        db_path: Database path
        steps: List of steps to run (1-5). If None, run all
        replace: Replace existing papers in step 1
        model: Sentence-transformer model name
        n_clusters: Number of clusters (default: 7)
        min_k: Minimum k for auto-determination
        max_k: Maximum k for auto-determination
        top_n: Number of top papers for cluster descriptions
        n_neighbors: Number of neighbors for UMAP
        min_dist: Minimum distance for UMAP
        metric: Distance metric for UMAP
        random_state: Random seed

    Returns:
        True if successful
    """
    if steps is None:
        steps = [1, 2, 3, 4, 5]

    print("="*80)
    print("NeurIPS Papers Clustering Pipeline")
    print("="*80)
    print(f"Database: {db_path}")
    print(f"Steps: {steps}")
    print("="*80)

    try:
        # Step 1: Load CSV(s)
        if 1 in steps:
            print("\n[Step 1/5] Loading data into database...")
            path = Path(csv_path)

            if not path.exists():
                print(f"Error: Path not found: {csv_path}")
                return False

            # Check if folder or file
            if path.is_dir():
                count = load_folder(csv_path, db_path, replace=replace)
            else:
                count = load_csv(csv_path, db_path, replace=replace)

            print(f"✓ Loaded {count} papers")

        # Step 2: Compute embeddings
        if 2 in steps:
            print("\n[Step 2/5] Computing embeddings...")
            count = compute_embeddings(
                db_path=db_path,
                model_name=model,
                force=False
            )
            if count > 0:
                print(f"✓ Computed embeddings for {count} papers")
            else:
                print("✓ All papers already have embeddings")

        # Step 3: Cluster papers
        if 3 in steps:
            print("\n[Step 3/5] Clustering papers...")
            n_created = cluster_papers(
                db_path=db_path,
                n_clusters=n_clusters,
                min_k=min_k,
                max_k=max_k,
                random_state=random_state
            )
            print(f"✓ Created {n_created} clusters")

        # Step 4: Generate cluster descriptions
        if 4 in steps:
            print("\n[Step 4/5] Generating cluster descriptions...")
            count = update_cluster_descriptions(
                db_path=db_path,
                top_n=top_n
            )
            print(f"✓ Updated {count} cluster descriptions")

        # Step 5: Generate UMAP coordinates
        if 5 in steps:
            print("\n[Step 5/5] Generating UMAP coordinates...")
            generate_umap(
                db_path=db_path,
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                metric=metric,
                random_state=random_state
            )
            print("✓ UMAP coordinates generated")

        print("\n" + "="*80)
        print("Pipeline completed successfully!")
        print("="*80)
        return True

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return False


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Run NeurIPS papers clustering pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline on folder
  python run.py paper_list/

  # Run full pipeline on single file
  python run.py "NeurIPS 2025 Events.csv"

  # Run only steps 3, 4, and 5 (clustering + descriptions + UMAP)
  python run.py paper_list/ --steps 3 4 5

  # Run with custom parameters
  python run.py paper_list/ --n-clusters 10 --model all-mpnet-base-v2 --n-neighbors 30
        """
    )

    parser.add_argument(
        "csv",
        help="Path to input CSV file or folder containing CSV files"
    )
    parser.add_argument(
        "-d", "--db",
        default="neurips.db",
        help="Database path (default: neurips.db)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="+",
        choices=[1, 2, 3, 4, 5],
        help="Steps to run (1=load, 2=embed, 3=cluster, 4=describe, 5=umap). Default: all steps"
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace existing papers in database (step 1)"
    )
    parser.add_argument(
        "-m", "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model (default: all-MiniLM-L6-v2)"
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
        "--top-n",
        type=int,
        default=20,
        help="Number of top papers for cluster descriptions (default: 20)"
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
        help="Distance metric for UMAP (default: cosine)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    success = run_pipeline(
        csv_path=args.csv,
        db_path=args.db,
        steps=args.steps,
        replace=args.replace,
        model=args.model,
        n_clusters=args.n_clusters,
        min_k=args.min_k,
        max_k=args.max_k,
        top_n=args.top_n,
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        metric=args.metric,
        random_state=args.seed
    )

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
