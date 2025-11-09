#!/usr/bin/env python3
"""
Step 2: Compute and store embeddings for papers.
"""
import sys
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from db import get_connection, serialize_embedding, get_papers_count, get_embeddings_count


def compute_embeddings(
    db_path: str = "neurips.db",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 32,
    force: bool = False
) -> int:
    """
    Compute embeddings for all papers and store in database.

    Args:
        db_path: Path to database
        model_name: Sentence-transformer model name
        batch_size: Batch size for encoding
        force: If True, recompute embeddings even if they exist

    Returns:
        Number of papers processed
    """
    conn = get_connection(db_path)

    # Check if embeddings already exist
    if not force:
        existing = get_embeddings_count(db_path)
        total = get_papers_count(db_path)
        if existing > 0:
            print(f"Found {existing}/{total} papers with embeddings.")
            print("Use --force to recompute embeddings.")
            if existing == total:
                print("All papers already have embeddings. Skipping.")
                return 0

    # Load model
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # Fetch papers without embeddings (or all if force=True)
    if force:
        query = "SELECT id, name, abstract FROM papers"
    else:
        query = "SELECT id, name, abstract FROM papers WHERE embedding IS NULL"

    cursor = conn.execute(query)
    papers = cursor.fetchall()

    if not papers:
        print("No papers to process.")
        return 0

    print(f"Processing {len(papers)} papers...")

    # Prepare texts: combine name + abstract
    texts = []
    paper_ids = []
    for paper in papers:
        text = f"{paper['name'] or ''} {paper['abstract'] or ''}"
        texts.append(text.strip())
        paper_ids.append(paper['id'])

    # Compute embeddings in batches
    print("Computing embeddings...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True
    )

    # Normalize for cosine similarity
    embeddings = normalize(embeddings)

    # Store embeddings
    print("Storing embeddings in database...")
    for paper_id, embedding in zip(paper_ids, embeddings):
        conn.execute(
            "UPDATE papers SET embedding = ? WHERE id = ?",
            (serialize_embedding(embedding), paper_id)
        )

        if paper_id % 1000 == 0:
            conn.commit()

    conn.commit()
    conn.close()

    return len(papers)


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compute embeddings for papers")
    parser.add_argument(
        "-d", "--db",
        default="neurips.db",
        help="Database path (default: neurips.db)"
    )
    parser.add_argument(
        "-m", "--model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformer model (default: all-MiniLM-L6-v2)"
    )
    parser.add_argument(
        "-b", "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute embeddings even if they exist"
    )

    args = parser.parse_args()

    try:
        count = compute_embeddings(
            db_path=args.db,
            model_name=args.model,
            batch_size=args.batch_size,
            force=args.force
        )
        if count > 0:
            print(f"✓ Successfully computed embeddings for {count} papers")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
