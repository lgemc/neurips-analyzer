#!/usr/bin/env python3
"""
Step 6: Create a web-optimized version of the NeurIPS database.

This script creates a lightweight version of the database by:
1. Removing embedding BLOBs from papers table (~43 MB saved)
2. Removing embedding BLOBs from clusters table (minimal savings)
3. Keeping all other data needed for the web viewer

Usage:
    python step6_web_db.py [--input INPUT_DB] [--output OUTPUT_DB]
"""

import sqlite3
import argparse
import os
from pathlib import Path


def create_web_database(input_db: str, output_db: str) -> tuple[float, float, float]:
    """
    Create a web-optimized database without embeddings.

    Args:
        input_db: Path to source database
        output_db: Path to output web-optimized database

    Returns:
        Tuple of (input_size_mb, output_size_mb, savings_percent)
    """
    # Remove output file if it exists
    if os.path.exists(output_db):
        os.remove(output_db)

    # Connect to source database
    source_conn = sqlite3.connect(input_db)
    source_cursor = source_conn.cursor()

    # Connect to destination database
    dest_conn = sqlite3.connect(output_db)
    dest_cursor = dest_conn.cursor()

    # Create papers table without embedding
    dest_cursor.execute("""
        CREATE TABLE papers (
            id INTEGER PRIMARY KEY,
            type TEXT,
            name TEXT,
            virtualsite_url TEXT,
            speakers_authors TEXT,
            abstract TEXT,
            year INTEGER,
            umap_x REAL,
            umap_y REAL
        )
    """)

    # Copy papers data without embedding
    source_cursor.execute("""
        SELECT id, type, name, virtualsite_url, speakers_authors, abstract, year, umap_x, umap_y
        FROM papers
    """)

    rows = source_cursor.fetchall()
    dest_cursor.executemany("""
        INSERT INTO papers (id, type, name, virtualsite_url, speakers_authors, abstract, year, umap_x, umap_y)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, rows)

    # Create clusters table without embedding
    dest_cursor.execute("""
        CREATE TABLE clusters (
            id INTEGER PRIMARY KEY,
            name TEXT,
            description TEXT,
            year INTEGER
        )
    """)

    source_cursor.execute("""
        SELECT id, name, description, year
        FROM clusters
    """)

    rows = source_cursor.fetchall()
    dest_cursor.executemany("""
        INSERT INTO clusters (id, name, description, year)
        VALUES (?, ?, ?, ?)
    """, rows)

    # Create index on clusters.year
    dest_cursor.execute("CREATE INDEX idx_cluster_year ON clusters(year)")

    # Create cluster_associations table (no changes needed)
    dest_cursor.execute("""
        CREATE TABLE cluster_associations (
            cluster_id INTEGER,
            paper_id INTEGER,
            score REAL,
            FOREIGN KEY (cluster_id) REFERENCES clusters(id),
            FOREIGN KEY (paper_id) REFERENCES papers(id),
            PRIMARY KEY (cluster_id, paper_id)
        )
    """)

    source_cursor.execute("""
        SELECT cluster_id, paper_id, score
        FROM cluster_associations
    """)

    rows = source_cursor.fetchall()
    dest_cursor.executemany("""
        INSERT INTO cluster_associations (cluster_id, paper_id, score)
        VALUES (?, ?, ?)
    """, rows)

    # Create indexes
    dest_cursor.execute("CREATE INDEX idx_paper_id ON cluster_associations(paper_id)")
    dest_cursor.execute("CREATE INDEX idx_cluster_id ON cluster_associations(cluster_id)")

    # Commit and optimize
    dest_conn.commit()
    dest_cursor.execute("VACUUM")
    dest_cursor.execute("ANALYZE")
    dest_conn.commit()

    # Close connections
    source_conn.close()
    dest_conn.close()

    # Get final file sizes
    input_size = os.path.getsize(input_db) / 1024 / 1024
    output_size = os.path.getsize(output_db) / 1024 / 1024
    savings = input_size - output_size
    savings_pct = (savings / input_size) * 100

    return input_size, output_size, savings_pct


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Create web-optimized NeurIPS database")
    parser.add_argument(
        "--input",
        default="docs/neurips.db",
        help="Input database path (default: docs/neurips.db)"
    )
    parser.add_argument(
        "--output",
        default="docs/neurips_web.db",
        help="Output database path (default: docs/neurips_web.db)"
    )

    args = parser.parse_args()

    # Validate input file exists
    if not os.path.exists(args.input):
        print(f"Error: Input database '{args.input}' not found")
        return 1

    print(f"Creating web-optimized database from {args.input}")
    print(f"Output: {args.output}")

    input_size, output_size, savings_pct = create_web_database(args.input, args.output)

    print("\n=== Results ===")
    print(f"Input database:  {input_size:.2f} MB")
    print(f"Output database: {output_size:.2f} MB")
    print(f"Space saved:     {input_size - output_size:.2f} MB ({savings_pct:.1f}%)")
    print(f"\nWeb-optimized database created: {args.output}")

    return 0


if __name__ == "__main__":
    exit(main())
