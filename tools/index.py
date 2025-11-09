#!/usr/bin/env python3
"""
Index NeurIPS 2025 Events CSV into SQLite FTS5 for full-text search.
"""

import csv
import sqlite3
import sys
from pathlib import Path


def create_index(csv_path: str, db_path: str = "neurips.db") -> None:
    """Create FTS5 index from CSV file."""
    conn = sqlite3.connect(db_path)

    # Drop existing table if it exists
    conn.execute("DROP TABLE IF EXISTS papers")

    # Create FTS5 virtual table
    conn.execute("""
        CREATE VIRTUAL TABLE papers USING fts5(
            type,
            name,
            virtualsite_url UNINDEXED,
            speakers_authors,
            abstract,
            tokenize = 'porter unicode61'
        )
    """)

    # Index the CSV
    indexed_count = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            conn.execute(
                "INSERT INTO papers VALUES (?, ?, ?, ?, ?)",
                (
                    row["type"],
                    row["name"],
                    row["virtualsite_url"],
                    row["speakers/authors"],
                    row["abstract"],
                ),
            )
            indexed_count += 1

    conn.commit()
    conn.close()

    print(f"✓ Indexed {indexed_count} papers into {db_path}")


def main() -> None:
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_file> [db_file]")
        sys.exit(1)

    csv_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 else "neurips.db"

    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    create_index(csv_path, db_path)


if __name__ == "__main__":
    main()
