#!/usr/bin/env python3
"""
Step 1: Load CSV data into papers table.
"""
import csv
import sys
from pathlib import Path
from db import init_schema, get_connection


def load_csv(csv_path: str, db_path: str = "neurips.db", replace: bool = False) -> int:
    """
    Load CSV into papers table.

    Args:
        csv_path: Path to input CSV file
        db_path: Path to database file
        replace: If True, clear existing papers before loading

    Returns:
        Number of papers loaded
    """
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Initialize schema
    init_schema(db_path)

    conn = get_connection(db_path)

    # Clear existing data if requested
    if replace:
        print("Clearing existing papers...")
        conn.execute("DELETE FROM papers")
        conn.commit()

    # Load CSV
    loaded_count = 0
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for row in reader:
            conn.execute(
                """
                INSERT INTO papers (type, name, virtualsite_url, speakers_authors, abstract)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    row.get("type", ""),
                    row.get("name", ""),
                    row.get("virtualsite_url", ""),
                    row.get("speakers/authors", ""),
                    row.get("abstract", ""),
                ),
            )
            loaded_count += 1

            if loaded_count % 1000 == 0:
                print(f"  Loaded {loaded_count} papers...")

    conn.commit()
    conn.close()

    return loaded_count


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_file> [db_file] [--replace]")
        print(f"  csv_file: Path to input CSV file")
        print(f"  db_file: Database path (default: neurips.db)")
        print(f"  --replace: Clear existing papers before loading")
        return 1

    csv_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "neurips.db"
    replace = "--replace" in sys.argv

    try:
        print(f"Loading papers from {csv_path}...")
        count = load_csv(csv_path, db_path, replace=replace)
        print(f"✓ Successfully loaded {count} papers into {db_path}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
