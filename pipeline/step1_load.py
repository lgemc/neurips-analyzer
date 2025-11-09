#!/usr/bin/env python3
"""
Step 1: Load CSV data into papers table.
"""
import csv
import sys
import re
from pathlib import Path
from db import init_schema, get_connection


def extract_year_from_filename(filename: str) -> int:
    """
    Extract year from filename.
    Supports: 'NeurIPS YYYY Events.csv' and 'NIPS YYYY Events.csv'

    Args:
        filename: CSV filename

    Returns:
        Year as integer
    """
    match = re.search(r'(NeurIPS|NIPS)\s+(\d{4})', filename)
    if match:
        return int(match.group(2))
    raise ValueError(f"Could not extract year from filename: {filename}")


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

    # Extract year from filename
    year = extract_year_from_filename(Path(csv_path).name)
    print(f"  Detected year: {year}")

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
                INSERT INTO papers (type, name, virtualsite_url, speakers_authors, abstract, year)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    row.get("type", ""),
                    row.get("name", ""),
                    row.get("virtualsite_url", ""),
                    row.get("speakers/authors", ""),
                    row.get("abstract", ""),
                    year,
                ),
            )
            loaded_count += 1

            if loaded_count % 1000 == 0:
                print(f"  Loaded {loaded_count} papers...")

    conn.commit()
    conn.close()

    return loaded_count


def load_folder(folder_path: str, db_path: str = "neurips.db", replace: bool = False) -> int:
    """
    Load all CSV files from a folder.

    Args:
        folder_path: Path to folder containing CSV files
        db_path: Path to database file
        replace: If True, clear existing papers before loading

    Returns:
        Total number of papers loaded
    """
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise ValueError(f"Folder not found: {folder_path}")

    # Find all CSV files
    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in {folder_path}")

    print(f"Found {len(csv_files)} CSV file(s)")

    # Initialize schema
    init_schema(db_path)

    # Clear if requested (only once)
    if replace:
        print("Clearing existing papers...")
        conn = get_connection(db_path)
        conn.execute("DELETE FROM papers")
        conn.commit()
        conn.close()

    total_count = 0
    for csv_file in csv_files:
        print(f"\nLoading {csv_file.name}...")
        count = load_csv(str(csv_file), db_path, replace=False)
        print(f"✓ Loaded {count} papers from {csv_file.name}")
        total_count += count

    return total_count


def main() -> int:
    """CLI entry point."""
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <csv_file_or_folder> [db_file] [--replace]")
        print(f"  csv_file_or_folder: Path to CSV file or folder containing CSVs")
        print(f"  db_file: Database path (default: neurips.db)")
        print(f"  --replace: Clear existing papers before loading")
        return 1

    input_path = sys.argv[1]
    db_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "neurips.db"
    replace = "--replace" in sys.argv

    try:
        path = Path(input_path)

        # Check if it's a folder or file
        if path.is_dir():
            print(f"Loading papers from folder {input_path}...")
            count = load_folder(input_path, db_path, replace=replace)
            print(f"\n✓ Successfully loaded {count} total papers into {db_path}")
        else:
            print(f"Loading papers from {input_path}...")
            count = load_csv(input_path, db_path, replace=replace)
            print(f"✓ Successfully loaded {count} papers into {db_path}")

        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
