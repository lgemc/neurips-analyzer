#!/usr/bin/env python3
"""
Search indexed NeurIPS 2025 papers using SQLite FTS5.
"""

import argparse
import json
import sqlite3
import sys
from pathlib import Path


def search(query: str, db_path: str = "neurips.db", limit: int = 10) -> list[dict]:
    """Search papers using FTS5 full-text search."""
    if not Path(db_path).exists():
        print(f"Error: Database {db_path} not found. Run index.py first.")
        sys.exit(1)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    cursor = conn.execute(
        """
        SELECT
            type,
            name,
            virtualsite_url,
            speakers_authors,
            abstract,
            rank
        FROM papers
        WHERE papers MATCH ?
        ORDER BY rank
        LIMIT ?
        """,
        (query, limit),
    )

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Search NeurIPS 2025 papers using full-text search"
    )
    parser.add_argument("query", help="Search query (FTS5 syntax)")
    parser.add_argument("-d", "--db", default="neurips.db", help="Database file")
    parser.add_argument("-n", "--limit", type=int, default=10, help="Max results")
    parser.add_argument("-j", "--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    results = search(args.query, args.db, args.limit)

    if args.json:
        json.dump(results, sys.stdout, ensure_ascii=False, indent=2)
        print()
    else:
        if not results:
            print("No results found.")
            return

        for i, paper in enumerate(results, 1):
            print(f"\n{'='*80}")
            print(f"[{i}] {paper['name']}")
            print(f"{'='*80}")
            print(f"Type: {paper['type']}")
            print(f"URL: {paper['virtualsite_url']}")
            print(f"Authors: {paper['speakers_authors']}")
            print(f"\nAbstract:\n{paper['abstract'][:300]}...")


if __name__ == "__main__":
    main()
