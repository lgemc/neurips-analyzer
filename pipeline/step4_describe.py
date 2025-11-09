#!/usr/bin/env python3
"""
Step 4: Generate semantic names and descriptions for clusters.
"""
import sys
import os
from anthropic import Anthropic
from db import get_connection


def get_top_papers(db_path: str, cluster_id: int, n: int = 20) -> list[dict]:
    """
    Get top N papers for a cluster by score.

    Args:
        db_path: Database path
        cluster_id: Cluster ID
        n: Number of papers to retrieve

    Returns:
        List of paper dictionaries
    """
    conn = get_connection(db_path)

    cursor = conn.execute(
        """
        SELECT p.name, p.abstract, ca.score
        FROM papers p
        JOIN cluster_associations ca ON p.id = ca.paper_id
        WHERE ca.cluster_id = ?
        ORDER BY ca.score DESC
        LIMIT ?
        """,
        (cluster_id, n)
    )

    papers = [dict(row) for row in cursor.fetchall()]
    conn.close()

    return papers


def generate_cluster_description(
    client: Anthropic,
    cluster_id: int,
    papers: list[dict],
    year: int = None
) -> tuple[str, str]:
    """
    Generate cluster name and description using Anthropic API.

    Args:
        client: Anthropic client
        cluster_id: Cluster ID
        papers: List of top papers
        year: Year of the cluster (optional)

    Returns:
        Tuple of (name, description)
    """
    # Create summary of papers
    papers_text = "\n\n".join([
        f"Paper {i+1}:\nName: {paper['name']}\nAbstract: {paper['abstract'] if paper['abstract'] else 'No abstract'}"
        for i, paper in enumerate(papers)
    ])

    year_context = f" from NeurIPS {year}" if year else ""
    prompt = f"""You are analyzing a cluster of NeurIPS conference papers{year_context}. Below are the 20 most representative papers from cluster {cluster_id}:

{papers_text}

Based on these papers, provide:
1. A short, descriptive name for this cluster (2-5 words)
2. A brief description of what this cluster represents (1-2 sentences)

Respond in this exact format:
NAME: <short name>
DESCRIPTION: <description>"""

    print(f"  Generating description for cluster {cluster_id}...")

    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=300,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )

    response_text = message.content[0].text

    # Parse response
    lines = response_text.strip().split('\n')
    name = ""
    description = ""

    for line in lines:
        if line.startswith("NAME:"):
            name = line.replace("NAME:", "").strip()
        elif line.startswith("DESCRIPTION:"):
            description = line.replace("DESCRIPTION:", "").strip()

    return name, description


def update_cluster_descriptions(
    db_path: str = "neurips.db",
    api_key: str = None,
    top_n: int = 20
) -> int:
    """
    Generate and update descriptions for all clusters.

    Args:
        db_path: Database path
        api_key: Anthropic API key
        top_n: Number of top papers to analyze per cluster

    Returns:
        Number of clusters updated
    """
    # Initialize Anthropic client
    if not api_key:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

    client = Anthropic(api_key=api_key)

    # Get all cluster IDs
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT id FROM clusters ORDER BY id")
    cluster_ids = [row['id'] for row in cursor.fetchall()]
    conn.close()

    if not cluster_ids:
        print("No clusters found. Run step3_cluster.py first.")
        return 0

    print(f"Generating descriptions for {len(cluster_ids)} clusters...")

    # Process each cluster
    updated = 0
    for cluster_id in cluster_ids:
        # Get top papers
        papers = get_top_papers(db_path, cluster_id, top_n)

        if not papers:
            print(f"  Warning: No papers found for cluster {cluster_id}")
            continue

        # Get cluster year
        conn = get_connection(db_path)
        cursor = conn.execute("SELECT year FROM clusters WHERE id = ?", (cluster_id,))
        row = cursor.fetchone()
        year = row['year'] if row else None
        conn.close()

        # Generate description
        try:
            name, description = generate_cluster_description(client, cluster_id, papers, year)

            # Update database
            conn = get_connection(db_path)
            conn.execute(
                "UPDATE clusters SET name = ?, description = ? WHERE id = ?",
                (name, description, cluster_id)
            )
            conn.commit()
            conn.close()

            year_label = f" ({year})" if year else ""
            print(f"    ✓ Cluster {cluster_id}{year_label}: {name}")
            updated += 1

        except Exception as e:
            print(f"    Error processing cluster {cluster_id}: {e}")
            continue

    return updated


def main() -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Generate cluster descriptions")
    parser.add_argument(
        "-d", "--db",
        default="neurips.db",
        help="Database path (default: neurips.db)"
    )
    parser.add_argument(
        "-n", "--top-n",
        type=int,
        default=20,
        help="Number of top papers per cluster (default: 20)"
    )
    parser.add_argument(
        "--api-key",
        help="Anthropic API key (default: from ANTHROPIC_API_KEY env var)"
    )

    args = parser.parse_args()

    try:
        count = update_cluster_descriptions(
            db_path=args.db,
            api_key=args.api_key,
            top_n=args.top_n
        )
        print(f"\n✓ Successfully updated {count} clusters")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
