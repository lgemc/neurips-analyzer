#!/usr/bin/env python3
"""
Generate semantic names and descriptions for clusters using Anthropic API.
"""
import argparse
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
from anthropic import Anthropic


def load_clustered_data(input_path: str) -> pd.DataFrame:
    """Load the clustered CSV file."""
    return pd.read_csv(input_path)


def create_text_field(df: pd.DataFrame) -> pd.Series:
    """Combine relevant text fields for embedding."""
    text_field = df['name'].fillna('') + ' ' + df['abstract'].fillna('')
    return text_field


def compute_embeddings(texts: pd.Series, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """Compute text embeddings using sentence-transformers."""
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)

    # Normalize embeddings
    embeddings = normalize(embeddings)

    return embeddings


def get_cluster_centroids(df: pd.DataFrame, embeddings: np.ndarray) -> dict:
    """
    Calculate centroids for each cluster.

    Returns:
        Dictionary mapping cluster_id -> centroid embedding
    """
    centroids = {}
    for cluster_id in df['cluster'].unique():
        cluster_mask = df['cluster'] == cluster_id
        cluster_embeddings = embeddings[cluster_mask]
        centroid = cluster_embeddings.mean(axis=0)
        # Normalize centroid
        centroid = centroid / np.linalg.norm(centroid)
        centroids[cluster_id] = centroid

    return centroids


def get_top_n_closest(df: pd.DataFrame, embeddings: np.ndarray, centroid: np.ndarray,
                      cluster_id: int, n: int = 20) -> pd.DataFrame:
    """
    Get top N items closest to the centroid for a given cluster.

    Args:
        df: DataFrame with cluster assignments
        embeddings: Normalized embeddings
        centroid: Centroid embedding for the cluster
        cluster_id: Cluster ID to filter
        n: Number of top items to return

    Returns:
        DataFrame with top N closest items
    """
    cluster_mask = df['cluster'] == cluster_id
    cluster_df = df[cluster_mask].copy()
    cluster_embeddings = embeddings[cluster_mask]

    # Calculate cosine similarity (dot product for normalized vectors)
    similarities = cluster_embeddings @ centroid

    # Get top N indices
    top_indices = np.argsort(similarities)[-n:][::-1]

    return cluster_df.iloc[top_indices]


def generate_cluster_name(client: Anthropic, top_items: pd.DataFrame, cluster_id: int) -> tuple[str, str]:
    """
    Use Anthropic API to generate cluster name and description.

    Args:
        client: Anthropic client
        top_items: DataFrame with top items from the cluster
        cluster_id: Cluster ID

    Returns:
        Tuple of (name, description)
    """
    # Create a summary of the top items
    items_text = "\n\n".join([
        f"Event {i+1}:\nName: {row['name']}\nAbstract: {row['abstract'] if pd.notna(row['abstract']) else 'No abstract'}"
        for i, (_, row) in enumerate(top_items.iterrows())
    ])

    prompt = f"""You are analyzing a cluster of NeurIPS conference events. Below are the 20 most representative events from cluster {cluster_id}:

{items_text}

Based on these events, provide:
1. A short, descriptive name for this cluster (2-5 words)
2. A brief description of what this cluster represents (1-2 sentences)

Respond in this exact format:
NAME: <short name>
DESCRIPTION: <description>"""

    print(f"Generating name for cluster {cluster_id}...")

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


def main():
    parser = argparse.ArgumentParser(
        description='Generate semantic names for clusters using Anthropic API'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to clustered CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='cluster_names.csv',
        help='Path to output CSV file (default: cluster_names.csv)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence-transformer model name (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '-n', '--top-n',
        type=int,
        default=20,
        help='Number of top items per cluster to analyze (default: 20)'
    )
    parser.add_argument(
        '--api-key-env',
        type=str,
        default='ANTHROPIC_API_KEY',
        help='Environment variable name for Anthropic API key (default: ANTHROPIC_API_KEY)'
    )

    args = parser.parse_args()

    # Get API key from environment
    api_key = os.getenv(args.api_key_env)
    if not api_key:
        print(f"Error: {args.api_key_env} environment variable not set")
        return 1

    # Initialize Anthropic client
    client = Anthropic(api_key=api_key)

    # Load clustered data
    print(f"Loading clustered data from: {args.input}")
    df = load_clustered_data(args.input)
    print(f"Loaded {len(df)} rows with {df['cluster'].nunique()} clusters")

    # Prepare text field
    texts = create_text_field(df)

    # Compute embeddings
    embeddings = compute_embeddings(texts, model_name=args.model)

    # Get centroids
    print("\nCalculating cluster centroids...")
    centroids = get_cluster_centroids(df, embeddings)

    # Generate names for each cluster
    results = []
    for cluster_id in sorted(centroids.keys()):
        centroid = centroids[cluster_id]

        # Get top N items closest to centroid
        top_items = get_top_n_closest(df, embeddings, centroid, cluster_id, n=args.top_n)

        # Generate name and description
        name, description = generate_cluster_name(client, top_items, cluster_id)

        results.append({
            'cluster_id': cluster_id,
            'name': name,
            'description': description
        })

        print(f"  Cluster {cluster_id}: {name}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(args.output, index=False)
    print(f"\nCluster names saved to: {args.output}")


if __name__ == '__main__':
    main()
