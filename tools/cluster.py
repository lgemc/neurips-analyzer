#!/usr/bin/env python3
"""
Text clustering tool for NeurIPS events using embeddings and cosine similarity.
"""
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
import sys


def load_csv(input_path: str) -> pd.DataFrame:
    """Load the input CSV file."""
    return pd.read_csv(input_path)


def create_text_field(df: pd.DataFrame) -> pd.Series:
    """Combine relevant text fields for embedding."""
    # Combine name and abstract for richer semantic representation
    text_field = df['name'].fillna('') + ' ' + df['abstract'].fillna('')
    return text_field


def compute_embeddings(texts: pd.Series, model_name: str = 'all-MiniLM-L6-v2') -> np.ndarray:
    """
    Compute text embeddings using sentence-transformers.

    Args:
        texts: Series of text to embed
        model_name: Name of the sentence-transformer model to use

    Returns:
        Array of embeddings normalized for cosine similarity
    """
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    print(f"Computing embeddings for {len(texts)} texts...")
    embeddings = model.encode(texts.tolist(), show_progress_bar=True)

    # Normalize embeddings for cosine similarity
    embeddings = normalize(embeddings)

    return embeddings


def cluster_embeddings(embeddings: np.ndarray, n_clusters: int = 5, random_state: int = 42) -> np.ndarray:
    """
    Cluster embeddings using KMeans with cosine similarity.

    Args:
        embeddings: Normalized embeddings
        n_clusters: Number of clusters
        random_state: Random seed for reproducibility

    Returns:
        Array of cluster labels
    """
    print(f"Clustering into {n_clusters} groups...")
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        max_iter=300
    )
    labels = kmeans.fit_predict(embeddings)

    return labels


def save_results(df: pd.DataFrame, labels: np.ndarray, output_path: str):
    """Save the dataframe with cluster assignments."""
    df['cluster'] = labels
    df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")

    # Print cluster distribution
    print("\nCluster distribution:")
    print(df['cluster'].value_counts().sort_index())


def main():
    parser = argparse.ArgumentParser(
        description='Cluster NeurIPS events by text similarity using embeddings'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to input CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='clustered_output.csv',
        help='Path to output CSV file (default: clustered_output.csv)'
    )
    parser.add_argument(
        '-n', '--n-clusters',
        type=int,
        default=5,
        help='Number of clusters (default: 5)'
    )
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        help='Sentence-transformer model name (default: all-MiniLM-L6-v2)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    # Load data
    print(f"Loading CSV from: {args.input}")
    df = load_csv(args.input)
    print(f"Loaded {len(df)} rows")

    # Prepare text field
    texts = create_text_field(df)

    # Compute embeddings
    embeddings = compute_embeddings(texts, model_name=args.model)

    # Cluster
    labels = cluster_embeddings(embeddings, n_clusters=args.n_clusters, random_state=args.seed)

    # Save results
    save_results(df, labels, args.output)


if __name__ == '__main__':
    main()
