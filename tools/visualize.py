#!/usr/bin/env python3
"""
Visualization tool for clustered NeurIPS events.
"""
import argparse
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize
import plotly.express as px
import plotly.graph_objects as go
from umap import UMAP


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


def reduce_dimensions(embeddings: np.ndarray, n_components: int = 2, random_state: int = 42) -> np.ndarray:
    """Reduce embeddings to 2D using UMAP."""
    print(f"Reducing dimensions with UMAP...")
    reducer = UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=15,
        min_dist=0.1,
        metric='cosine'
    )
    reduced = reducer.fit_transform(embeddings)
    return reduced


def create_visualization(df: pd.DataFrame, reduced_embeddings: np.ndarray, output_path: str):
    """Create interactive plotly visualization."""
    # Add reduced dimensions to dataframe
    df['x'] = reduced_embeddings[:, 0]
    df['y'] = reduced_embeddings[:, 1]

    # Create hover text with truncated abstracts
    df['hover_text'] = df.apply(
        lambda row: f"<b>{row['name']}</b><br>Cluster: {row['cluster']}<br><br>{str(row['abstract'])[:200] if pd.notna(row['abstract']) else 'No abstract'}...",
        axis=1
    )

    # Create scatter plot
    fig = px.scatter(
        df,
        x='x',
        y='y',
        color='cluster',
        hover_data={'hover_text': True, 'x': False, 'y': False, 'cluster': False},
        title='NeurIPS Events Cluster Visualization',
        labels={'x': 'UMAP Dimension 1', 'y': 'UMAP Dimension 2'},
        color_continuous_scale='viridis'
    )

    # Update hover template
    fig.update_traces(
        hovertemplate='%{customdata[0]}<extra></extra>',
        marker=dict(size=5, opacity=0.7, line=dict(width=0.5, color='white'))
    )

    # Update layout
    fig.update_layout(
        width=1200,
        height=800,
        hovermode='closest',
        font=dict(size=12)
    )

    # Save to HTML
    fig.write_html(output_path)
    print(f"\nVisualization saved to: {output_path}")
    print(f"Open the file in your browser to interact with the plot.")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize clustered NeurIPS events'
    )
    parser.add_argument(
        'input',
        type=str,
        help='Path to clustered CSV file'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='cluster_visualization.html',
        help='Path to output HTML file (default: cluster_visualization.html)'
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

    # Load clustered data
    print(f"Loading clustered data from: {args.input}")
    df = load_clustered_data(args.input)
    print(f"Loaded {len(df)} rows with {df['cluster'].nunique()} clusters")

    # Prepare text field
    texts = create_text_field(df)

    # Compute embeddings
    embeddings = compute_embeddings(texts, model_name=args.model)

    # Reduce dimensions
    reduced = reduce_dimensions(embeddings, random_state=args.seed)

    # Create visualization
    create_visualization(df, reduced, args.output)


if __name__ == '__main__':
    main()
