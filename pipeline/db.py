#!/usr/bin/env python3
"""
Shared database utilities and schema for NeurIPS pipeline.
"""
import sqlite3
import numpy as np
from typing import Optional


def get_connection(db_path: str = "neurips.db") -> sqlite3.Connection:
    """Get database connection."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_schema(db_path: str = "neurips.db") -> None:
    """Initialize database schema."""
    conn = get_connection(db_path)

    # Papers table - main table with all CSV columns + embedding
    conn.execute("""
        CREATE TABLE IF NOT EXISTS papers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT,
            name TEXT,
            virtualsite_url TEXT,
            speakers_authors TEXT,
            abstract TEXT,
            embedding BLOB
        )
    """)

    # Clusters table
    conn.execute("""
        CREATE TABLE IF NOT EXISTS clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            description TEXT
        )
    """)

    # Cluster associations - soft clustering with scores
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cluster_associations (
            cluster_id INTEGER,
            paper_id INTEGER,
            score REAL,
            FOREIGN KEY (cluster_id) REFERENCES clusters(id),
            FOREIGN KEY (paper_id) REFERENCES papers(id),
            PRIMARY KEY (cluster_id, paper_id)
        )
    """)

    # Create indices for faster queries
    conn.execute("CREATE INDEX IF NOT EXISTS idx_paper_id ON cluster_associations(paper_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_cluster_id ON cluster_associations(cluster_id)")

    conn.commit()
    conn.close()


def serialize_embedding(embedding: np.ndarray) -> bytes:
    """Convert numpy array to bytes for storage."""
    return embedding.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes, dim: int) -> np.ndarray:
    """Convert bytes back to numpy array."""
    return np.frombuffer(blob, dtype=np.float32).reshape(dim)


def clear_embeddings(db_path: str = "neurips.db") -> None:
    """Clear all embeddings from papers table."""
    conn = get_connection(db_path)
    conn.execute("UPDATE papers SET embedding = NULL")
    conn.commit()
    conn.close()


def clear_clusters(db_path: str = "neurips.db") -> None:
    """Clear clusters and associations."""
    conn = get_connection(db_path)
    conn.execute("DELETE FROM cluster_associations")
    conn.execute("DELETE FROM clusters")
    conn.commit()
    conn.close()


def get_papers_count(db_path: str = "neurips.db") -> int:
    """Get total number of papers."""
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM papers")
    count = cursor.fetchone()[0]
    conn.close()
    return count


def get_embeddings_count(db_path: str = "neurips.db") -> int:
    """Get number of papers with embeddings."""
    conn = get_connection(db_path)
    cursor = conn.execute("SELECT COUNT(*) FROM papers WHERE embedding IS NOT NULL")
    count = cursor.fetchone()[0]
    conn.close()
    return count
