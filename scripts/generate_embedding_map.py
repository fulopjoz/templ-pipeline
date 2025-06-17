#!/usr/bin/env python3
"""
Generate precomputed t-SNE/UMAP map for protein embedding visualization.
This script processes the protein embeddings database and creates 2D coordinates
for interactive visualization in the Streamlit app.
"""

import os
import argparse
import numpy as np
import logging
from pathlib import Path
from typing import Tuple, Dict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_embeddings(embedding_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load embeddings and PDB IDs from NPZ file."""
    try:
        data = np.load(embedding_path, allow_pickle=True)
        embeddings = data['embeddings']
        pdb_ids = data['pdb_ids']
        logger.info(f"Loaded {len(embeddings)} embeddings from {embedding_path}")
        return embeddings, pdb_ids
    except Exception as e:
        logger.error(f"Error loading embeddings: {e}")
        raise

def compute_tsne_map(embeddings: np.ndarray, perplexity: int = 30, random_state: int = 42) -> np.ndarray:
    """Compute t-SNE 2D coordinates."""
    try:
        from sklearn.manifold import TSNE
        
        # Adjust perplexity if dataset is smaller
        actual_perplexity = min(perplexity, (len(embeddings) - 1) // 3)
        logger.info(f"Computing t-SNE with perplexity={actual_perplexity} for {len(embeddings)} proteins...")
        
        tsne = TSNE(n_components=2, perplexity=actual_perplexity, random_state=random_state, 
                   init='pca', learning_rate='auto', n_jobs=-1)
        coordinates = tsne.fit_transform(embeddings)
        logger.info("t-SNE computation completed")
        return coordinates
    except ImportError:
        logger.error("scikit-learn not found. Please install: pip install scikit-learn")
        raise
    except Exception as e:
        logger.error(f"Error computing t-SNE: {e}")
        raise

def compute_umap_map(embeddings: np.ndarray, n_neighbors: int = 15, random_state: int = 42) -> np.ndarray:
    """Compute UMAP 2D coordinates."""
    try:
        import umap
        
        # Adjust n_neighbors if dataset is smaller
        actual_n_neighbors = min(n_neighbors, len(embeddings) - 1)
        logger.info(f"Computing UMAP with n_neighbors={actual_n_neighbors} for {len(embeddings)} proteins...")
        
        reducer = umap.UMAP(n_components=2, n_neighbors=actual_n_neighbors, random_state=random_state)
        coordinates = reducer.fit_transform(embeddings)
        logger.info("UMAP computation completed")
        return coordinates
    except ImportError:
        logger.warning("UMAP not found. Install with: pip install umap-learn")
        return None
    except Exception as e:
        logger.error(f"Error computing UMAP: {e}")
        return None

def save_embedding_map(output_path: str, pdb_ids: np.ndarray, tsne_coords: np.ndarray, 
                      umap_coords: np.ndarray = None, embeddings: np.ndarray = None):
    """Save the computed coordinates to NPZ file."""
    save_data = {
        'pdb_ids': pdb_ids,
        'tsne_coordinates': tsne_coords,
        'method': 'tsne_umap',
        'total_proteins': len(pdb_ids)
    }
    
    if umap_coords is not None:
        save_data['umap_coordinates'] = umap_coords
    
    # Save embeddings for similarity calculation
    if embeddings is not None:
        save_data['embeddings'] = embeddings
    
    np.savez_compressed(output_path, **save_data)
    logger.info(f"Saved embedding map to {output_path}")

def main():
    """Main function to generate embedding map."""
    parser = argparse.ArgumentParser(description="Generate embedding visualization map")
    parser.add_argument("--embedding-path", type=str, 
                       default="/home/ubuntu/mcs/templ_pipeline/data/embeddings/protein_embeddings_base.npz",
                       help="Path to embeddings NPZ file")
    parser.add_argument("--output-dir", type=str,
                       default="/home/ubuntu/mcs/templ_pipeline/data/embeddings",
                       help="Output directory for the map")
    parser.add_argument("--max-proteins", type=int, default=None,
                       help="Maximum number of proteins to include (default: all)")
    parser.add_argument("--perplexity", type=int, default=30,
                       help="t-SNE perplexity parameter")
    parser.add_argument("--n-neighbors", type=int, default=15,
                       help="UMAP n_neighbors parameter")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_path = output_dir / "embedding_map.npz"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load embeddings
        embeddings, pdb_ids = load_embeddings(args.embedding_path)
        
        # Apply optional subsampling for performance
        if args.max_proteins and len(embeddings) > args.max_proteins:
            logger.info(f"Subsampling from {len(embeddings)} to {args.max_proteins} proteins for visualization.")
            indices = np.random.choice(len(embeddings), args.max_proteins, replace=False)
            embeddings = embeddings[indices]
            pdb_ids = pdb_ids[indices]
        else:
            logger.info(f"Processing all {len(embeddings)} proteins for visualization.")
        
        # Compute t-SNE coordinates
        tsne_coords = compute_tsne_map(embeddings, perplexity=args.perplexity)
        
        # Compute UMAP coordinates (optional)
        umap_coords = compute_umap_map(embeddings, n_neighbors=args.n_neighbors)
        
        # Save the map
        save_embedding_map(str(output_path), pdb_ids, tsne_coords, umap_coords, embeddings)
        
        logger.info("Embedding map generation completed successfully!")
        logger.info(f"Generated coordinates for {len(pdb_ids)} proteins")
        if args.max_proteins:
            logger.info(f"Note: Used subset of {len(pdb_ids)}/{len(embeddings)} proteins due to --max-proteins limit")
        
    except Exception as e:
        logger.error(f"Failed to generate embedding map: {e}")
        raise

if __name__ == "__main__":
    main() 