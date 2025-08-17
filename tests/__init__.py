# SPDX-FileCopyrightText: 2025 TEMPL Team
# SPDX-License-Identifier: MIT
"""
Test package for the TEMPL Pipeline.

This package contains tests for all modules of the TEMPL Pipeline.
"""

import os


def get_test_data_path(data_type, pdb_id=None):
    """
    Helper function to resolve test data paths.

    Args:
        data_type: Type of data (embeddings, pdbbind_other, pdbbind_refined, splits)
        pdb_id: Optional PDB ID for specific file

    Returns:
        Path to the requested data
    """
    # Base paths that might be used
    base_paths = {
        "embeddings": [
            "data/embeddings/templ_protein_embeddings_v1.0.0.npz",
            "templ_pipeline/data/embeddings/templ_protein_embeddings_v1.0.0.npz",
            "../data/embeddings/templ_protein_embeddings_v1.0.0.npz",
            # Legacy fallbacks for existing test environments
            "data/embeddings/protein_embeddings_base.npz",
            "templ_pipeline/data/embeddings/protein_embeddings_base.npz",
            "/home/ubuntu/mcs/mcs_bench/data/protein_embeddings_base.npz",
            "/home/ubuntu/mcs/templ_pipeline/data/embeddings/protein_embeddings_base.npz",
        ],
        "pdbbind_other": [
            "data/PDBbind_v2020_other_PL/v2020-other-PL",
            "mcs_bench/data/PDBbind_v2020_other_PL/v2020-other-PL",
            "/home/ubuntu/mcs/mcs_bench/data/PDBbind_v2020_other_PL/v2020-other-PL",
        ],
        "pdbbind_refined": [
            "data/PDBbind_v2020_refined/refined-set",
            "mcs_bench/data/PDBbind_v2020_refined/refined-set",
            "/home/ubuntu/mcs/mcs_bench/data/PDBbind_v2020_refined/refined-set",
        ],
        "splits": [
            "data/splits",
            "templ_pipeline/data/splits",
            "../data/splits",
            "mcs_bench/data/time_splits",
            "/home/ubuntu/mcs/mcs_bench/data/time_splits",
        ],
    }

    # Try all paths
    paths = base_paths.get(data_type, [])
    for path in paths:
        if pdb_id:
            full_path = os.path.join(path, pdb_id)
            if os.path.exists(full_path):
                return full_path
        elif os.path.exists(path):
            return path

    # Return the first path as default, caller should check existence
    return paths[0] if paths else None
