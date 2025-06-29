"""Service modules for TEMPL Pipeline"""

from .pipeline_service import run_pipeline
from .protein_similarity_service import ProteinSimilarityService

__all__ = ['run_pipeline', 'ProteinSimilarityService']
