"""
Scoring Framework for Probabilistic Crime Scene Reconstruction
===============================================================

Implements the unified multi-objective reconstruction score:

    S(R) = Σ_i  w_i · score_i(R)

Components:
    1. Semantic Alignment     — text ↔ scene match (CLIP + recall)
    2. Spatial Consistency    — constraint satisfaction
    3. Physical Plausibility  — gravity, support, scale realism
    4. Visual Realism         — image quality via CLIP aesthetics
    5. Probabilistic Prior    — Visual Genome spatial statistics
    6. Multi-View Consistency — cross-view object persistence
    7. Perceptual Believability — human realism proxy
"""

from src.scoring.unified_scorer import UnifiedScorer, ScoringWeights, ScoreBreakdown

__all__ = ["UnifiedScorer", "ScoringWeights", "ScoreBreakdown"]
