"""
Experiments Package
====================
Research-grade experimental evaluation framework.

Components:
  - ExperimentRunner: Comparative evaluation across configurations
  - AblationRunner: Systematic component ablation studies
  - ResearchLogger: Structured logging for reproducible research
"""

from src.experiments.experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult
from src.experiments.ablation_runner import AblationRunner, AblationResult
from src.experiments.research_logger import ResearchLogger

__all__ = [
    "ExperimentRunner",
    "ExperimentConfig",
    "ExperimentResult",
    "AblationRunner",
    "AblationResult",
    "ResearchLogger",
]
