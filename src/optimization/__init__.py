"""
Optimization Package
=====================
Energy-based optimization over the reconstruction space.

Components:
  - EnergyOptimizer: Simulated annealing / evolutionary layout search
  - WeightCalibrator: Automatic scoring weight tuning
"""

from src.optimization.energy_optimizer import EnergyOptimizer, OptimizationResult
from src.optimization.weight_calibration import WeightCalibrator, CalibrationResult

__all__ = [
    "EnergyOptimizer",
    "OptimizationResult",
    "WeightCalibrator",
    "CalibrationResult",
]
