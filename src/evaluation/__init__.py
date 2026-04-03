"""
Evaluation module for the neuro-symbolic arithmetic system.

Provides:
  - Per-class accuracy (digits and operators)
  - Expression-level accuracy
  - Result-level accuracy
  - Abduction rate and effectiveness
  - ECE calibration metrics
  - Confusion matrices
  - Ablation study framework (NGA vs WMC vs Pure Neural)
"""

from src.evaluation.metrics import (
    PerClassAccuracy,
    ExpressionAccuracy,
    ResultAccuracy,
    AbductionTracker,
    CalibrationMetrics,
    ConfusionMatrix,
    EvaluationSuite,
)

from src.evaluation.ablation_studies import (
    AblationConfig,
    AblationRunner,
    ABLATION_CONFIGS,
)
