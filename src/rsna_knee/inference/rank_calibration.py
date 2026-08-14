"""
Target-Specific Percentile Rank Calibration and Temperature Scaling for Multi-Label ROC-AUC.
"""

from typing import Dict, List, Optional
import numpy as np
import scipy.stats as stats


def calibrate_probabilities_by_target(
    probs: np.ndarray,
    target_temperatures: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Applies target-specific temperature scaling to soften or sharpen probability distributions
    per target abnormality while preserving rank ordering.
    """
    if target_temperatures is None:
        return probs

    calibrated = np.copy(probs)
    for i, t in enumerate(target_temperatures):
        temp = target_temperatures[t]
        if temp > 0:
            # Shift logit
            eps = 1e-7
            p = np.clip(calibrated[:, i], eps, 1.0 - eps)
            logits = np.log(p / (1.0 - p))
            scaled_logits = logits / temp
            calibrated[:, i] = 1.0 / (1.0 + np.exp(-scaled_logits))

    return calibrated
