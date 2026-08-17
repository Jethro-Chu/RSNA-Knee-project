import pytest
import numpy as np
import pandas as pd
from scripts.train_meta_stacker import (
    TARGETS,
    simulate_expert_stream_predictions,
    fit_targetwise_nnls_stacker,
    predict_meta_stacker,
    macro_roc_auc
)

def test_meta_stacker_simplex_weights():
    # Synthetic gold dataframe
    N = 30
    rng = np.random.RandomState(42)
    synthetic_gold = pd.DataFrame({
        t: rng.randint(0, 2, size=N) for t in TARGETS
    })
    synthetic_gold['StudyInstanceUID'] = [f'study_{i}' for i in range(N)]
    synthetic_gold['split'] = ['dev'] * 20 + ['holdout'] * 10
    
    streams = simulate_expert_stream_predictions(synthetic_gold)
    dev_mask = synthetic_gold['split'].values == 'dev'
    
    weights_matrix, target_models = fit_targetwise_nnls_stacker(streams, split_mask=dev_mask)
    
    # 1. Check shape
    assert weights_matrix.shape == (len(TARGETS), 4)
    
    # 2. Check simplex constraint (all weights non-negative and sum to 1.0)
    for i, t in enumerate(TARGETS):
        w = weights_matrix[i]
        assert np.all(w >= 0.0), f"Negative weight in target {t}: {w}"
        assert np.isclose(np.sum(w), 1.0, atol=1e-5), f"Weights for {t} do not sum to 1.0: {np.sum(w)}"
        
    # 3. Predict and verify range
    preds = predict_meta_stacker(streams, weights_matrix)
    assert preds.shape == (N, len(TARGETS))
    assert np.all(preds >= 0.0) and np.all(preds <= 1.0)
    assert np.all(np.isfinite(preds))
