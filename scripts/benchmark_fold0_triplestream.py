"""
Fold 0 Controlled Experiment:
1. Control: Shared-Stem HMIL Model (Fold 0 baseline = 0.6413)
2. Candidate A: Triple-Stream HMIL Model (Neutral Initialization)
3. Candidate B: Triple-Stream HMIL Model (Pathology-Aware Inductive Prior Initialization)
"""

import time
import torch
import numpy as np
import pandas as pd
from rsna_knee.constants import TARGET_NAMES
from rsna_knee.models.multimodal_hmil import MultimodalHMILModel
from rsna_knee.models.triple_stream_hmil import TripleStreamHMILModel

device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"[*] Benchmark Running on Device: {device}")

# 1. Parameter counts
m_shared = MultimodalHMILModel(num_targets=12)
m_neutral = TripleStreamHMILModel(num_targets=12, use_anatomical_priors=False)
m_prior = TripleStreamHMILModel(num_targets=12, use_anatomical_priors=True)

p_shared = sum(p.numel() for p in m_shared.parameters())
p_neutral = sum(p.numel() for p in m_neutral.parameters())
p_prior = sum(p.numel() for p in m_prior.parameters())

print(f"Shared-Stem HMIL Parameters: {p_shared:,}")
print(f"Triple-Stream Neutral Parameters: {p_neutral:,}")
print(f"Triple-Stream Prior Parameters: {p_prior:,}")

# 2. Benchmark Inference Speed per study (Batch size = 1, Slices = 8 per plane, 224x224)
m_shared.to(device).eval()
m_neutral.to(device).eval()
m_prior.to(device).eval()

B = 1
S = 8
sag = torch.randn(B, S, 3, 224, 224, device=device)
cor = torch.randn(B, S, 3, 224, 224, device=device)
ax = torch.randn(B, S, 3, 224, 224, device=device)
meta = torch.randn(B, 16, device=device)

# Warmup
for _ in range(5):
    with torch.no_grad():
        _ = m_shared(sag, cor, ax, metadata=meta)
        _ = m_neutral(sag, cor, ax, metadata=meta)

# Timing Shared
t0 = time.perf_counter()
N_reps = 50
with torch.no_grad():
    for _ in range(N_reps):
        _ = m_shared(sag, cor, ax, metadata=meta)
if device.type == "cuda":
    torch.cuda.synchronize()
t_shared = (time.perf_counter() - t0) / N_reps * 1000.0 # ms

# Timing Triple-Stream
t0 = time.perf_counter()
with torch.no_grad():
    for _ in range(N_reps):
        _ = m_neutral(sag, cor, ax, metadata=meta)
if device.type == "cuda":
    torch.cuda.synchronize()
t_triple = (time.perf_counter() - t0) / N_reps * 1000.0 # ms

print(f"Inference Latency per study (Shared-Stem): {t_shared:.2f} ms")
print(f"Inference Latency per study (Triple-Stream): {t_triple:.2f} ms (Delta: {t_triple - t_shared:+.2f} ms)")
