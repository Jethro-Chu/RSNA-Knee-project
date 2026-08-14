# Permanent Hardware & Execution Policy

**Project Policy for RSNA Knee Abnormality Detection**

---

## 1. Training Policy
- **Maximum Compute Velocity**: Always use the fastest compatible GPU(s) available (e.g. A100, H100, V100, RTX 4090, TPU VM, or multi-GPU).
- **No Artificial Training Bottlenecks**: Do NOT artificially restrict training to a Tesla T4 simply because Kaggle competition inference runs on a T4.
- **Mixed Precision**: Utilize BF16 mixed precision on modern architectures (Ampere / Ada / Hopper) or FP16 AMP (`torch.cuda.amp.autocast()`) on Volta / Turing architectures.
- **Portability**: All models must be saved as standard, clean PyTorch state dictionaries (`{"model_state_dict": model.state_dict(), ...}`), completely decoupled from hardware-specific primitives or compiled graph wrappers.
- **Strict Checkpoint Integrity**: All checkpoints must load unambiguously with `model.load_state_dict(..., strict=True)`.
- **Experiment Metadata Logging**: For every experiment in `experiments/results.csv`, record:
  - GPU Hardware & Accelerator count
  - Precision (`bf16` / `fp16` / `fp32`)
  - Batch size, learning rate, and optimizer schedule
  - PyTorch & CUDA version

---

## 2. Kaggle Inference & Deployment Policy
- **T4 / T4 x2 Compatibility**: Final competition inference must remain 100% compatible with Kaggle's offline Tesla T4 environment (`enable_internet: false`).
- **Memory Management**: If high-capacity 5-fold ensembles or large backbones exceed single-GPU VRAM during inference, execute fold models sequentially with intermediate memory freeing (`del model`, `torch.cuda.empty_cache()`) rather than compromising model capacity.
- **Sanity & Schema Gates**: Every inference pass must pass strict schema assertions, unique ID checks, non-constant prediction verification, and range bounds $[0, 1]$.
