# CPU-Only Constraints Analysis for PhysDiffuser Transformer Model

**Date:** 2026-02-14
**Scope:** This document analyzes the hardware constraints of CPU-only deployment for the PhysDiffuser masked discrete diffusion transformer, providing a detailed memory budget, model sizing rationale, quantization strategy, inference time budget, training feasibility analysis, and framework selection with benchmarks. All design decisions are made under the assumption of a single-machine CPU environment with 16GB RAM and no GPU acceleration.

---

## 1. Memory Budget Analysis (16GB RAM)

### 1.1 Baseline Overhead

The operating system, Python interpreter, PyTorch runtime, and loaded libraries consume a non-trivial portion of available memory before any model or data is loaded. On a typical Linux system running Python 3.10+ with PyTorch 2.x:

- **OS kernel and services:** ~0.5GB
- **Python interpreter + imported libraries (NumPy, SciPy, SymPy, PyTorch):** ~1.5GB
- **Total baseline overhead:** ~2.0GB

This leaves approximately **14GB** available for model parameters, optimizer states, activations, and data buffers.

### 1.2 Model Parameter Memory

PhysDiffuser's parameters are stored in FP32 (32-bit floating point) during training, consuming 4 bytes per parameter:

- **Base model (~50M parameters):** 50M x 4 bytes = **200MB** (0.2GB)
- **With Adam optimizer states (momentum + variance, 2x model):** 50M x 8 bytes = **400MB** additional
- **Gradient buffers (1x model):** 50M x 4 bytes = **200MB** additional
- **Total model-related memory during training:** 200 + 400 + 200 = **800MB** (0.8GB)

For a hypothetical maximum model of 150M parameters (the upper bound of our budget):

- **Model weights (FP32):** 150M x 4 bytes = **600MB** (0.6GB)
- **Optimizer states (Adam):** 150M x 8 bytes = **1,200MB** (1.2GB)
- **Gradient buffers:** 150M x 4 bytes = **600MB** (0.6GB)
- **Total at 150M params:** 600 + 1,200 + 600 = **2,400MB** (2.4GB)

Both the actual 50M model and the theoretical 150M maximum fit comfortably within the available 14GB.

### 1.3 Activation Memory

Activation memory depends on batch size, sequence length, model dimension, and number of layers. For PhysDiffuser's decoder (the dominant component):

- **Configuration:** 8 layers, 8 heads, dim=256, FFN inner dim=1024
- **Sequence length:** 64 tokens (typical equation length in prefix notation)
- **Batch size:** 32

Per-layer activation storage (approximate):
- Self-attention: Q, K, V matrices + attention scores = 4 x batch x seq x dim + batch x heads x seq x seq = 4 x 32 x 64 x 256 + 32 x 8 x 64 x 64 = 2.1MB + 1.0MB = 3.1MB
- Cross-attention: similar = ~3.1MB
- FFN: batch x seq x ffn_dim = 32 x 64 x 1024 x 4 bytes = 8.4MB
- Layer norm, residuals: ~1MB
- Per-layer total: ~15.6MB

For 8 layers: 8 x 15.6MB = **~125MB**

Add the encoder (4 ISAB layers processing 200 observation points):
- ISAB layers with 32 inducing points: ~40MB total
- PMA output: negligible

**Total activation memory per batch: ~165MB (~0.16GB)**

With gradient checkpointing (trading compute for memory), this can be reduced to approximately 1/8th by only storing activations at layer boundaries, yielding ~21MB. However, at 165MB without checkpointing, this is already small enough that gradient checkpointing is unnecessary.

### 1.4 Data Buffers

- **Batch of 32 equations:** 32 x 64 tokens x 4 bytes (int32 token IDs) = 8KB (negligible)
- **Observation data:** 32 x 200 points x 10 variables x 2 bytes (FP16 encoding) = 128KB (negligible)
- **IEEE-754 multi-hot encoding:** 32 x 200 x 10 x 16 bits = 1.2MB
- **DataLoader prefetch buffers:** ~50MB (configurable)
- **Total data buffers:** ~52MB (~0.05GB)

### 1.5 LoRA Adapter Memory

During test-time adaptation, LoRA adapters are loaded alongside the base model:

- **Rank-8 LoRA on Q and V projections across 8 layers:**
  - Per projection: rank x dim + dim x rank = 8 x 256 + 256 x 8 = 4,096 parameters
  - 2 projections (Q, V) x 8 layers = 16 adapters
  - Total: 16 x 4,096 = 65,536 parameters
  - Memory: 65,536 x 4 bytes = **262KB** (~0.0003GB)
- **With optimizer states for TTA:** 65,536 x 12 bytes (params + Adam states) = **786KB** (~0.001GB)
- **Total LoRA memory: ~1MB** (negligible)

Note: The LoRA parameter count quoted in the architecture analysis (~500K) includes additional adapters on the FFN layers, which are optional. The core Q/V-only configuration uses ~66K parameters, while extending to all linear projections (Q, K, V, O, FFN up, FFN down) yields ~500K parameters.

### 1.6 Consolidated Memory Budget Table

| Component | Training (GB) | Inference (GB) | Notes |
|-----------|--------------|----------------|-------|
| OS + Python runtime | 2.0 | 2.0 | Linux + PyTorch 2.x |
| Model weights (FP32) | 0.2 | 0.2 | 50M params x 4 bytes |
| Optimizer states (Adam) | 0.4 | 0.0 | 2x model size |
| Gradient buffers | 0.2 | 0.0 | 1x model size |
| Activations | 0.16 | 0.02 | Batch=32 train, batch=1 infer |
| Data buffers + DataLoader | 0.05 | 0.01 | IEEE-754 encoded observations |
| LoRA adapters + optimizer | 0.0 | 0.001 | Only during TTA at inference |
| INT8 quantized model | 0.0 | 0.05 | Replaces FP32 weights at inference |
| **Total** | **3.01** | **2.28** | |
| **Available headroom** | **12.99** | **13.72** | Out of 16GB |

**Conclusion:** The 50M parameter PhysDiffuser model uses less than 20% of available memory during training and less than 15% during inference. Even the theoretical 150M parameter maximum would use less than 35% during training. Memory is not a binding constraint for this project.

### 1.7 Implications of Memory Headroom

The substantial memory headroom (13-14GB) enables several optimization strategies:

1. **Larger batch sizes:** Batch size can be increased to 64 or 128 without memory pressure, improving training throughput via better hardware utilization.
2. **Multiple model copies:** For ensemble inference, multiple quantized model copies could be held in memory simultaneously.
3. **Caching:** Encoder outputs for the entire validation/test set can be precomputed and cached in memory, avoiding redundant encoder forward passes during evaluation.
4. **Future expansion:** The model can grow to 150M parameters if additional capacity is needed, with training still fitting within 6GB total.

---

## 2. Target Model Size: <=150M Parameters

### 2.1 Architecture Component Breakdown

PhysDiffuser comprises four major components. The following analysis derives the parameter count for each.

#### 2.1.1 Set Transformer Encoder (~12M parameters)

The encoder maps variable-length observation sets to a fixed-dimensional latent vector using Induced Set Attention Blocks (ISABs) and Pooling by Multihead Attention (PMA).

| Sub-component | Calculation | Parameters |
|---------------|-------------|------------|
| Input projection (IEEE-754 to dim) | 160 x 256 + 256 (bias) | 41,216 |
| ISAB layers (4 layers, 32 inducing points) | 4 x 2 x MAB(256, 256, 8 heads) | 8,400,000 |
| PMA (pool to single vector) | MAB(256, 256, 8 heads) + seed vector | 530,000 |
| Output projection + layer norm | 256 x 256 + 256 + 256 + 256 | 66,048 |
| **Encoder total** | | **~9.0M** |

Each MAB (Multihead Attention Block) contains:
- Q, K, V projections: 3 x (dim x dim) = 3 x 65,536 = 196,608
- Output projection: dim x dim = 65,536
- 2x layer norm: 2 x 2 x dim = 1,024
- FFN: dim x (4 x dim) + (4 x dim) x dim = 256 x 1024 + 1024 x 256 = 524,288
- Per MAB total: ~787,456
- Each ISAB = 2 MABs = ~1,575,000
- 4 ISABs = ~6,300,000

Revised encoder total: **~9.0M parameters**

#### 2.1.2 PhysDiffuser Decoder (~35M parameters)

The masked diffusion transformer decoder generates equation tokens conditioned on the encoder latent and the current masked/unmasked token state.

| Sub-component | Calculation | Parameters |
|---------------|-------------|------------|
| Token embedding (vocab=50, dim=256) | 50 x 256 | 12,800 |
| Positional embedding (max_len=256, dim=256) | 256 x 256 | 65,536 |
| Diffusion time embedding (MLP) | 256 x 256 + 256 x 256 | 131,072 |
| Mask embedding (learnable) | 256 | 256 |
| Self-attention (8 layers, 8 heads) | 8 x 4 x (256 x 256 + 256) | 2,105,344 |
| Cross-attention (8 layers, 8 heads) | 8 x 4 x (256 x 256 + 256) | 2,105,344 |
| FFN (8 layers, inner=1024) | 8 x (256 x 1024 + 1024 + 1024 x 256 + 256) | 4,210,688 |
| Layer norms (8 layers, 3 per layer) | 8 x 3 x (256 + 256) | 12,288 |
| Output head (dim to vocab) | 256 x 50 + 50 | 12,850 |
| Soft-mask schedule MLP | 256 x 128 + 128 x 1 | 32,896 |
| **Decoder subtotal** | | **~8.7M** |

This is considerably smaller than the 35M initially estimated. To reach a more capable decoder, we can increase the architecture:

**Expanded decoder configuration (for higher capacity):**
- Layers: 12 (up from 8)
- FFN inner dimension: 2048 (up from 1024)
- Model dimension: 384 (up from 256)

| Sub-component | Expanded calculation | Parameters |
|---------------|---------------------|------------|
| Token + positional embeddings | 50 x 384 + 256 x 384 | 117,504 |
| Self-attention (12 layers) | 12 x 4 x (384 x 384 + 384) | 7,096,320 |
| Cross-attention (12 layers) | 12 x 4 x (384 x 384 + 384) | 7,096,320 |
| FFN (12 layers, inner=2048) | 12 x (384 x 2048 + 2048 + 2048 x 384 + 384) | 18,903,552 |
| Layer norms + output head | ~50,000 | 50,000 |
| **Expanded decoder total** | | **~33.3M** |

We adopt the **default (compact) decoder configuration** (8 layers, dim=256, FFN=1024) for initial development, targeting ~8.7M decoder parameters. The expanded configuration is available as a scaling option.

#### 2.1.3 Physics Priors Module (~3M parameters)

| Sub-component | Calculation | Parameters |
|---------------|-------------|------------|
| Dimensional analysis MLP | 256 x 512 + 512 x 256 + 256 x 1 | 262,400 |
| Unit embedding table (7 base units) | 7 x 256 | 1,792 |
| Operator arity constraint network | 50 x 128 + 128 x 3 | 6,784 |
| Symmetry detection heads (4 heads) | 4 x (256 x 256 + 256) | 263,168 |
| Compositionality decomposition MLP | 256 x 512 + 512 x 256 + 256 x 64 | 278,528 |
| **Physics priors total** | | **~0.8M** |

Revised estimate: The physics priors module is lighter than initially estimated at **~0.8M parameters**, since most of its functionality comes from loss computation rather than learned parameters.

#### 2.1.4 LoRA Adapters (~0.5M parameters)

| Sub-component | Calculation | Parameters |
|---------------|-------------|------------|
| Q projection adapters (8 layers) | 8 x (256 x 8 + 8 x 256) | 32,768 |
| V projection adapters (8 layers) | 8 x (256 x 8 + 8 x 256) | 32,768 |
| Optional: K, O, FFN adapters | ~400,000 | 400,000 |
| **LoRA total (Q+V only)** | | **~66K** |
| **LoRA total (all projections)** | | **~500K** |

### 2.2 Grand Total Parameter Count

| Component | Parameters | Percentage |
|-----------|-----------|------------|
| Set Transformer Encoder | 9.0M | 47.6% |
| PhysDiffuser Decoder (compact) | 8.7M | 46.0% |
| Physics Priors Module | 0.8M | 4.2% |
| LoRA Adapters (all projections) | 0.5M | 2.6% |
| **Grand Total** | **~19M** | **100%** |

With the expanded decoder: **~43.6M total** (still well under 150M).

### 2.3 Justification for Model Size

The final model size of ~19M (compact) to ~44M (expanded) parameters is deliberately conservative relative to the 150M budget. This decision is motivated by several factors:

1. **Comparative sizing in the literature:**
   - NeSymReS: 24M parameters (Biggio et al., 2021)
   - ODEFormer: 86M parameters (d'Ascoli et al., 2024)
   - E2E-Transformer: ~30M parameters (Kamienny et al., 2022)
   - Our compact model (19M) is comparable to NeSymReS; the expanded model (44M) sits between NeSymReS and ODEFormer.

2. **CPU training speed:** Smaller models train faster. At 19M parameters, a training step takes approximately 60-80ms on CPU (vs. ~325ms at 50M). This enables more rapid experimentation and hyperparameter search during development.

3. **Inference latency:** Each diffusion refinement step requires a full forward pass. With 50 refinement steps x 8 trajectories = 400 forward passes, the per-pass latency directly determines total inference time. A 19M parameter model completes a forward pass in ~25ms, yielding ~10s for refinement alone, well within the 30s budget.

4. **Diminishing returns on equation vocabulary:** The equation token vocabulary is small (~50 tokens), and the maximum sequence length is short (~64 tokens). This is fundamentally a simpler sequence modeling task than natural language, requiring less model capacity. The bottleneck for symbolic regression accuracy is typically the quality of the conditioning (encoder quality and observation coverage) rather than the decoder capacity.

5. **Headroom for future expansion:** Starting with a compact model and scaling up only if accuracy plateaus is a sound engineering practice. The 150M budget provides a 7.5x scaling factor from the compact model, which can be deployed through wider dimensions, deeper layers, or additional architectural components.

### 2.4 Scaling Strategy

If the compact model proves insufficient on complex equations (5+ variables, 10+ operators), the following scaling path is planned:

1. **Step 1 (19M -> 33M):** Increase FFN inner dimension from 1024 to 2048
2. **Step 2 (33M -> 44M):** Increase model dimension from 256 to 384
3. **Step 3 (44M -> 80M):** Increase decoder layers from 8 to 12 and encoder layers from 4 to 6
4. **Step 4 (80M -> 150M):** Increase model dimension to 512 and FFN to 2048

Each step can be evaluated independently against the Feynman benchmark to determine whether additional capacity improves performance.

---

## 3. Quantization Plan

### 3.1 Strategy: Dynamic INT8 Quantization for Inference

Training is performed in full FP32 precision, as gradient stability requires high-precision arithmetic. After training, the model is quantized to INT8 for inference deployment. This approach is standard for CPU deployment and well-supported by PyTorch.

### 3.2 Implementation

PyTorch provides `torch.quantization.quantize_dynamic`, which applies dynamic quantization to specified layer types at runtime:

```python
import torch.quantization

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},  # Quantize all linear layers
    dtype=torch.qint8
)
```

Dynamic quantization computes per-tensor scale factors at runtime based on the actual activation values, eliminating the need for a calibration dataset. This is simpler than static quantization while achieving comparable performance for transformer models.

### 3.3 Expected Benefits

| Metric | FP32 | INT8 Dynamic | Improvement |
|--------|------|-------------|-------------|
| Model file size | ~76MB (19M params) | ~19MB | 4x reduction |
| Forward pass latency | ~25ms | ~8-12ms | 2-3x speedup |
| Memory footprint | ~76MB | ~19MB | 4x reduction |
| Accuracy (R-squared) | baseline | -0.5% to -1.5% | Negligible loss |

The speedup on CPU comes from INT8 GEMM (General Matrix Multiply) operations that exploit:
- **AVX-512 VNNI instructions** (available on Intel Cascade Lake and later): native 8-bit dot product operations
- **Cache efficiency:** 4x more parameters fit in L1/L2 cache, reducing memory bandwidth bottleneck
- **Reduced memory traffic:** Loading 1 byte per parameter instead of 4 bytes

### 3.4 Accuracy Validation Protocol

After quantization, accuracy must be validated against the FP32 baseline:

1. **Run the full Feynman benchmark** with both FP32 and INT8 models
2. **Compare per-equation metrics:** exact match rate, R-squared, tree-edit distance
3. **Acceptance criterion:** less than 2% relative drop in exact match rate
4. **If accuracy drop exceeds 2%:**
   - **Option A: Selective quantization.** Skip quantization on attention QKV projections (which are more sensitive to precision) and only quantize FFN layers. This typically recovers 50-70% of the accuracy loss while retaining most of the speedup.
   - **Option B: Quantization-aware training (QAT).** Insert fake-quantization nodes during the final 5 epochs of training, allowing the model to adapt its weight distributions to the quantization grid. QAT typically reduces accuracy loss to <0.5%.
   - **Option C: INT8 with FP32 attention.** Keep attention computations in FP32 (critical for softmax precision) while quantizing only the linear projections and FFN layers.

### 3.5 Alternatives Considered and Rejected

| Alternative | Rationale for Rejection |
|-------------|------------------------|
| **FP16 (half precision)** | Not natively supported on most x86 CPUs. Intel AMX provides BF16 support on Sapphire Rapids and later, but this cannot be assumed for general deployment. FP16 on CPU without hardware support is emulated via FP32, providing no speedup. |
| **Static INT8 quantization** | Requires a representative calibration dataset and more complex implementation. Dynamic INT8 is sufficient for our model size and avoids the calibration step. |
| **ONNX Runtime INT8** | Provides potentially better optimized INT8 kernels through operator fusion, but adds a dependency (ONNX export + ORT runtime). Considered as a secondary optimization path if PyTorch dynamic quantization proves insufficient. |
| **INT4 / GPTQ-style quantization** | Aggressive quantization designed for very large models (7B+). At 19-50M parameters, INT4 provides diminishing returns and introduces significant accuracy risk. Not recommended. |
| **Pruning (structured/unstructured)** | At 19M parameters, the model is already compact. Pruning adds implementation complexity without meaningful benefit at this scale. |

### 3.6 Quantization for LoRA Adapters

LoRA adapters used during test-time adaptation (TTA) remain in FP32 during the adaptation loop, as gradient computation requires full precision. The base model can be quantized to INT8 during TTA since only the LoRA parameters receive gradients; however, the forward pass through the base model must still produce FP32 activations for the gradient computation through LoRA layers. PyTorch's dynamic quantization handles this automatically by dequantizing on the fly during the forward pass.

---

## 4. Inference Time Budget: <=30s Per Equation on Single CPU Core

### 4.1 Inference Pipeline Breakdown

The full PhysDiffuser inference pipeline for a single equation consists of six stages, executed sequentially:

#### Stage 1: IEEE-754 Observation Encoding
- **Operation:** Convert raw observation data (x, y pairs) to IEEE-754 half-precision multi-hot bit vectors
- **Complexity:** O(n_points x n_vars x 16) element-wise operations
- **Expected time:** ~10ms for 200 points x 10 variables
- **Budget allocation:** 0.1s

#### Stage 2: Set Encoder Forward Pass
- **Operation:** Process encoded observations through 4 ISAB layers and PMA to produce latent vector z
- **Complexity:** O(n_points x n_inducing x dim) for each ISAB layer
- **Expected time:** ~50ms for 200 points, 32 inducing points, dim=256
- **Budget allocation:** 0.5s
- **Note:** The encoder output z is computed once and reused across all diffusion steps and trajectories

#### Stage 3: Masked Diffusion Refinement (50 steps x 8 trajectories)
- **Operation:** Iterative soft-mask denoising through the decoder transformer
- **Per-step cost:** One full decoder forward pass = ~25ms (INT8) to ~50ms (FP32)
- **Total forward passes:** 50 steps x 8 trajectories = 400 passes
- **Expected time (INT8):** 400 x 12ms = ~4.8s
- **Expected time (FP32):** 400 x 25ms = ~10.0s
- **Budget allocation:** 12.0s

#### Stage 4: Test-Time Adaptation (32 LoRA steps)
- **Operation:** LoRA adapter training with mask-and-predict self-supervision
- **Per-step cost:** Forward pass (~25ms) + backward pass through LoRA (~50ms) + optimizer step (~5ms) = ~80ms
- **Total steps:** 32
- **Expected time:** 32 x 80ms = ~2.6s
- **Budget allocation:** 8.0s

#### Stage 5: BFGS Constant Fitting
- **Operation:** Fit placeholder constants (C) in top candidate equations using scipy.optimize.minimize (L-BFGS-B)
- **Per-candidate cost:** ~50ms for equations with 1-3 constants, ~200ms for 4+ constants
- **Number of candidates:** ~10 unique candidates after canonicalization
- **Expected time:** 10 x 100ms = ~1.0s
- **Budget allocation:** 2.0s

#### Stage 6: Candidate Selection and Scoring
- **Operation:** Compute visit counts, R-squared scores, complexity penalties; rank candidates
- **Per-candidate cost:** ~20ms (R-squared on 1000 test points + SymPy simplification)
- **Expected time:** 10 x 20ms = ~0.2s
- **Budget allocation:** 0.5s

### 4.2 Consolidated Time Budget

| Stage | Budget (s) | Expected (s) | Fraction |
|-------|-----------|-------------|----------|
| IEEE-754 encoding | 0.1 | 0.01 | 0.3% |
| Set encoder forward | 0.5 | 0.05 | 1.7% |
| Diffusion refinement (INT8) | 12.0 | 4.8 | 40.0% |
| Test-time adaptation | 8.0 | 2.6 | 26.7% |
| BFGS constant fitting | 2.0 | 1.0 | 6.7% |
| Candidate selection | 0.5 | 0.2 | 1.7% |
| **Total** | **23.1** | **8.7** | |
| **Headroom within 30s** | **6.9** | **21.3** | |

### 4.3 Sensitivity Analysis

The expected total of ~8.7s provides substantial headroom. The following scenarios stress-test the budget:

| Scenario | Diffusion (s) | TTA (s) | BFGS (s) | Total (s) | Within Budget? |
|----------|--------------|---------|----------|-----------|----------------|
| Best case (INT8, simple eq) | 3.0 | 1.5 | 0.3 | 5.0 | Yes |
| Expected case (INT8, moderate eq) | 4.8 | 2.6 | 1.0 | 8.7 | Yes |
| Worst case (FP32, complex eq) | 10.0 | 5.0 | 3.0 | 18.5 | Yes |
| Extreme (FP32, no optimizations) | 15.0 | 8.0 | 5.0 | 28.5 | Yes (barely) |
| Over budget (expanded model, FP32) | 25.0 | 12.0 | 5.0 | 42.5 | No |

The expanded model (44M params) with FP32 inference would exceed the 30s budget. This confirms that **INT8 quantization is mandatory for the expanded model** and strongly recommended even for the compact model.

### 4.4 Optimization Strategies

The following optimizations are prioritized by expected impact:

1. **INT8 dynamic quantization** (2-3x speedup on linear layers): This is the single most impactful optimization and should be applied by default for inference.

2. **Pre-compute and cache encoder output z:** The encoder forward pass is executed once, and the resulting latent vector z is reused across all 400+ decoder forward passes. This saves ~20s of redundant computation (400 x 50ms).

3. **`torch.set_num_threads(1)`:** Force single-threaded execution to avoid overhead from thread synchronization on single-core workloads. PyTorch defaults to using all available cores, which introduces thread management overhead when only one core is available.

4. **`torch.compile()` (PyTorch 2.x):** Kernel fusion and graph-level optimizations provide 20-30% speedup on transformer forward passes. The `reduce-overhead` mode minimizes Python overhead in the inference loop. Compilation happens once and is cached.

5. **Reduce diffusion steps adaptively:** For equations where the model reaches high confidence early (measured by low entropy of output logits), terminate refinement before 50 steps. This provides an adaptive speed-accuracy trade-off.

6. **Skip TTA as fallback:** If the diffusion refinement stage takes longer than 20s (approaching the budget), skip test-time adaptation entirely. TTA provides incremental improvement (5-8%) but is not essential for well-trained models on simpler equations.

7. **Batch trajectories:** Instead of running 8 trajectories sequentially, batch them into a single forward pass with batch size 8. This improves hardware utilization through better cache behavior and SIMD utilization, even on a single core.

---

## 5. Training Feasibility Analysis

### 5.1 Dataset and Training Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| Training equations | 500,000 | Generated by synthetic data generator |
| Support points per equation | 200 | Sampled uniformly in [-5, 5] |
| Maximum variables per equation | 9 | Following Feynman benchmark range |
| Batch size | 32 | Fits comfortably in memory |
| Effective batch size | 128 | Via gradient accumulation (4 mini-batches) |
| Learning rate | 1e-4 | AdamW with cosine schedule |
| Weight decay | 0.01 | Standard for transformers |
| Steps per epoch | 15,625 | 500K / 32 |
| Target epochs | 50 | With early stopping |

### 5.2 Per-Step Timing Breakdown

For the compact model (19M parameters) on a modern CPU (Intel Core i7/Xeon, AVX-512 capable):

| Operation | Time (ms) | Notes |
|-----------|----------|-------|
| Data loading + encoding | 5 | num_workers=0, precomputed cache |
| Encoder forward pass | 15 | 9M params, 200 observation points |
| Decoder forward pass | 25 | 8.7M params, 64 token sequence |
| Loss computation (CE on masked) | 2 | Cross-entropy on ~50% of positions |
| Backward pass (encoder + decoder) | 60 | ~1.5x forward (PyTorch autograd) |
| Optimizer step (AdamW) | 10 | 19M parameter updates |
| Gradient accumulation overhead | 3 | Conditional optimizer step every 4 batches |
| **Total per step** | **~120ms** | |

For the expanded model (44M parameters):

| Operation | Time (ms) | Notes |
|-----------|----------|-------|
| Data loading + encoding | 5 | Same as compact |
| Encoder forward pass | 20 | Slightly larger encoder |
| Decoder forward pass | 60 | 33M param decoder |
| Loss computation | 3 | Larger logit tensor |
| Backward pass | 130 | ~1.5x forward |
| Optimizer step | 25 | 44M parameter updates |
| **Total per step** | **~243ms** | |

### 5.3 Training Time Estimates

#### Compact Model (19M parameters)

| Metric | Value |
|--------|-------|
| Time per step | ~120ms |
| Steps per epoch | 15,625 |
| Time per epoch | 15,625 x 0.12s = **31 minutes** |
| 50 epochs | 50 x 31min = **26 hours** (~1.1 days) |
| With torch.compile (~30% speedup) | **~18 hours** |

#### Expanded Model (44M parameters)

| Metric | Value |
|--------|-------|
| Time per step | ~243ms |
| Steps per epoch | 15,625 |
| Time per epoch | 15,625 x 0.243s = **63 minutes** |
| 50 epochs | 50 x 63min = **53 hours** (~2.2 days) |
| With torch.compile (~30% speedup) | **~37 hours** (~1.5 days) |

### 5.4 Progressive Training Strategy

To maximize iteration speed during development, training is organized in phases:

#### Phase A: Rapid Prototyping (2-4 hours)
- **Compact model** on **10K equations** (2% of full dataset)
- Steps per epoch: 312
- Time per epoch: ~37 seconds
- Run 50 epochs in ~31 minutes
- Purpose: Validate loss convergence, debug training loop, tune hyperparameters

#### Phase B: Intermediate Validation (6-8 hours)
- **Compact model** on **100K equations** (20% of full dataset)
- Steps per epoch: 3,125
- Time per epoch: ~6.3 minutes
- Run 50 epochs in ~5.2 hours
- Purpose: Evaluate Feynman benchmark performance, identify capacity bottlenecks

#### Phase C: Full Training (18-37 hours)
- **Compact or expanded model** on **500K equations** (full dataset)
- Run with early stopping (patience=5 epochs on validation loss)
- Purpose: Final model for benchmark evaluation
- Practical approach: Start Friday evening, results available Monday morning

### 5.5 Curriculum Learning

Training equations are ordered by complexity to improve convergence:

1. **Epochs 1-10:** Trivial equations only (1-2 variables, <=3 operators). The model learns basic operator semantics and the prefix notation structure.
2. **Epochs 11-25:** Simple and moderate equations added (up to 5 variables, <=8 operators). The model learns to compose operators and handle multi-variable relationships.
3. **Epochs 26-50:** Full dataset including complex and multi-step equations. The model refines its handling of deep expression trees and compositional structure.

This curriculum reduces training time by 10-20% compared to random ordering, as measured in prior symbolic regression work (ODEFormer reports similar benefits).

### 5.6 Training Optimizations for CPU

1. **`torch.compile(mode="reduce-overhead")`:** Fuses operations, eliminates Python overhead, and optimizes memory access patterns. Expected 20-30% training speedup.

2. **`num_workers=0` in DataLoader:** On single-CPU systems, spawning worker processes for data loading introduces multiprocessing overhead (IPC, context switching) that exceeds the benefit of parallel data preparation. Synchronous data loading with preprocessing cached to disk is faster.

3. **Gradient accumulation:** Effective batch size of 128 (4 accumulation steps x batch 32) provides more stable gradients without increasing peak memory. The optimizer step is executed only every 4 mini-batches.

4. **Mixed precision NOT used:** Unlike GPU training where FP16 provides 2x throughput, CPU FP16 operations are emulated via FP32 on most x86 hardware, providing no speedup. All training is in FP32.

5. **Pin memory disabled:** `pin_memory=True` is a GPU optimization (for faster CPU-to-GPU transfer) and provides no benefit for CPU-only training. It is disabled to avoid unnecessary memory allocation.

6. **Precomputed data caching:** IEEE-754 multi-hot encodings are precomputed and stored on disk (approximately 15GB for 500K equations at 200 points each). During training, encoded data is memory-mapped, avoiding redundant encoding computation.

---

## 6. Framework Selection

### 6.1 Decision: PyTorch 2.x with torch.compile

After evaluating the major deep learning frameworks for CPU-only transformer training and inference, **PyTorch 2.x** is selected as the primary framework.

### 6.2 Framework Comparison

| Criterion | PyTorch 2.x | TensorFlow 2.x | JAX | ONNX Runtime | LibTorch (C++) |
|-----------|-------------|-----------------|-----|-------------|----------------|
| **CPU training support** | Excellent | Good | Good | None (inference only) | Good |
| **torch.compile / JIT** | Native, mature | tf.function (XLA) | jax.jit (XLA) | N/A | TorchScript |
| **Dynamic computation graphs** | Native | tf.function limits | Functional transforms | Static only | Static only |
| **LoRA integration** | Native (via PEFT library) | Manual implementation | Manual implementation | Not supported | Manual |
| **Dynamic INT8 quantization** | Built-in | TF Lite (mobile-focused) | Not built-in | Built-in, optimized | Not built-in |
| **SymPy integration** | Seamless (Python) | Seamless (Python) | Seamless (Python) | Requires Python wrapper | C++ / Python bridge |
| **Ecosystem (SR-specific)** | NeSymReS, ODEFormer, TPSR all use PyTorch | Limited SR implementations | Limited SR implementations | N/A | N/A |
| **Development velocity** | High | Moderate | Moderate | Low (export step) | Low (C++ development) |
| **Community/debugging** | Largest community | Large community | Growing community | Smaller community | Small community |

### 6.3 Benchmark Estimates

The following benchmarks are estimated based on published results for similar-scale transformer models (~20M parameters) on CPU:

#### Training Step (Forward + Backward + Optimizer)

| Framework | Time (ms) | Relative | Notes |
|-----------|----------|----------|-------|
| PyTorch eager mode | 150 | 1.0x | Baseline |
| PyTorch + torch.compile | 105 | 1.4x | 30% speedup from kernel fusion |
| TensorFlow + tf.function | 130 | 1.15x | XLA optimization |
| JAX + jax.jit | 110 | 1.36x | XLA optimization |

#### Inference Forward Pass (Batch=1, Sequence=64)

| Framework | Time (ms) | Relative | Notes |
|-----------|----------|----------|-------|
| PyTorch eager (FP32) | 25 | 1.0x | Baseline |
| PyTorch + compile (FP32) | 18 | 1.4x | Kernel fusion |
| PyTorch dynamic INT8 | 10 | 2.5x | INT8 GEMM on AVX-512 |
| ONNX Runtime (FP32) | 15 | 1.7x | Graph-level optimization |
| ONNX Runtime (INT8 static) | 6 | 4.2x | Fully optimized INT8 graph |

ONNX Runtime achieves the best raw inference performance due to its graph-level operator fusion and optimized INT8 kernels. However, this comes at the cost of:
- An additional export step (PyTorch to ONNX)
- Loss of dynamic graph capabilities needed for variable-length sequences and TTA
- Inability to perform test-time LoRA adaptation (no gradient computation)

### 6.4 Rationale for PyTorch 2.x

1. **Development ecosystem alignment:** All major prior work in transformer-based symbolic regression (NeSymReS, ODEFormer, TPSR, E2E-Transformer) is implemented in PyTorch. Building on PyTorch enables direct reuse of reference implementations, pretrained components, and comparison code.

2. **torch.compile provides competitive CPU performance:** With torch.compile, PyTorch closes most of the performance gap with specialized inference engines like ONNX Runtime, while retaining full flexibility for dynamic computation graphs, gradient computation (needed for TTA), and interactive debugging.

3. **Native LoRA support:** The Hugging Face PEFT library and PyTorch's built-in parameter group mechanisms make LoRA implementation straightforward. Test-time adaptation requires gradient computation through LoRA parameters while keeping the base model frozen, which is natively supported by PyTorch's autograd system.

4. **Dynamic quantization simplicity:** PyTorch's `quantize_dynamic` is a single function call that quantizes all linear layers to INT8 without requiring a calibration dataset or export step. This is the simplest path to quantized inference.

5. **SymPy and SciPy integration:** PhysDiffuser's pipeline involves SymPy (for equation canonicalization, simplification, and symbolic equivalence checking) and SciPy (for BFGS constant fitting). Both integrate seamlessly with PyTorch tensors through NumPy conversion. No framework boundary issues.

6. **Debugging and profiling:** PyTorch's eager mode allows standard Python debugging (breakpoints, print statements, pdb). The `torch.profiler` module provides detailed CPU performance profiling including per-operator timing, memory allocation tracking, and call stack analysis. This is critical during development and optimization.

### 6.5 ONNX Runtime as Secondary Optimization

While PyTorch is the primary framework for development and training, ONNX Runtime is identified as a secondary optimization path for inference-only deployment:

1. **When to consider:** If INT8 PyTorch inference exceeds the 30s budget on the expanded model
2. **Implementation:** Export the trained PyTorch model to ONNX format, apply ONNX Runtime's static INT8 quantization with a calibration dataset (100 equations), and run inference through the ORT C++ runtime
3. **Expected benefit:** Additional 1.5-2x speedup over PyTorch INT8, bringing the expanded model within budget
4. **Limitation:** TTA (test-time adaptation) cannot run in ONNX Runtime, so a hybrid pipeline would be needed: ONNX Runtime for the diffusion refinement steps (the bottleneck), PyTorch for TTA

This hybrid approach is deferred unless performance profiling reveals it is necessary.

### 6.6 Hardware-Specific Considerations

| CPU Feature | Impact on PhysDiffuser | Detection |
|-------------|----------------------|-----------|
| **AVX-512** | 2x throughput for FP32 GEMM vs AVX2 | `lscpu \| grep avx512` |
| **AVX-512 VNNI** | Native INT8 dot products, critical for INT8 quantization speedup | `lscpu \| grep vnni` |
| **Intel AMX (BF16)** | Native BF16 matrix multiply on Sapphire Rapids+; potential 2x over FP32 | `lscpu \| grep amx` |
| **L3 cache size** | Larger cache benefits transformer attention computation; 19M params (76MB FP32) may not fit entirely in L3 | `lscpu \| grep "L3 cache"` |
| **NUMA topology** | Single-socket deployment assumed; multi-socket adds NUMA complexity | `numactl --hardware` |

The training and inference scripts should detect available CPU features at startup and log them, enabling performance diagnosis and reproducibility.

---

## 7. Summary and Recommendations

### 7.1 Key Design Decisions

| Constraint | Decision | Rationale |
|-----------|----------|-----------|
| **Memory (16GB)** | 19M param compact model, ~3GB peak training | Uses <20% of available memory; headroom for batch size scaling |
| **Model size (<=150M)** | Start at 19M, scale to 44M if needed | Comparable to NeSymReS (24M); faster iteration; 150M headroom preserved |
| **Quantization** | Dynamic INT8 for inference, FP32 for training | 2-3x inference speedup; <1% accuracy loss; single function call |
| **Inference (<=30s)** | Expected ~8.7s (INT8), worst case ~18.5s (FP32) | Well within budget with substantial headroom |
| **Training** | ~18 hours (compact, 50 epochs, torch.compile) | Feasible for overnight runs; progressive training for rapid iteration |
| **Framework** | PyTorch 2.x with torch.compile | Best ecosystem alignment; competitive CPU performance; native LoRA + quantization |

### 7.2 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| INT8 accuracy drop >2% | Low | Medium | Selective quantization or QAT (Section 3.4) |
| Inference exceeds 30s on complex equations | Low | High | Adaptive diffusion steps, skip TTA, ONNX Runtime fallback |
| Training exceeds 3 days | Low | Medium | Reduce to 20 epochs with early stopping; use 100K subset |
| Memory pressure from large batches | Very Low | Low | Reduce batch size; gradient checkpointing available |
| CPU lacks AVX-512 / VNNI | Medium | Medium | Fall back to FP32 inference (still within budget for compact model) |

### 7.3 Validation Checkpoints

The following checkpoints validate that CPU constraints are met throughout development:

1. **After encoder implementation:** Verify encoder forward pass completes in <100ms on 200 observation points
2. **After decoder implementation:** Verify decoder forward pass completes in <50ms on 64-token sequence
3. **After training loop implementation:** Verify per-step time is within 150ms (eager) or 120ms (compiled)
4. **After quantization:** Verify INT8 model accuracy within 2% of FP32 on validation set
5. **After full pipeline integration:** Verify end-to-end inference completes within 30s on all 120 Feynman benchmark equations
6. **After TTA implementation:** Verify 32 LoRA adaptation steps complete within 10s

These checkpoints are integrated into the CI/test pipeline as performance regression tests.
