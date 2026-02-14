# PhysMDT Architecture Design

## 1. Pipeline Overview (Block Diagram)

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          PhysMDT Pipeline                              │
│                                                                         │
│  ┌──────────┐    ┌───────────┐    ┌──────────────┐    ┌─────────────┐  │
│  │  Input    │    │  Set      │    │  Masked      │    │  Symbolic   │  │
│  │  Data     │───▶│  Encoder  │───▶│  Diffusion   │───▶│  Output     │  │
│  │  Matrix   │    │  (DeepSets│    │  Transformer │    │  (RPN       │  │
│  │  {x_i,y_i}│    │  +MHA)    │    │  (Bidirect.) │    │   tokens)   │  │
│  └──────────┘    └───────────┘    └──────────────┘    └─────────────┘  │
│                        │                │                      │        │
│                        │          ┌─────┴──────┐              │        │
│                        │          │ Tree-Aware  │              │        │
│                        │          │ 2D RoPE     │              │        │
│                        │          └────────────┘              │        │
│                        │                                       │        │
│                  ┌─────┴──────────────────────┐               │        │
│                  │  Soft-Masking Recursion     │               │        │
│                  │  (50 refinement steps)      │──────────────▶│        │
│                  └────────────────────────────┘               │        │
│                                                                │        │
│                  ┌────────────────────────────┐               │        │
│                  │  Test-Time Finetuning       │               │        │
│                  │  (LoRA, 128 steps/eq)       │               │        │
│                  └────────────────────────────┘               │        │
│                                                                         │
│  ┌──────────┐    ┌───────────┐                    ┌─────────────┐      │
│  │  RPN     │    │  Tokenizer│                    │  SymPy      │      │
│  │  Token   │◀──│  (encode/ │                    │  Decode &   │      │
│  │  Sequence│    │   decode) │                    │  Simplify   │      │
│  └──────────┘    └───────────┘                    └─────────────┘      │
└─────────────────────────────────────────────────────────────────────────┘
```

## 2. Model Dimensions

### Base Model (PhysMDT-Base, ~45M parameters)

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Embedding** | d_model | 512 |
| | Vocabulary size | 128 |
| **Set Encoder** | Layers | 4 |
| | Attention heads | 8 |
| | FFN dim | 2048 |
| | Input dim | d_vars + 1 (variables + target) |
| | Output dim | 512 |
| **Diffusion Transformer** | Layers | 8 |
| | Attention heads | 8 |
| | FFN dim | 2048 |
| | Max sequence length | 64 tokens |
| | Dropout | 0.1 |
| **Total parameters** | | ~45M |

### Scaled Model (PhysMDT-Scaled, ~180M parameters)

| Component | Parameter | Value |
|-----------|-----------|-------|
| **Embedding** | d_model | 1024 |
| | Vocabulary size | 128 |
| **Set Encoder** | Layers | 6 |
| | Attention heads | 16 |
| | FFN dim | 4096 |
| | Output dim | 1024 |
| **Diffusion Transformer** | Layers | 16 |
| | Attention heads | 16 |
| | FFN dim | 4096 |
| | Max sequence length | 64 tokens |
| | Dropout | 0.1 |
| **Total parameters** | | ~180M |

### Parameter Count Breakdown (Base)

- Embedding layer: 128 × 512 = 65K
- Set Encoder (4 layers × (4 × 512² + 512 × 2048 × 2)): ~17M
- Diffusion Transformer (8 layers × (4 × 512² + 512 × 2048 × 2)): ~34M
- Output projection: 512 × 128 = 65K
- Positional encoding: ~100K
- **Total: ~45M (within ≤50M budget)**

## 3. Tokenization Scheme

### Reverse Polish Notation (RPN)

Expressions are converted to RPN for unambiguous tree-to-sequence mapping:
- `(a + b) * c` → `a b + c *`
- `sin(x * y)` → `x y * sin`

### Vocabulary (128 tokens)

| Token Range | Category | Examples |
|-------------|----------|----------|
| 0 | PAD | Padding |
| 1 | BOS | Begin of sequence |
| 2 | EOS | End of sequence |
| 3 | MASK | Mask token for diffusion |
| 4-13 | Operators | +, -, *, /, ^, sqrt, sin, cos, tan, log |
| 14-15 | More operators | exp, abs |
| 16-24 | Variables | x1, x2, ..., x9 |
| 25-26 | Constants | π, e |
| 27-126 | Numeric tokens | Integers -49..49 + decimal tokens |
| 127 | UNK | Unknown token |

## 4. Tree-Aware 2D Positional Encoding

### Mathematical Formulation

For an RPN token sequence s = (s_1, ..., s_L), we first reconstruct the implicit expression tree T by simulating the RPN stack:

1. **Tree reconstruction**: Process tokens left-to-right. Operands push nodes; operators pop children and create parent nodes.
2. **Position assignment**: Each token s_j is assigned a 2D position (d_j, h_j) where:
   - d_j = depth of the corresponding node in the tree (root = 0)
   - h_j = horizontal index at that depth level (left-to-right ordering)

### Rotary Positional Embedding (2D Golden Gate RoPE)

Following Su et al. [su2021rope] and the Golden Gate adaptation [architects2025arc], we split the embedding dimension into K directional components:

For a token at position (d, h), the rotation matrix is:

```
R(d, h) = ⊗_{k=1}^{K} R_k(d, h)
```

where each directional component k applies rotation:

```
R_k(d, h) = Rot(θ_k^{(d)} · d + θ_k^{(h)} · h)
```

with K = 4 directional frequency bases (capturing depth, horizontal, diagonal, and anti-diagonal relationships):

```
θ_1 = (1, 0)      # Pure depth direction
θ_2 = (0, 1)      # Pure horizontal direction
θ_3 = (1, 1)      # Diagonal (depth + horizontal)
θ_4 = (1, -1)     # Anti-diagonal (depth - horizontal)
```

The rotation for dimension pair (2i, 2i+1) within direction k:

```
Rot_k(pos) = [[cos(pos/10000^{2i/d_k}), -sin(pos/10000^{2i/d_k})],
               [sin(pos/10000^{2i/d_k}),  cos(pos/10000^{2i/d_k})]]
```

where d_k = d_model / K is the dimension per direction.

The embedding dimensions are split into 4 equal groups of d_model/4, each handling one direction. This allows attention to capture relationships along tree depth, sibling order, and diagonal patterns simultaneously.

### Integration

The 2D RoPE is applied to query and key vectors in each attention head:
```
q_j' = R(d_j, h_j) · q_j
k_j' = R(d_j, h_j) · k_j
```

This replaces standard sinusoidal positional encoding and can be toggled via config for ablation.

## 5. Soft-Masking Recursion Inference Loop

### Pseudocode

```python
def soft_masking_inference(model, data_encoding, seq_len, vocab_size,
                           num_steps=50, temperature=1.0,
                           noise_scale=0.0, num_restarts=2):
    """
    Soft-masking recursion for equation generation.

    Args:
        model: Trained PhysMDT masked diffusion transformer
        data_encoding: z ∈ R^{d_model} from set encoder
        seq_len: Maximum output sequence length L
        vocab_size: |V| = 128
        num_steps: Refinement steps per restart (default 50)
        temperature: Softmax temperature for logit normalization
        noise_scale: Optional noise injection magnitude
        num_restarts: Number of cold restarts (default 2)

    Returns:
        best_equation: Token sequence of most-visited candidate
    """

    candidate_counts = {}  # Track candidate frequencies
    steps_per_restart = num_steps // num_restarts

    for restart in range(num_restarts):
        # Step 1: Initialize with fully masked logits
        logits = zeros(seq_len, vocab_size)
        logits[:, MASK_TOKEN_ID] = 1.0  # All mass on [MASK]

        for step in range(steps_per_restart):
            # Step 2: Convert logits to soft embeddings
            # (continuous, no argmax discretization)
            soft_probs = softmax(logits / temperature)
            soft_embeddings = soft_probs @ embedding_matrix  # [L, d_model]

            # Step 3: Add mask embedding to all positions
            # This signals "all positions need refinement"
            mask_emb = embedding_matrix[MASK_TOKEN_ID]
            soft_embeddings = soft_embeddings + mask_emb

            # Step 4: Forward pass through model
            logits = model.forward(
                data_encoding=data_encoding,
                token_embeddings=soft_embeddings,
                use_tree_pe=True
            )  # Output: [L, vocab_size]

            # Step 5: Optional noise injection for diversity
            if noise_scale > 0:
                noise = randn_like(logits) * noise_scale * (1 - step/steps_per_restart)
                logits = logits + noise

            # Step 6: Record discrete candidate at this step
            candidate = argmax(logits, dim=-1)
            candidate_key = tuple(candidate.tolist())
            candidate_counts[candidate_key] = candidate_counts.get(candidate_key, 0) + 1

            # Step 7: Normalize logits for next iteration
            logits = logits / temperature

    # Step 8: Select most-visited candidate across all steps
    best_key = max(candidate_counts, key=candidate_counts.get)
    best_equation = list(best_key)

    return best_equation
```

### Key Design Decisions

1. **No argmax between steps**: The soft probability distributions are maintained as continuous vectors, allowing the model to express uncertainty between tokens (e.g., 60% `sin`, 40% `cos`) and gradually resolve it.

2. **Mask embedding addition**: Adding the mask embedding to every position (not just truly masked ones) signals the model that all positions should be reconsidered, enabling self-correction.

3. **Cold restarts**: Periodically re-initializing from fully masked state (while preserving candidate counts) helps escape local optima.

4. **Most-visited-candidate selection**: Rather than taking the final prediction (which may oscillate), selecting the most frequently visited discrete candidate provides a robust consensus across the refinement trajectory.

5. **Decaying noise**: Optional noise injection starts high (for exploration) and decays to zero (for convergence), similar to simulated annealing.

## 6. Test-Time Finetuning (TTF) Protocol

### Configuration

| Parameter | Base Model | Scaled Model |
|-----------|-----------|--------------|
| **LoRA rank** | 32 | 32 |
| **LoRA alpha** | 64 | 64 |
| **LoRA target modules** | Q, K, V projections | Q, K, V projections |
| **Learning rate** | 1e-4 | 5e-5 |
| **Optimizer** | AdamW (β₁=0.9, β₂=0.999) | AdamW |
| **Weight decay** | 0.01 | 0.01 |
| **Steps per equation** | 128 | 128 |
| **Batch size** | 1 | 1 |
| **Masking rate** | Uniform(0.3, 0.9) | Uniform(0.3, 0.9) |

### TTF Procedure

```python
def test_time_finetune(model, data_matrix, target_tokens, config):
    """
    Per-equation test-time finetuning with LoRA.

    1. Apply LoRA adapters to Q, K, V projections
    2. Freeze all original model weights
    3. For each step:
       a. Sample random augmentation (noise, variable scaling)
       b. Apply augmentation to data_matrix
       c. Sample random masking rate p ~ Uniform(0.3, 0.9)
       d. Mask target_tokens with rate p
       e. Forward pass, compute CE loss on masked positions
       f. Backward pass, update only LoRA parameters
    4. Run soft-masking recursion inference with adapted model
    5. Remove LoRA adapters, restore original weights
    """

    # Apply LoRA
    lora_params = apply_lora(model, rank=config.lora_rank, alpha=config.lora_alpha)
    optimizer = AdamW(lora_params, lr=config.ttf_lr, weight_decay=config.weight_decay)

    for step in range(config.ttf_steps):
        # Random augmentation per step
        aug_data = augment_data(data_matrix,
                               noise_std=0.01 * random(),
                               scale_factor=1.0 + 0.1 * (random() - 0.5))

        # Random masking
        mask_rate = uniform(0.3, 0.9)
        masked_tokens, mask = random_mask(target_tokens, mask_rate)

        # Forward and loss
        data_enc = model.encode_data(aug_data)
        logits = model.forward(data_enc, masked_tokens)
        loss = cross_entropy(logits[mask], target_tokens[mask])

        # Update only LoRA params
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Inference with adapted model
    equation = soft_masking_inference(model, data_enc, ...)

    # Cleanup
    remove_lora(model)

    return equation
```

### Data Augmentation During TTF

Each TTF step uses a unique random augmentation:
1. **Gaussian noise**: Add N(0, σ²) with σ ~ Uniform(0, 0.05) to data values
2. **Variable scaling**: Multiply each variable by (1 + ε) where ε ~ Uniform(-0.1, 0.1)
3. **Subsampling**: Randomly select 80-100% of data points per step
4. **Permutation**: Shuffle the order of data points (set encoder is permutation-invariant, but this changes batch statistics)

## 7. Training Configuration

### Base Model Training

```yaml
# configs/base_model.yaml
model:
  type: physmdt_base
  d_model: 512
  n_layers: 8
  n_heads: 8
  ffn_dim: 2048
  max_seq_len: 64
  vocab_size: 128
  dropout: 0.1
  positional_encoding: tree_aware_2d_rope

encoder:
  type: deepsets_mha
  d_model: 512
  n_layers: 4
  n_heads: 8

training:
  optimizer: adamw
  lr: 3e-4
  weight_decay: 0.01
  warmup_steps: 5000
  total_steps: 100000
  batch_size: 64
  mask_rate_min: 0.1
  mask_rate_max: 0.9
  gradient_clip: 1.0
  seed: 42
  checkpoint_interval: 10000

data:
  fsred_train: true
  procedural_newtonian: 50000
  noise_levels: [0.0]
  augmentation: true
  physics_prior_weight: 0.1
```

### Scaled Model Training

```yaml
# configs/scaled_model.yaml
model:
  type: physmdt_scaled
  d_model: 1024
  n_layers: 16
  n_heads: 16
  ffn_dim: 4096
  max_seq_len: 64
  vocab_size: 128
  dropout: 0.1
  positional_encoding: tree_aware_2d_rope

encoder:
  type: deepsets_mha
  d_model: 1024
  n_layers: 6
  n_heads: 16

training:
  optimizer: adamw
  lr: 1e-4
  weight_decay: 0.01
  warmup_steps: 10000
  total_steps: 500000
  batch_size: 32
  seed: 42
  checkpoint_interval: 10000
```
