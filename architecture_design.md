# PhysDiffuse: Architecture Design

## 1. Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PhysDiffuse Architecture                         │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  INPUT: Observation Table (N×D matrix)                                  │
│  ┌──────────────────────────────────────────┐                           │
│  │  x_0  x_1  x_2  ...  x_D  │  y          │                           │
│  │  1.5  3.2  -    ...  -    │  4.8         │  N rows × (D+1) cols     │
│  │  0.8  9.8  -    ...  -    │  7.84        │                           │
│  │  ...  ...  ...  ...  ...  │  ...         │                           │
│  └──────────────┬───────────────────────────┘                           │
│                 │                                                        │
│                 ▼                                                        │
│  ┌──────────────────────────────────────────┐                           │
│  │       SET-TRANSFORMER ENCODER             │                          │
│  │  ┌────────────────────────────────────┐  │                           │
│  │  │  Input Embedding: Linear(D+1, 512) │  │                           │
│  │  │  + Positional ID Embedding          │  │                           │
│  │  └──────────────┬─────────────────────┘  │                           │
│  │                 ▼                         │                           │
│  │  ┌────────────────────────────────────┐  │                           │
│  │  │  ISAB × 4 (Induced Set Attention)  │  │                           │
│  │  │  - d_model=512, 8 heads            │  │                           │
│  │  │  - 32 inducing points              │  │                           │
│  │  │  - Permutation invariant           │  │                           │
│  │  └──────────────┬─────────────────────┘  │                           │
│  │                 ▼                         │                           │
│  │  ┌────────────────────────────────────┐  │                           │
│  │  │  PMA (Pooling by Multi-Head Attn)  │  │                           │
│  │  │  → K=16 summary vectors            │  │                           │
│  │  └──────────────┬─────────────────────┘  │                           │
│  │                 │                         │                           │
│  │  Output: enc_out ∈ R^{16 × 512}         │                           │
│  └──────────────┬───────────────────────────┘                           │
│                 │                                                        │
│                 │  cross-attention                                       │
│                 ▼                                                        │
│  ┌──────────────────────────────────────────┐                           │
│  │     MASKED-DIFFUSION DECODER              │                          │
│  │  ┌────────────────────────────────────┐  │                           │
│  │  │  Token Embedding: Embed(V, 512)    │  │                           │
│  │  │  + Positional Encoding (sinusoidal)│  │                           │
│  │  │  + Mask Embedding (learnable)      │  │                           │
│  │  └──────────────┬─────────────────────┘  │                           │
│  │                 ▼                         │                           │
│  │  ┌────────────────────────────────────┐  │                           │
│  │  │  Transformer Block × 8             │  │                           │
│  │  │  - Bidirectional Self-Attention    │  │                           │
│  │  │  - Cross-Attention to enc_out      │  │                           │
│  │  │  - FFN (512 → 2048 → 512)         │  │                           │
│  │  │  - LayerNorm, Dropout(0.1)         │  │                           │
│  │  └──────────────┬─────────────────────┘  │                           │
│  │                 ▼                         │                           │
│  │  ┌────────────────────────────────────┐  │                           │
│  │  │  Output Head: Linear(512, V)       │  │                           │
│  │  │  → logits per position             │  │                           │
│  │  └──────────────┬─────────────────────┘  │                           │
│  └──────────────┬───────────────────────────┘                           │
│                 │                                                        │
│                 ▼                                                        │
│  ┌──────────────────────────────────────────┐                           │
│  │     TRAINING: Masked Diffusion Loss       │                          │
│  │  L = -E[Σ_{i∈masked} log p(x_i|x_unmasked, enc_out)]               │
│  └──────────────────────────────────────────┘                           │
│                                                                         │
│  ┌──────────────────────────────────────────┐                           │
│  │     INFERENCE: Recursive Soft-Masking     │                          │
│  │  1. Start: all positions masked           │                          │
│  │  2. For t = 1..T:                         │                          │
│  │     a. Forward pass → logits              │                          │
│  │     b. Normalize logits (temperature)     │                          │
│  │     c. Sample tokens from p(x_i|...)      │                          │
│  │     d. Apply soft-masking schedule        │                          │
│  │  3. Optional: cold restart (re-mask all)  │                          │
│  │  4. Select most-visited candidate         │                          │
│  └──────────────────────────────────────────┘                           │
│                                                                         │
│  ┌──────────────────────────────────────────┐                           │
│  │     POST-PROCESSING                       │                          │
│  │  1. SymPy canonical simplification        │                          │
│  │  2. BFGS constant optimization            │                          │
│  │  3. Dimensional analysis filter           │                          │
│  │  4. Pareto-optimal selection              │                          │
│  └──────────────────────────────────────────┘                           │
│                                                                         │
│  OUTPUT: Symbolic equation in prefix notation                           │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Mathematical Formulation of Soft-Masking Refinement

### Training Objective (Masked Diffusion)

Given a sequence of tokens `x = (x_1, ..., x_L)` representing an equation in prefix notation:

1. **Forward process:** Sample masking ratio `t ~ Uniform(0, 1)`. For each position i, independently mask with probability t:
   ```
   m_i ~ Bernoulli(t)
   x̃_i = [MASK] if m_i = 1, else x_i
   ```

2. **Training loss:** Cross-entropy only on masked positions, conditioned on encoder output `h_enc`:
   ```
   L_MDM = -E_{t,m} [ Σ_{i: m_i=1} log p_θ(x_i | x̃, h_enc) ]
   ```
   This is equivalent to the MDLM/LLaDA objective (Sahoo et al., 2024; Nie et al., 2025).

### Inference: Recursive Soft-Masking Refinement

Inspired by the ARC2025 ARChitects solution:

**Input:** Encoder output `h_enc`, total refinement steps `T`, number of rounds `R`, temperature schedule `τ(t)`

```
Algorithm: Recursive Soft-Masking Refinement
─────────────────────────────────────────────
candidates = {}
for round r = 1..R:
    x̃ = [MASK, MASK, ..., MASK]  (length L)    // cold start
    for step t = 1..T/R:
        // Forward pass
        logits = Decoder(x̃, h_enc)              // (L, V) logit matrix

        // Logit normalization (prevent overconfidence)
        logits = logits / (||logits||₂ / √d + ε)

        // Temperature-controlled sampling
        τ_t = τ_start * (τ_end / τ_start)^(t/(T/R))  // geometric annealing
        probs = softmax(logits / τ_t)
        x_sampled = sample(probs)                       // per-position sampling

        // Confidence-based unmasking schedule
        conf = max(probs, dim=-1)                       // per-position confidence
        n_unmask = floor(L * t / (T/R))                 // linear unmasking schedule
        top_indices = argsort(conf, descending=True)[:n_unmask]

        // Update: unmask high-confidence positions
        for i in top_indices:
            x̃[i] = x_sampled[i]

    // Store final candidate from this round
    candidates[hash(x̃)] += 1

// Select most-visited candidate
output = argmax_{c ∈ candidates} candidates[c]
```

**Logit Normalization Detail:**
Following the ARC2025 approach, we apply L2-normalization to logits before softmax to prevent the model from becoming overconfident in early refinement steps:
```
logits_normalized = logits * (√d_model / (||logits||₂ + ε))
```
where `d_model = 512` and `ε = 1e-6`.

---

## 3. Encoder Specification: Set-Transformer

The encoder processes N observation points `{(x_i, y_i)}_{i=1}^N` where each `x_i ∈ R^D` is a vector of independent variables and `y_i ∈ R` is the dependent variable.

### Architecture
- **Input projection:** Linear(D+1, d_model) mapping each observation point to a d_model-dimensional vector
- **Variable ID embedding:** Learnable embedding added to identify which input dimensions correspond to which variables
- **ISAB layers:** 4 Induced Set Attention Blocks (Lee et al., 2019)
  - Each ISAB uses M=32 inducing points
  - Multi-head attention with 8 heads, d_model=512
  - This makes the encoder O(N*M) instead of O(N²)
- **PMA layer:** Pooling by Multi-head Attention producing K=16 summary vectors
  - Output: `h_enc ∈ R^{K × d_model}` = R^{16 × 512}

### Key Properties
- **Permutation invariant:** Output is invariant to reordering of observation points
- **Variable-count agnostic:** Handles 1-10 input variables by zero-padding
- **Scalable:** O(N*M) complexity handles N=200 points efficiently

### Parameter Count
- Input projection: (10+1) × 512 = 5,632
- Variable ID embedding: 10 × 512 = 5,120
- ISAB layers (4): 4 × (2 × (3 × 512 × 512 + 512) + 2 × 512 × 2048 + 2048 + 512) ≈ 4 × 5.3M ≈ 21M
- PMA: ~2.7M
- **Encoder total: ~24M parameters**

---

## 4. Decoder Specification: Masked-Diffusion Transformer

### Architecture
- **Token embedding:** `Embedding(V, d_model)` where V ≤ 128 (vocabulary size), d_model=512
- **Position encoding:** Sinusoidal positional encoding for sequence positions 0..L_max (L_max=64)
- **Mask embedding:** Learnable vector `e_mask ∈ R^{d_model}` added to masked positions
- **Transformer blocks:** 8 layers, each containing:
  1. **Bidirectional self-attention:** 8 heads, d_k = d_v = 64, no causal mask
  2. **Cross-attention to encoder:** 8 heads attending to `h_enc` (16 key-value pairs)
  3. **FFN:** Linear(512, 2048) → GELU → Linear(2048, 512)
  4. **Pre-LayerNorm, Dropout(0.1), residual connections**
- **Output head:** Linear(512, V) producing logits over vocabulary

### Key Design Choices
- **Bidirectional attention:** Unlike autoregressive decoders, all positions attend to all other positions. This is crucial for the masked diffusion objective where any subset of tokens may be masked.
- **Cross-attention integration:** Encoder output provides the "conditioning signal" — the numerical data that the equation should fit.
- **No causal mask:** Since we're not generating left-to-right, all positions see all other unmasked positions.

### Parameter Count
- Token embedding: 128 × 512 = 65,536
- Position encoding: 0 (sinusoidal, no parameters)
- Mask embedding: 512
- Transformer blocks (8): 8 × (3 × 512² + 512 + 3 × 512² + 512 + 512 × 2048 + 2048 + 2048 × 512 + 512 + 4 × 512) ≈ 8 × 6.8M ≈ 54M
- Output head: 512 × 128 = 65,536
- **Decoder total: ~55M parameters**

---

## 5. Dimensional Analysis Constraint Loss

### Unit System
Each physical variable `x_i` has associated SI base-unit exponents `[M_i, L_i, T_i]`:
- Mass (M), Length (L), Time (T)
- Example: velocity → [0, 1, -1], force → [1, 1, -2], energy → [1, 2, -2]

### Unit Propagation Rules
For an expression tree node with operator `op` and children with units `u_left`, `u_right`:
```
units(add(a, b)) = u_a  (requires u_a == u_b)
units(sub(a, b)) = u_a  (requires u_a == u_b)
units(mul(a, b)) = u_a + u_b  (element-wise addition of exponents)
units(div(a, b)) = u_a - u_b  (element-wise subtraction)
units(pow(a, n)) = n * u_a  (scalar multiplication)
units(sqrt(a))   = 0.5 * u_a
units(sin(a))    = [0, 0, 0]  (requires u_a == [0, 0, 0])
units(cos(a))    = [0, 0, 0]  (requires u_a == [0, 0, 0])
units(exp(a))    = [0, 0, 0]  (requires u_a == [0, 0, 0])
units(log(a))    = [0, 0, 0]  (requires u_a == [0, 0, 0])
```

### Auxiliary Loss
```
L_dim = λ_dim × Σ_{nodes with add/sub} ||u_left - u_right||²₂
      + λ_dim × Σ_{nodes with sin/cos/exp/log} ||u_child||²₂
```
Default: `λ_dim = 0.1`

### Inference-time Filter
During candidate selection, compute dimensional consistency score:
```
dim_score(expr) = 1 if all constraints satisfied, else exp(-α * total_violation)
```
Candidates are weighted by `dim_score` during most-visited selection.

---

## 6. Tokenization Scheme

### Vocabulary (V = 73 tokens)

| Category | Tokens | Count |
|----------|--------|-------|
| **Operators** | `add`, `sub`, `mul`, `div`, `pow`, `sqrt`, `sin`, `cos`, `exp`, `log`, `neg`, `abs`, `inv` | 13 |
| **Variables** | `x_0`, `x_1`, ..., `x_9` | 10 |
| **Learnable constants** | `c_0`, `c_1`, ..., `c_9` | 10 |
| **Integer constants** | `int_0`, `int_1`, `int_2`, `int_3`, `int_4`, `int_5` | 6 |
| **Special constants** | `pi`, `e_const`, `half`, `third`, `quarter` | 5 |
| **Digit tokens** | `d_0`, ..., `d_9`, `d_dot`, `d_neg` | 12 |
| **Control** | `<SOS>`, `<EOS>`, `<PAD>`, `<MASK>` | 4 |
| **Reserved** | (future expansion) | 13 |
| **Total** | | **73** (< 128) |

### Prefix Notation Encoding
Expressions are encoded as pre-order traversals of the expression tree:
- `F = m * a` → `<SOS> mul x_0 x_1 <EOS>`
- `E = 0.5 * m * v^2` → `<SOS> mul half mul x_0 pow x_1 int_2 <EOS>`
- `F = G*m1*m2/r^2` → `<SOS> div mul mul c_0 x_0 x_1 pow x_2 int_2 <EOS>`

### Maximum Sequence Length
L_max = 64 tokens (sufficient for all Tier 1-4 equations in the benchmark).

---

## 7. Parameter Count Summary

| Component | Parameters |
|-----------|-----------|
| Set-Transformer Encoder | ~24M |
| Masked-Diffusion Decoder | ~55M |
| Dimensional Analysis Head | ~0.5M |
| **Total** | **~80M** |

With LoRA adapters for TTT (rank 32 on all attention layers):
- LoRA parameters per equation: ~2M additional (not counted in base model)

**Memory estimation (mixed precision, batch size 16):**
- Model parameters: 80M × 2 bytes = 160MB
- Optimizer states (AdamW): 80M × 8 bytes = 640MB
- Activations (batch 16, seq 64): ~2GB
- **Total: ~3GB** (well within 40GB A100)

Note: We target 80M parameters rather than the original 100-300M range to ensure fast iteration and to demonstrate that even a relatively compact model can achieve strong results. If needed, the model can be scaled up by increasing d_model to 768 (yielding ~180M parameters).

---

## 8. Pseudocode: Full Forward Pass and Inference Loop

### Training Forward Pass

```python
def train_step(obs_table, target_tokens, mask_ratio=None):
    """
    obs_table: (B, N, D+1) observation data
    target_tokens: (B, L) ground truth token sequences
    mask_ratio: float or None (sample from Uniform(0,1) if None)
    """
    # Encode observations
    h_enc = set_transformer_encoder(obs_table)          # (B, K, d_model)

    # Sample masking ratio
    if mask_ratio is None:
        t = torch.rand(B, 1, device=device)             # (B, 1)
    else:
        t = torch.full((B, 1), mask_ratio, device=device)

    # Create masks (per-position Bernoulli)
    mask = torch.rand(B, L, device=device) < t           # (B, L) boolean

    # Mask input tokens
    masked_tokens = target_tokens.clone()
    masked_tokens[mask] = MASK_TOKEN_ID

    # Decoder forward (bidirectional attention)
    logits = masked_diffusion_decoder(masked_tokens, h_enc)  # (B, L, V)

    # Loss: cross-entropy only on masked positions
    loss_mdm = F.cross_entropy(
        logits[mask].view(-1, V),
        target_tokens[mask].view(-1),
        reduction='mean'
    )

    # Optional: dimensional analysis loss
    if use_dim_analysis:
        loss_dim = compute_dim_loss(logits, variable_units)
        loss = loss_mdm + lambda_dim * loss_dim
    else:
        loss = loss_mdm

    return loss
```

### Inference Loop (Recursive Soft-Masking Refinement)

```python
def inference(obs_table, T=64, R=2, n_samples=128, tau_start=1.0, tau_end=0.1):
    """
    obs_table: (1, N, D+1) single observation table
    T: total refinement steps
    R: number of rounds (cold restarts)
    n_samples: number of parallel candidates
    """
    # Encode observations (shared across all candidates)
    h_enc = set_transformer_encoder(obs_table)          # (1, K, d_model)
    h_enc = h_enc.expand(n_samples, -1, -1)             # (n_samples, K, d_model)

    candidate_counts = Counter()

    for round in range(R):
        # Cold start: all positions masked
        x = torch.full((n_samples, L), MASK_TOKEN_ID)   # (n_samples, L)

        steps_per_round = T // R
        for step in range(1, steps_per_round + 1):
            # Forward pass
            logits = masked_diffusion_decoder(x, h_enc)  # (n_samples, L, V)

            # Logit normalization
            norm = logits.norm(dim=-1, keepdim=True) / (d_model ** 0.5) + 1e-6
            logits = logits / norm

            # Temperature annealing
            tau = tau_start * (tau_end / tau_start) ** (step / steps_per_round)
            probs = F.softmax(logits / tau, dim=-1)      # (n_samples, L, V)

            # Sample tokens
            x_sampled = torch.multinomial(
                probs.view(-1, V), 1
            ).view(n_samples, L)

            # Confidence-based unmasking
            conf = probs.max(dim=-1).values               # (n_samples, L)
            n_unmask = int(L * step / steps_per_round)

            # For each sample, unmask top-n_unmask positions by confidence
            for b in range(n_samples):
                top_k = conf[b].topk(n_unmask).indices
                x[b, top_k] = x_sampled[b, top_k]

        # Count final candidates
        for b in range(n_samples):
            seq = tuple(x[b].tolist())
            candidate_counts[seq] += 1

    # Select most-visited candidate
    best_seq = candidate_counts.most_common(1)[0][0]

    # Post-process: SymPy simplification + BFGS constant optimization
    equation = decode_prefix(best_seq)
    equation = sympy_simplify(equation)
    equation = bfgs_optimize_constants(equation, obs_table)

    return equation
```

### Test-Time Training (TTT) Loop

```python
def test_time_train(model, obs_table, n_aug=64, n_steps=128, lr=1e-4, lora_rank=32):
    """
    Attach LoRA adapters and fine-tune on augmented test instance.
    """
    # Attach LoRA to all attention layers
    lora_params = attach_lora(model.decoder, rank=lora_rank)
    optimizer = Adam(lora_params, lr=lr)

    # Generate augmentations of the test observation table
    aug_tables = []
    for i in range(n_aug):
        aug = augment(obs_table,
            noise_level=random.choice([0.01, 0.05, 0.10]),
            subsample_ratio=random.uniform(0.5, 1.0),
            permute_variables=True,
            rescale_units=True
        )
        aug_tables.append(aug)
    aug_batch = torch.stack(aug_tables)  # (n_aug, N, D+1)

    # Self-supervised TTT: masked reconstruction on augmented data
    for step in range(n_steps):
        # Sample a mini-batch of augmentations
        idx = torch.randint(0, n_aug, (16,))
        batch = aug_batch[idx]

        # Encode
        h_enc = model.encoder(batch)

        # For TTT, we use the model's own predictions as pseudo-targets
        with torch.no_grad():
            pseudo_targets = model.generate(batch, T=16)  # quick generation

        # Masked reconstruction loss
        t = torch.rand(16, 1)
        mask = torch.rand(16, L) < t
        masked = pseudo_targets.clone()
        masked[mask] = MASK_TOKEN_ID

        logits = model.decoder(masked, h_enc)
        loss = F.cross_entropy(logits[mask].view(-1, V), pseudo_targets[mask].view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Generate candidates with TTT-adapted model
    equation = inference(obs_table, T=64, R=2, n_samples=128)

    # Remove LoRA adapters
    remove_lora(model.decoder)

    return equation
```
