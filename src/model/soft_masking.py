"""Soft-masking recursion inference for PhysMDT.

Implements the ARChitects' core innovation: model output logits are fed back
as continuous input combined with mask embeddings to trigger iterative
refinement without hard discretization between steps.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple
from collections import Counter


def soft_masking_inference(
    model,
    data_matrix: torch.Tensor,
    seq_len: int = 64,
    num_steps: int = 50,
    temperature: float = 1.0,
    noise_scale: float = 0.0,
    num_restarts: int = 2,
    mask_token_id: int = 3,
    bos_token_id: int = 1,
    eos_token_id: int = 2,
    pos_encoding: Optional[torch.Tensor] = None,
) -> Tuple[List[int], Dict]:
    """Soft-masking recursion for equation generation.

    Args:
        model: Trained PhysMDT model.
        data_matrix: (1, n_points, input_dim) single data matrix.
        seq_len: Output sequence length.
        num_steps: Total refinement steps across all restarts.
        temperature: Softmax temperature.
        noise_scale: Optional noise injection magnitude.
        num_restarts: Number of cold restarts.
        mask_token_id: Token ID for [MASK].
        bos_token_id: Token ID for [BOS].
        eos_token_id: Token ID for [EOS].
        pos_encoding: Optional positional encoding tensor.

    Returns:
        best_tokens: Best equation token sequence.
        info: Dict with refinement trajectory information.
    """
    device = data_matrix.device
    vocab_size = model.vocab_size
    embedding_matrix = model.token_embedding.weight  # (vocab, d_model)
    mask_emb = embedding_matrix[mask_token_id]  # (d_model,)

    candidate_counts = Counter()
    trajectory = []  # Track NED/quality over steps

    steps_per_restart = max(num_steps // num_restarts, 1)

    for restart in range(num_restarts):
        # Initialize with fully masked logits
        logits = torch.zeros(1, seq_len, vocab_size, device=device)
        logits[:, :, mask_token_id] = 1.0

        # Set BOS at position 0
        logits[:, 0, :] = -1e9
        logits[:, 0, bos_token_id] = 10.0

        for step in range(steps_per_restart):
            # Convert logits to soft embeddings (continuous, no argmax)
            soft_probs = F.softmax(logits / temperature, dim=-1)  # (1, seq_len, vocab)
            soft_embeddings = soft_probs @ embedding_matrix  # (1, seq_len, d_model)

            # Add mask embedding to all positions
            soft_embeddings = soft_embeddings + mask_emb.unsqueeze(0).unsqueeze(0)

            # Forward pass through model
            logits = model.forward_soft(data_matrix, soft_embeddings, pos_encoding)

            # Force BOS at position 0
            logits[:, 0, :] = -1e9
            logits[:, 0, bos_token_id] = 10.0

            # Optional noise injection (decaying)
            if noise_scale > 0:
                decay = 1.0 - step / steps_per_restart
                noise = torch.randn_like(logits) * noise_scale * decay
                logits = logits + noise

            # Record discrete candidate
            candidate = logits.argmax(dim=-1)[0].tolist()
            candidate_key = tuple(candidate)
            candidate_counts[candidate_key] = candidate_counts.get(candidate_key, 0) + 1

            # Track trajectory
            trajectory.append({
                'restart': restart,
                'step': step,
                'candidate': candidate,
                'confidence': float(F.softmax(logits.detach() / temperature, dim=-1).max(dim=-1).values.mean()),
            })

    # Select most-visited candidate
    if candidate_counts:
        best_key = max(candidate_counts, key=candidate_counts.get)
        best_tokens = list(best_key)
    else:
        best_tokens = [bos_token_id] + [mask_token_id] * (seq_len - 2) + [eos_token_id]

    info = {
        'num_candidates': len(candidate_counts),
        'best_visit_count': candidate_counts.get(tuple(best_tokens), 0),
        'total_steps': num_steps,
        'trajectory_length': len(trajectory),
    }

    return best_tokens, info


def compute_refinement_ned_trajectory(
    model, data_matrix, true_expr_str, tokenizer,
    num_steps=50, temperature=1.0,
    pos_encoding=None,
) -> List[float]:
    """Run soft-masking and compute NED at each step.

    Used to verify monotonically improving NED over refinement.

    Returns:
        List of NED values at each refinement step.
    """
    from src.evaluation.metrics import normalized_edit_distance

    device = data_matrix.device
    vocab_size = model.vocab_size
    seq_len = 64
    embedding_matrix = model.token_embedding.weight
    mask_emb = embedding_matrix[3]

    ned_trajectory = []

    logits = torch.zeros(1, seq_len, vocab_size, device=device)
    logits[:, :, 3] = 1.0
    logits[:, 0, :] = -1e9
    logits[:, 0, 1] = 10.0

    for step in range(num_steps):
        soft_probs = F.softmax(logits / temperature, dim=-1)
        soft_embeddings = soft_probs @ embedding_matrix
        soft_embeddings = soft_embeddings + mask_emb.unsqueeze(0).unsqueeze(0)
        logits = model.forward_soft(data_matrix, soft_embeddings, pos_encoding)
        logits[:, 0, :] = -1e9
        logits[:, 0, 1] = 10.0

        # Decode current prediction
        pred_ids = logits.argmax(dim=-1)[0].tolist()
        pred_str = tokenizer.decode(pred_ids)
        ned = normalized_edit_distance(pred_str, true_expr_str)
        ned_trajectory.append(ned)

    return ned_trajectory
