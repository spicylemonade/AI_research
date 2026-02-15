"""
Curriculum learning strategy with progressive complexity scheduling.
Multi-phase training from simple to complex equations.
"""
import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Subset, DataLoader, WeightedRandomSampler

from src.data.generator import PhysicsDataset


class CurriculumScheduler:
    """Multi-phase curriculum scheduler for progressive complexity training.

    Phases:
    A: Tier 1 only (kinematic equations)
    B: Tier 1-2 (add force laws)
    C: Tier 1-3 (add conservation laws)
    D: Full Tier 1-4 (all equations)
    """

    def __init__(
        self,
        dataset: PhysicsDataset,
        phase_thresholds: Dict[str, float] = None,
        hard_mining_weight: float = 2.0,
    ):
        self.dataset = dataset
        self.phase_thresholds = phase_thresholds or {
            'A': 0.70,  # Move to B when Tier 1 ESM > 70%
            'B': 0.50,  # Move to C when Tier 2 ESM > 50%
            'C': 0.30,  # Move to D when Tier 3 ESM > 30%
        }
        self.hard_mining_weight = hard_mining_weight

        # Build tier indices
        self.tier_indices = {}
        for tier in [1, 2, 3, 4]:
            self.tier_indices[tier] = dataset.get_tier_indices(tier)

        self.current_phase = 'A'
        self.phase_history = []

    def get_phase_tiers(self, phase: str) -> List[int]:
        """Get which tiers are active in a phase."""
        phase_map = {
            'A': [1],
            'B': [1, 2],
            'C': [1, 2, 3],
            'D': [1, 2, 3, 4],
        }
        return phase_map[phase]

    def get_phase_indices(self, phase: str) -> List[int]:
        """Get dataset indices for current phase."""
        tiers = self.get_phase_tiers(phase)
        indices = []
        for tier in tiers:
            indices.extend(self.tier_indices[tier])
        return indices

    def should_advance(self, tier_esm: Dict[int, float]) -> bool:
        """Check if we should advance to the next phase."""
        if self.current_phase == 'A':
            return tier_esm.get(1, 0) >= self.phase_thresholds['A']
        elif self.current_phase == 'B':
            return tier_esm.get(2, 0) >= self.phase_thresholds['B']
        elif self.current_phase == 'C':
            return tier_esm.get(3, 0) >= self.phase_thresholds['C']
        return False  # Phase D is final

    def advance_phase(self):
        """Advance to next phase."""
        phase_order = ['A', 'B', 'C', 'D']
        idx = phase_order.index(self.current_phase)
        if idx < len(phase_order) - 1:
            old = self.current_phase
            self.current_phase = phase_order[idx + 1]
            self.phase_history.append({
                'from': old, 'to': self.current_phase,
            })
            print(f"  Curriculum: Advancing from Phase {old} -> Phase {self.current_phase}")
            return True
        return False

    def get_dataloader(
        self, batch_size: int = 64, num_workers: int = 4,
        failure_indices: Optional[List[int]] = None,
        collate_fn=None,
    ) -> DataLoader:
        """Get DataLoader for current phase with optional hard mining."""
        indices = self.get_phase_indices(self.current_phase)

        if failure_indices and self.hard_mining_weight > 1:
            # Weighted sampling: upweight failed examples
            weights = np.ones(len(indices))
            failure_set = set(failure_indices)
            for i, idx in enumerate(indices):
                if idx in failure_set:
                    weights[i] = self.hard_mining_weight

            sampler = WeightedRandomSampler(
                weights, num_samples=len(indices), replacement=True
            )
            return DataLoader(
                Subset(self.dataset, indices),
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=True,
            )
        else:
            return DataLoader(
                Subset(self.dataset, indices),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
                pin_memory=True,
                collate_fn=collate_fn,
                drop_last=True,
            )

    def get_status(self) -> Dict:
        """Get current curriculum status."""
        return {
            'current_phase': self.current_phase,
            'active_tiers': self.get_phase_tiers(self.current_phase),
            'n_samples': len(self.get_phase_indices(self.current_phase)),
            'phase_history': self.phase_history,
        }
