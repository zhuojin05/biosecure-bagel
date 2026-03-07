"""
Multi-signal triage engine for aggregation / prion risk assessment.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from __future__ import annotations

import numpy as np

from bagel.oracles.embedding.base import EmbeddingOracle
from bagel.oracles.folding.base import FoldingOracle

from .robustness import LocalRobustnessProbe
from .signals import (
    AggregationSignal,
    EmbeddingSignal,
    MotifDetectionSignal,
    SequenceHomologySignal,
    StructuralPropensitySignal,
)


class MultiSignalTriageEngine:
    """
    Combines multiple aggregation signals into a single risk assessment.

    Fast (sequence-only) signals are always present.  Slow (oracle-dependent)
    signals are enabled by passing the corresponding oracle.  Robustness
    probing is applied to fast signals only.

    Parameters
    ----------
    esmfold : FoldingOracle | None
        If provided, enables :class:`StructuralPropensitySignal`.
    esm2 : EmbeddingOracle | None
        If provided, enables :class:`EmbeddingSignal`.
    use_modal : bool
        Passed through to oracle construction (not used here; oracles are
        created externally and injected).
    n_perturbations : int
        Number of single-residue perturbations for robustness probing.
    """

    # Default signal weights.  Homology is weighted 3x motif because the
    # sequence-homology signal is a strong discriminator while the motif
    # signal captures complementary but noisier features.
    DEFAULT_SIGNAL_WEIGHTS: dict[str, float] = {
        'homology': 0.75,
        'motif': 0.25,
        'structure': 1.0,
        'embedding': 1.0,
    }

    def __init__(
        self,
        esmfold: FoldingOracle | None = None,
        esm2: EmbeddingOracle | None = None,
        use_modal: bool = False,
        n_perturbations: int = 30,
        signal_weights: dict[str, float] | None = None,
    ) -> None:
        self.fast_signals: dict[str, AggregationSignal] = {
            'homology': SequenceHomologySignal(),
            'motif': MotifDetectionSignal(),
        }
        self.slow_signals: dict[str, AggregationSignal] = {}
        if esmfold is not None:
            self.slow_signals['structure'] = StructuralPropensitySignal(esmfold)
        if esm2 is not None:
            self.slow_signals['embedding'] = EmbeddingSignal(esm2)

        self.probe = LocalRobustnessProbe()
        self.n_perturbations = n_perturbations
        self.signal_weights: dict[str, float] = signal_weights if signal_weights is not None else dict(self.DEFAULT_SIGNAL_WEIGHTS)

    def assess_risk(self, sequence: str) -> dict:
        """
        Assess aggregation risk for a single amino-acid sequence.

        Returns
        -------
        dict with keys:
            combined_score : float
                Weighted combination of all signal scores.  Range ~ [0, 2].
            risk_level : str
                'HIGH' | 'MEDIUM' | 'LOW'
            signal_scores : dict[str, float]
                Raw score from each signal.
            robustness_scores : dict[str, float]
                Robustness score for each signal (slow signals fixed at 1.0).
        """
        all_signals: dict[str, AggregationSignal] = {**self.fast_signals, **self.slow_signals}

        signal_scores = {name: sig.score(sequence) for name, sig in all_signals.items()}

        # Robustness probing on fast signals only (no oracle calls needed)
        robustness_scores = self.probe.probe_robustness(sequence, self.fast_signals, self.n_perturbations)
        # Slow signals treated as maximally robust
        for name in self.slow_signals:
            robustness_scores[name] = 1.0

        weights = {n: self.signal_weights.get(n, 1.0) for n in all_signals}
        total_weight = sum(weights.values())
        combined = float(
            sum(weights[n] * signal_scores[n] * (1.0 + robustness_scores[n]) for n in all_signals)
            / total_weight
        )

        if combined > 1.2:
            risk_level = 'HIGH'
        elif combined > 0.6:
            risk_level = 'MEDIUM'
        else:
            risk_level = 'LOW'

        return {
            'combined_score': combined,
            'risk_level': risk_level,
            'signal_scores': signal_scores,
            'robustness_scores': robustness_scores,
        }
