"""
Local robustness probing for aggregation risk signals.

A signal whose score remains elevated across random single-residue mutations
is a more robust indicator of genuine risk than a one-off hit.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from __future__ import annotations

import numpy as np

import bagel as bg
from bagel.constants import mutation_bias_no_cystein
from bagel.mutation import Canonical

from .signals import AggregationSignal, _sequence_to_chain


def _sequence_to_system(sequence: str) -> bg.System:
    """Build a minimal System with no energy terms — safe to mutate without oracle calls."""
    chain = _sequence_to_chain(sequence)
    state = bg.State(name='probe', chains=[chain], energy_terms=[])
    return bg.System(states=[state], name='probe')


class LocalRobustnessProbe:
    """
    Probes signal stability under single-residue perturbations.

    Parameters
    ----------
    n_mutations : int
        Number of substitutions per perturbation step (default: 1).
    mutation_bias : dict
        Per-residue sampling bias (default: uniform excluding cysteine).
    """

    def __init__(
        self,
        n_mutations: int = 1,
        mutation_bias: dict[str, float] = mutation_bias_no_cystein,
    ) -> None:
        self.n_mutations = n_mutations
        self.mutation_bias = mutation_bias
        self._protocol = Canonical(n_mutations=n_mutations, mutation_bias=mutation_bias)

    def generate_local_variants(self, sequence: str, n_variants: int = 30) -> list[str]:
        """
        Generate ``n_variants`` single-residue substitution variants.

        Each variant is generated independently from the original sequence
        (not from previous variants), ensuring i.i.d. perturbations.
        """
        variants: list[str] = []
        for _ in range(n_variants):
            system = _sequence_to_system(sequence)
            mutated_system, _ = self._protocol.one_step(system)
            mutated_seq = mutated_system.states[0].chains[0].sequence
            variants.append(mutated_seq)
        return variants

    def probe_robustness(
        self,
        sequence: str,
        signals: dict[str, AggregationSignal],
        n_perturbations: int = 30,
    ) -> dict[str, float]:
        """
        Score robustness of each signal across local perturbations.

        Robustness is defined as::

            robustness_i = max(0, 1 - std(scores_i) / (mean(scores_i) + 1e-6))

        A signal with low coefficient of variation across perturbations is
        considered robust (score near 1.0); a highly variable signal returns
        a score near 0.0.

        Parameters
        ----------
        sequence : str
            Query amino-acid sequence.
        signals : dict
            Mapping from signal name to :class:`AggregationSignal` instance.
        n_perturbations : int
            Number of random single-residue variants to evaluate.

        Returns
        -------
        dict
            ``{signal_name: robustness_score}`` for each signal.
        """
        variants = self.generate_local_variants(sequence, n_variants=n_perturbations)

        robustness: dict[str, float] = {}
        for name, signal in signals.items():
            scores = np.array([signal.score(v) for v in variants], dtype=float)
            std = float(np.std(scores))
            mean = float(np.mean(scores))
            rob = max(0.0, 1.0 - std / (mean + 1e-6))
            robustness[name] = rob

        return robustness
