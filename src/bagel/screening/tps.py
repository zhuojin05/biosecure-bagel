"""
Toxin Proximity Score (TPS) via Monte Carlo walk toward a toxin template.

Runs a constrained SimulatedAnnealing walk from the query sequence toward a
known toxin template and measures how easily the energy funnel is traversed.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

from biotite.structure import AtomArray, get_residue_starts

import bagel as bg
from bagel.callbacks import Callback, CallbackContext
from bagel.energies import TemplateMatchEnergy
from bagel.minimizer import SimulatedAnnealing
from bagel.mutation import Canonical
from bagel.oracles.folding.base import FoldingOracle
from bagel.state import State
from bagel.system import System

from .signals import _sequence_to_chain


class TPSCallback(Callback):
    """
    Records system_energy at every MC step for TPS computation.

    Attach to a :class:`SimulatedAnnealing` via its ``callbacks`` list.
    After ``minimize_system`` completes, read :attr:`trajectory` and call
    :meth:`tps` / :meth:`tps_auc`.
    """

    def __init__(self) -> None:
        self.trajectory: list[float] = []

    def on_step_end(self, context: CallbackContext) -> None:
        self.trajectory.append(context.metrics['system_energy'])

    def tps(self) -> float:
        """Fractional energy drop: (E(0) - E(N)) / E(0)."""
        if not self.trajectory:
            return 0.0
        e0 = self.trajectory[0]
        if e0 == 0.0:
            return 0.0
        return (e0 - self.trajectory[-1]) / e0

    def tps_auc(self) -> float:
        """Area under convergence curve: 1 - (1/N) * sum(E(t)/E(0))."""
        if not self.trajectory:
            return 0.0
        e0 = self.trajectory[0]
        if e0 == 0.0:
            return 0.0
        n = len(self.trajectory)
        return 1.0 - (1.0 / n) * sum(e / e0 for e in self.trajectory)


class ToxinProximityWalk:
    """
    Constrained Monte Carlo walk from a query sequence toward a toxin template.

    Builds a minimal :class:`System` with :class:`TemplateMatchEnergy`,
    runs :class:`SimulatedAnnealing`, and returns the energy trajectory plus
    TPS scores.

    Parameters
    ----------
    oracle : FoldingOracle
        Folding oracle used for structure prediction during the walk.
    template_atoms : AtomArray
        Atom array of the toxin template structure.
    n_steps : int
        Number of Monte Carlo steps.
    initial_temperature : float
        Starting annealing temperature.
    final_temperature : float
        Final annealing temperature.
    template_weight : float
        Weight for the :class:`TemplateMatchEnergy` term.
    """

    def __init__(
        self,
        oracle: FoldingOracle,
        template_atoms: AtomArray,
        n_steps: int = 500,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
        template_weight: float = 1.0,
    ) -> None:
        self.oracle = oracle
        self.template_atoms = template_atoms
        self.n_steps = n_steps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self.template_weight = template_weight

    def run(self, sequence: str) -> dict:
        """
        Run the MC walk from ``sequence`` toward the toxin template.

        Returns
        -------
        dict
            - ``tps``          : float — fractional energy drop
            - ``tps_auc``      : float — area under convergence curve
            - ``trajectory``   : list[float] — E(t) for each step
            - ``final_energy`` : float — last recorded energy
        """
        chain = _sequence_to_chain(sequence)

        # TemplateMatchEnergy requires equal atom counts in template and query.
        # Equalise by trimming to the shorter residue count, then using backbone
        # atoms only (N/CA/C/O = exactly 4 per residue, independent of AA type).
        n_query = len(chain.residues)
        template_res_starts = get_residue_starts(self.template_atoms)
        n_template = len(template_res_starts)
        n = min(n_query, n_template)

        if n < n_template:
            template_atoms = self.template_atoms[: template_res_starts[n]]
        else:
            template_atoms = self.template_atoms

        residues = chain.residues[:n]

        energy_terms = [
            TemplateMatchEnergy(
                oracle=self.oracle,
                template_atoms=template_atoms,
                residues=residues,
                backbone_only=True,
                weight=self.template_weight,
            )
        ]
        state = State(name='tps_probe', chains=[chain], energy_terms=energy_terms)
        system = System(states=[state])

        tps_cb = TPSCallback()
        with tempfile.TemporaryDirectory() as tmpdir:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', DeprecationWarning)
                minimizer = SimulatedAnnealing(
                    mutator=Canonical(n_mutations=1),
                    initial_temperature=self.initial_temperature,
                    final_temperature=self.final_temperature,
                    n_steps=self.n_steps,
                    log_path=Path(tmpdir),
                    callbacks=[tps_cb],
                )
            minimizer.minimize_system(system)

        return {
            'tps': tps_cb.tps(),
            'tps_auc': tps_cb.tps_auc(),
            'trajectory': tps_cb.trajectory,
            'final_energy': tps_cb.trajectory[-1] if tps_cb.trajectory else float('nan'),
        }
