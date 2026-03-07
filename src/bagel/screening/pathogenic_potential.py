"""
BAGEL-PP: Pathogenic Potential Assessment Engine.

Implements modified Casadevall PP_structural scoring and full risk assessment
combining structural similarity, functional conservation, sequence evasion
potential, and landscape-geometry TPS walk scores.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from __future__ import annotations

import numpy as np
from biotite.structure import AtomArray, superimpose

from bagel.oracles.folding.base import FoldingOracle
from bagel.oracles.folding.utils import sequence_from_atomarray

from .robustness import LocalRobustnessProbe
from .signals import _sequence_to_chain
from .tps import ToxinProximityWalk


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_rmsd(query: AtomArray, template: AtomArray) -> float:
    """CA-only RMSD between query and template after superimposition."""
    query_ca = query[query.atom_name == 'CA']
    template_ca = template[template.atom_name == 'CA']

    n_q = len(query_ca)
    n_t = len(template_ca)

    if n_q == 0 or n_t == 0:
        return 0.0

    # Truncate to the shorter length so dimensions match
    n = min(n_q, n_t)
    query_ca = query_ca[:n]
    template_ca = template_ca[:n]

    if n < 3:
        # superimpose requires >= 3 atoms; fall back to raw RMSD
        diffs = query_ca.coord - template_ca.coord
        return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))

    superimposed, _ = superimpose(fixed=query_ca, mobile=template_ca)
    diffs = query_ca.coord - superimposed.coord
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))


def _compute_active_site_rmsd(
    query: AtomArray, template: AtomArray, positions: list[int]
) -> float:
    """CA-only RMSD restricted to active_site_positions residues (0-based index)."""
    query_ca = query[query.atom_name == 'CA']
    template_ca = template[template.atom_name == 'CA']

    n = min(len(query_ca), len(template_ca))
    valid = [p for p in positions if p < n]

    if not valid:
        return 0.0

    q_coords = query_ca.coord[valid]
    t_coords = template_ca.coord[valid]
    diffs = q_coords - t_coords
    return float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))))


def _sequence_identity(seq_a: str, seq_b: str) -> float:
    """
    Pairwise sequence identity from BLOSUM62 optimal alignment.

    Returns the fraction of aligned positions that are identical,
    using the same biotite pattern as :class:`SequenceHomologySignal`.
    """
    from biotite.sequence import ProteinSequence
    from biotite.sequence.align import SubstitutionMatrix, align_optimal

    if not seq_a or not seq_b:
        return 0.0

    matrix = SubstitutionMatrix.std_protein_matrix()
    ps_a = ProteinSequence(seq_a)
    ps_b = ProteinSequence(seq_b)
    alignments = align_optimal(ps_a, ps_b, matrix, terminal_penalty=False)
    alignment = alignments[0]

    trace = alignment.trace
    identical = sum(
        1
        for i, j in trace
        if i >= 0 and j >= 0 and seq_a[i] == seq_b[j]
    )
    aligned = sum(1 for i, j in trace if i >= 0 and j >= 0)
    if aligned == 0:
        return 0.0
    return identical / aligned


def _get_template_sequence(atoms: AtomArray) -> str:
    """Extract amino-acid sequence from an AtomArray (via CA residues)."""
    return sequence_from_atomarray(atoms)


# ---------------------------------------------------------------------------
# BAGELPPScreener
# ---------------------------------------------------------------------------


class BAGELPPScreener:
    """
    Pathogenic Potential screener combining structural similarity, functional
    conservation, sequence evasion potential, and landscape-geometry (TPS).

    Parameters
    ----------
    toxin_template : AtomArray or str
        Known toxin template — either a pre-folded :class:`AtomArray` or an
        amino-acid sequence string (folded lazily on first use).
    active_site_positions : list[int]
        0-based residue indices defining the functional active site.
    oracle : FoldingOracle
        Folding oracle for structure prediction.
    n_steps : int
        MC steps for the TPS walk (used in :meth:`risk_assessment`).
    initial_temperature : float
        Starting temperature for the TPS walk.
    final_temperature : float
        Final temperature for the TPS walk.
    """

    def __init__(
        self,
        toxin_template: AtomArray | str,
        active_site_positions: list[int],
        oracle: FoldingOracle,
        n_steps: int = 500,
        initial_temperature: float = 1.0,
        final_temperature: float = 0.01,
    ) -> None:
        self._template_input = toxin_template
        self._template_atoms: AtomArray | None = None
        self.active_site_positions = active_site_positions
        self.oracle = oracle
        self.n_steps = n_steps
        self.initial_temperature = initial_temperature
        self.final_temperature = final_temperature
        self._tps_walk: ToxinProximityWalk | None = None

    def _get_template_atoms(self) -> AtomArray:
        """Return cached template AtomArray, folding lazily if input was a sequence."""
        if self._template_atoms is None:
            if isinstance(self._template_input, str):
                chain = _sequence_to_chain(self._template_input)
                result = self.oracle.predict([chain])
                self._template_atoms = result.structure
            else:
                self._template_atoms = self._template_input
        return self._template_atoms

    def _get_tps_walk(self) -> ToxinProximityWalk:
        """Return cached ToxinProximityWalk (built after template is loaded)."""
        if self._tps_walk is None:
            self._tps_walk = ToxinProximityWalk(
                oracle=self.oracle,
                template_atoms=self._get_template_atoms(),
                n_steps=self.n_steps,
                initial_temperature=self.initial_temperature,
                final_temperature=self.final_temperature,
            )
        return self._tps_walk

    def calculate_pathogenic_potential(self, sequence: str) -> dict:
        """
        Single-sequence PP_structural score.

        No MC walk — fast, can run without GPU if oracle is mocked.

        PP_structural formula
        ---------------------
        ::

            tm_score_proxy    = 1.0 / (1.0 + RMSD_to_template)
            function_retained = exp(-active_site_RMSD)
            evasion_factor    = max(0, (0.5 - blast_identity) * 2)
            PP_structural     = tm_score_proxy * function_retained * (1 + evasion_factor)

        Returns
        -------
        dict
            - ``pp_score``          : float — PP_structural in [0, ~2]
            - ``tm_score_proxy``    : float — in (0, 1]
            - ``function_retained`` : float — in (0, 1]
            - ``blast_identity``    : float — fraction of identical aligned residues
            - ``evasion_factor``    : float — >0 only if identity < 50 %
            - ``evades_blast``      : bool  — identity < 50 %
            - ``above_efsa``        : bool  — tm > 0.8 AND function > 0.7
        """
        template_atoms = self._get_template_atoms()

        # 1. Fold query
        chain = _sequence_to_chain(sequence)
        result = self.oracle.predict([chain])
        query_atoms = result.structure

        # 2. Overall RMSD → TM-score proxy
        rmsd_overall = _compute_rmsd(query_atoms, template_atoms)
        tm_score_proxy = 1.0 / (1.0 + rmsd_overall)

        # 3. Active-site RMSD → function retained
        rmsd_active = _compute_active_site_rmsd(
            query_atoms, template_atoms, self.active_site_positions
        )
        function_retained = float(np.exp(-rmsd_active))

        # 4. Sequence identity → evasion factor
        template_seq = _get_template_sequence(template_atoms)
        blast_identity = _sequence_identity(sequence, template_seq)
        evasion_factor = max(0.0, (0.5 - blast_identity) * 2.0)

        pp_score = tm_score_proxy * function_retained * (1.0 + evasion_factor)

        return {
            'pp_score': pp_score,
            'tm_score_proxy': tm_score_proxy,
            'function_retained': function_retained,
            'blast_identity': blast_identity,
            'evasion_factor': evasion_factor,
            'evades_blast': blast_identity < 0.5,
            'above_efsa': tm_score_proxy > 0.8 and function_retained > 0.7,
        }

    def risk_assessment(self, query_sequence: str, n_variants: int = 30) -> dict:
        """
        Full PP assessment combining TPS landscape walk and PP_structural scores.

        Steps
        -----
        1. Run TPS MC walk (landscape geometry) → ``tps_auc``.
        2. Compute PP_structural for the query sequence.
        3. Generate ``n_variants`` local single-residue variants and score each.
        4. Aggregate distribution metrics and assign risk category.

        Returns
        -------
        dict
            - ``tps_auc``             : float
            - ``tps_trajectory``      : list[float]
            - ``query_pp``            : dict (from :meth:`calculate_pathogenic_potential`)
            - ``max_pp``              : float — maximum PP across query + variants
            - ``evasion_efficiency``  : float — fraction of variants with identity < 50 %
            - ``efsa_trigger_count``  : int
            - ``pp_distribution``     : list[float] — PP scores for all variants
            - ``risk_category``       : 'HIGH' | 'MEDIUM' | 'LOW'
        """
        # 1. TPS walk
        tps_walk = self._get_tps_walk()
        tps_result = tps_walk.run(query_sequence)

        # 2. Query PP
        query_pp = self.calculate_pathogenic_potential(query_sequence)

        # 3. Local variants
        probe = LocalRobustnessProbe(n_mutations=1)
        variants = probe.generate_local_variants(query_sequence, n_variants=n_variants)
        variant_pps = [self.calculate_pathogenic_potential(v) for v in variants]

        # 4. Aggregate
        pp_distribution = [v['pp_score'] for v in variant_pps]
        all_pp_scores = [query_pp['pp_score']] + pp_distribution
        max_pp = float(max(all_pp_scores))

        evasion_count = sum(1 for v in variant_pps if v['evades_blast'])
        evasion_efficiency = evasion_count / n_variants if n_variants > 0 else 0.0

        efsa_trigger_count = sum(1 for v in variant_pps if v['above_efsa'])
        if query_pp['above_efsa']:
            efsa_trigger_count += 1

        # Risk category
        if max_pp > 1.0 or tps_result['tps_auc'] > 0.5:
            risk_category = 'HIGH'
        elif max_pp > 0.5 or efsa_trigger_count > 0:
            risk_category = 'MEDIUM'
        else:
            risk_category = 'LOW'

        return {
            'tps_auc': tps_result['tps_auc'],
            'tps_trajectory': tps_result['trajectory'],
            'query_pp': query_pp,
            'max_pp': max_pp,
            'evasion_efficiency': evasion_efficiency,
            'efsa_trigger_count': efsa_trigger_count,
            'pp_distribution': pp_distribution,
            'risk_category': risk_category,
        }
