"""
Aggregation risk signals for the BAGEL screening module.

Each signal returns a score in [0, 1] where higher values indicate greater
aggregation / prion-like risk.

MIT License

Copyright (c) 2025 Jakub Lála, Ayham Al-Saffar, Stefano Angioletti-Uberti
"""

from __future__ import annotations

import pathlib as pl
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

import bagel as bg
from bagel.oracles.embedding.base import EmbeddingOracle
from bagel.oracles.folding.base import FoldingOracle
from biotite.structure import annotate_sse

from .data.reference_sequences import REFERENCE_SEQUENCES

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------

_DATA_DIR = pl.Path(__file__).parent / 'data'
_REFERENCE_EMBEDDINGS_PATH = _DATA_DIR / 'reference_embeddings.npy'


def _sequence_to_chain(sequence: str, chain_id: str = 'A') -> bg.Chain:
    """Convert a bare amino-acid string to a :class:`bg.Chain`."""
    residues = [bg.Residue(name=aa, chain_ID=chain_id, index=i, mutable=True) for i, aa in enumerate(sequence)]
    return bg.Chain(residues=residues)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class AggregationSignal(ABC):
    """Abstract base for aggregation risk signals."""

    @abstractmethod
    def score(self, sequence: str) -> float:
        """Return risk score in [0, 1]. Higher = more risk."""
        ...


# ---------------------------------------------------------------------------
# Kyte-Doolittle hydrophobicity scale (per residue, one-letter code)
# ---------------------------------------------------------------------------

_KYTE_DOOLITTLE: dict[str, float] = {
    'A': 1.8,
    'R': -4.5,
    'N': -3.5,
    'D': -3.5,
    'C': 2.5,
    'Q': -3.5,
    'E': -3.5,
    'G': -0.4,
    'H': -3.2,
    'I': 4.5,
    'L': 3.8,
    'K': -3.9,
    'M': 1.9,
    'F': 2.8,
    'P': -1.6,
    'S': -0.8,
    'T': -0.7,
    'W': -0.9,
    'Y': -1.3,
    'V': 4.2,
}

# Theoretical maximum for a window of length 7 (all Ile = 4.5)
_KD_MAX_WINDOW = 7 * 4.5

# Known amyloid-forming steric-zipper hexapeptides (ZipperDB cores and WALTZ hits)
_AMYLOID_HEXAPEPTIDES: frozenset[str] = frozenset(
    [
        'VQIVYK',  # tau
        'NNQQNY',  # Sup35
        'GNNQQNY',  # Sup35 (7-mer included for substring matching)
        'VEALYL',  # insulin B
        'MVGGVV',  # Aβ
        'KLVFFA',  # Aβ
        'NFGAIL',  # IAPP
        'SSTNVG',  # IAPP
        'GGVVIA',  # Aβ42
        'STVIIE',  # TTR
        'YTIAALL',  # apoAI
        'IFQINS',  # lysozyme
        'IQRTPK',  # β2m
        'LYQLEN',  # β2m
        'SNQNNY',  # prion-like
        'AGGAIN',  # silkworm
        'IGFKVF',  # FUS
        'FGGSSG',  # TDP-43
        'QYQNQY',  # polyQ-adjacent
        'GGQQQT',  # poly-Q
    ]
)


# ---------------------------------------------------------------------------
# Signal 1: Sequence Homology
# ---------------------------------------------------------------------------


class SequenceHomologySignal(AggregationSignal):
    """
    Sequence-only signal based on optimal pairwise alignment against a corpus
    of known aggregating sequences (BLOSUM62).

    Score = max alignment score normalised by each reference self-alignment
    score, clipped to [0, 1].
    """

    def __init__(self) -> None:
        self._references = REFERENCE_SEQUENCES
        self._matrix = self._load_blosum62()
        self._self_scores = self._compute_self_scores()

    def _load_blosum62(self) -> object:
        """Load the BLOSUM62 substitution matrix from biotite."""
        from biotite.sequence.align import SubstitutionMatrix

        return SubstitutionMatrix.std_protein_matrix()

    def _compute_self_scores(self) -> list[float]:
        """Pre-compute self-alignment score for each reference sequence."""
        from biotite.sequence import ProteinSequence
        from biotite.sequence.align import align_optimal

        scores: list[float] = []
        for _, seq in self._references:
            ps = ProteinSequence(seq)
            alignments = align_optimal(ps, ps, self._matrix, terminal_penalty=False)
            scores.append(float(alignments[0].score))
        return scores

    def score(self, sequence: str) -> float:
        from biotite.sequence import ProteinSequence
        from biotite.sequence.align import align_optimal

        query = ProteinSequence(sequence)
        best = 0.0
        for i, (_, ref_seq) in enumerate(self._references):
            ref = ProteinSequence(ref_seq)
            alignments = align_optimal(query, ref, self._matrix, terminal_penalty=False)
            raw = float(alignments[0].score)
            self_score = self._self_scores[i]
            if self_score > 0:
                normalised = raw / self_score
                best = max(best, normalised)
        return float(np.clip(best, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Signal 2: Motif Detection
# ---------------------------------------------------------------------------


class MotifDetectionSignal(AggregationSignal):
    """
    Sequence-only signal that combines three sub-scores:

    1. Sliding-window Kyte-Doolittle hydrophobicity (window=7)
    2. Fraction of hexamers matching a curated amyloidogenic hexapeptide set
    3. Low-complexity / homopolymeric stretch length

    Final score = max(sub_score_1, sub_score_2, sub_score_3).
    """

    _WINDOW = 7
    _LC_THRESHOLD = 5  # runs >= this length count as low-complexity

    def score(self, sequence: str) -> float:
        if len(sequence) == 0:
            return 0.0
        s1 = self._hydrophobicity_score(sequence)
        s2 = self._hexapeptide_score(sequence)
        s3 = self._low_complexity_score(sequence)
        return float(max(s1, s2, s3))

    # -- sub-scores --

    def _hydrophobicity_score(self, sequence: str) -> float:
        if len(sequence) < self._WINDOW:
            values = [_KYTE_DOOLITTLE.get(aa, 0.0) for aa in sequence]
            window_max = sum(values) if sum(values) > 0 else 0.0
        else:
            kd_values = [_KYTE_DOOLITTLE.get(aa, 0.0) for aa in sequence]
            window_sums = [sum(kd_values[i : i + self._WINDOW]) for i in range(len(sequence) - self._WINDOW + 1)]
            window_max = max(window_sums)
        if _KD_MAX_WINDOW <= 0:
            return 0.0
        return float(np.clip(window_max / _KD_MAX_WINDOW, 0.0, 1.0))

    def _hexapeptide_score(self, sequence: str) -> float:
        if len(sequence) < 6:
            return 0.0
        n_hexamers = len(sequence) - 5
        hits = sum(1 for i in range(n_hexamers) if sequence[i : i + 6] in _AMYLOID_HEXAPEPTIDES)
        return float(hits / n_hexamers)

    def _low_complexity_score(self, sequence: str) -> float:
        if len(sequence) == 0:
            return 0.0
        longest_run = 1
        current_run = 1
        for i in range(1, len(sequence)):
            if sequence[i] == sequence[i - 1]:
                current_run += 1
                longest_run = max(longest_run, current_run)
            else:
                current_run = 1
        if longest_run < self._LC_THRESHOLD:
            return 0.0
        return float(np.clip(longest_run / len(sequence), 0.0, 1.0))


# ---------------------------------------------------------------------------
# Signal 3: Structural Propensity (oracle-dependent)
# ---------------------------------------------------------------------------


class StructuralPropensitySignal(AggregationSignal):
    """
    Oracle-dependent signal that uses ESMFold predictions to assess:
    - Beta-sheet fraction (proxy for amyloid propensity)
    - Low pLDDT (disordered regions associated with aggregation)

    score = beta_sheet_fraction * 0.6 + (1.0 - mean_pLDDT) * 0.4

    NOT used inside the LocalRobustnessProbe to keep robustness probing cheap.
    """

    def __init__(self, oracle: FoldingOracle) -> None:
        self.oracle = oracle

    def score(self, sequence: str) -> float:
        chain = _sequence_to_chain(sequence)
        result = self.oracle.predict([chain])
        structure = result.structure  # type: ignore[attr-defined]
        local_plddt = result.local_plddt  # type: ignore[attr-defined]

        sse_labels = annotate_sse(structure)
        beta_fraction = float(np.mean(sse_labels == 'b')) if len(sse_labels) > 0 else 0.0
        mean_plddt = float(np.mean(local_plddt[0])) if local_plddt.size > 0 else 0.0

        raw = beta_fraction * 0.6 + (1.0 - mean_plddt) * 0.4
        return float(np.clip(raw, 0.0, 1.0))


# ---------------------------------------------------------------------------
# Signal 4: Embedding Similarity (oracle-dependent)
# ---------------------------------------------------------------------------


class EmbeddingSignal(AggregationSignal):
    """
    Oracle-dependent signal that computes cosine similarity between the
    query sequence's mean ESM2 embedding and pre-computed reference embeddings
    from known aggregating sequences.

    score = 1.0 - max(cosine_similarity(query_emb, ref_emb_i))
            for i in reference rows

    NOT used inside the LocalRobustnessProbe to keep robustness probing cheap.
    """

    def __init__(
        self,
        oracle: EmbeddingOracle,
        reference_embeddings: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.oracle = oracle
        if reference_embeddings is None:
            if not _REFERENCE_EMBEDDINGS_PATH.exists():
                raise FileNotFoundError(
                    f'Reference embeddings not found at {_REFERENCE_EMBEDDINGS_PATH}. '
                    'Run scripts/generate_reference_embeddings.py to create them.'
                )
            reference_embeddings = np.load(_REFERENCE_EMBEDDINGS_PATH)
        norms = np.linalg.norm(reference_embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._reference_embeddings: npt.NDArray[np.float64] = reference_embeddings / norms

    def score(self, sequence: str) -> float:
        chain = _sequence_to_chain(sequence)
        result = self.oracle.predict([chain])
        embeddings: npt.NDArray[np.float64] = result.embeddings  # type: ignore[attr-defined]

        # mean-pool residue embeddings → shape (D,)
        query_emb = np.mean(embeddings, axis=0)
        norm = np.linalg.norm(query_emb)
        if norm == 0:
            return 0.5
        query_emb = query_emb / norm

        # cosine similarity against each reference row (already L2-normalised)
        similarities = self._reference_embeddings @ query_emb
        max_sim = float(np.max(similarities))
        raw = 1.0 - max_sim
        return float(np.clip(raw, 0.0, 1.0))
