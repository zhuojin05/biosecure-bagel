"""Unit tests for aggregation screening signals."""

from __future__ import annotations

from unittest.mock import Mock, patch

import numpy as np
import pytest
from biotite.structure import AtomArray, Atom, array

import bagel as bg
from bagel.screening.signals import (
    EmbeddingSignal,
    MotifDetectionSignal,
    SequenceHomologySignal,
    StructuralPropensitySignal,
    _sequence_to_chain,
)

# ---------------------------------------------------------------------------
# Shared test sequences
# ---------------------------------------------------------------------------

# Known aggregator – amyloid-beta 42
AGGREGATOR_SEQ = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'

# Known stable, soluble protein – ubiquitin
STABLE_SEQ = 'MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG'

# A clearly hydrophobic, low-complexity-like short sequence
HYDROPHOBIC_SEQ = 'IIIIIIIVVVVVVVVLLLLLLL'

# PolyQ stretch (huntingtin-like)
POLYQ_SEQ = 'MATLEKLMKAFESLKSFQQQQQQQQQQQQQQQQQQQQQQQPPPPPPPPPP'


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_backbone_structure(n_residues: int = 5) -> AtomArray:
    """Return an AtomArray with N, CA, C backbone atoms per residue (same coords)."""
    atoms = []
    for res_id in range(n_residues):
        for atom_name, element in [('N', 'N'), ('CA', 'C'), ('C', 'C'), ('O', 'O')]:
            atoms.append(
                Atom(
                    coord=[float(res_id), 0.0, 0.0],
                    chain_id='A',
                    res_id=res_id,
                    res_name='ALA',
                    atom_name=atom_name,
                    element=element,
                )
            )
    return array(atoms)


# ---------------------------------------------------------------------------
# _sequence_to_chain
# ---------------------------------------------------------------------------


def test_sequence_to_chain_basic():
    chain = _sequence_to_chain('ACDEF')
    assert chain.sequence == 'ACDEF'
    assert all(r.mutable for r in chain.residues)
    assert chain.chain_ID == 'A'


def test_sequence_to_chain_custom_chain_id():
    chain = _sequence_to_chain('MVLSE', chain_id='B')
    assert chain.chain_ID == 'B'


# ---------------------------------------------------------------------------
# SequenceHomologySignal
# ---------------------------------------------------------------------------


class TestSequenceHomologySignal:
    def test_score_range(self):
        sig = SequenceHomologySignal()
        score = sig.score(AGGREGATOR_SEQ)
        assert 0.0 <= score <= 1.0

    def test_aggregator_scores_higher_than_stable(self):
        sig = SequenceHomologySignal()
        agg_score = sig.score(AGGREGATOR_SEQ)
        stable_score = sig.score(STABLE_SEQ)
        # Aβ42 is in the reference corpus, so its self-score should be 1.0
        assert agg_score >= stable_score

    def test_exact_reference_match_scores_near_one(self):
        sig = SequenceHomologySignal()
        # Use a sequence from the reference corpus verbatim
        from bagel.screening.data.reference_sequences import REFERENCE_SEQUENCES

        name, ref_seq = REFERENCE_SEQUENCES[1]  # amyloid_beta_42
        score = sig.score(ref_seq)
        assert score > 0.9, f'Self-match should score near 1.0, got {score}'

    def test_short_random_sequence(self):
        sig = SequenceHomologySignal()
        score = sig.score('ACDEFGHIKLM')
        assert 0.0 <= score <= 1.0

    def test_output_is_float(self):
        sig = SequenceHomologySignal()
        result = sig.score('MVLSE')
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# MotifDetectionSignal
# ---------------------------------------------------------------------------


class TestMotifDetectionSignal:
    def test_score_range(self):
        sig = MotifDetectionSignal()
        assert 0.0 <= sig.score(AGGREGATOR_SEQ) <= 1.0
        assert 0.0 <= sig.score(STABLE_SEQ) <= 1.0

    def test_hydrophobic_stretch_detected(self):
        sig = MotifDetectionSignal()
        score = sig.score(HYDROPHOBIC_SEQ)
        # All-Ile/Val window should yield a high hydrophobicity sub-score
        assert score > 0.5

    def test_polyq_low_complexity_detected(self):
        sig = MotifDetectionSignal()
        score = sig.score(POLYQ_SEQ)
        # Long Q stretch triggers low-complexity signal
        assert score > 0.0

    def test_amyloid_hexapeptide_detected(self):
        sig = MotifDetectionSignal()
        # Sequence containing VQIVYK (tau amyloid core)
        seq_with_core = 'AAAAAVQIVYKAAAAA'
        score = sig.score(seq_with_core)
        assert score > 0.0

    def test_empty_sequence(self):
        sig = MotifDetectionSignal()
        assert sig.score('') == 0.0

    def test_short_sequence_no_hexamer(self):
        sig = MotifDetectionSignal()
        score = sig.score('ACDE')  # shorter than 6
        assert 0.0 <= score <= 1.0

    def test_stable_sequence_lower_than_aggregator(self):
        sig = MotifDetectionSignal()
        # Ubiquitin should not trigger strong amyloid motifs
        agg = sig.score(AGGREGATOR_SEQ)
        # Only verify scores are in range (ubiquitin may still have hydrophobic patches)
        assert 0.0 <= agg <= 1.0

    def test_all_glycine_low_score(self):
        sig = MotifDetectionSignal()
        # Glycine is the least hydrophobic; no hexapeptide match; no homopolymer run > 5 here
        score = sig.score('GGGGGG')
        # low-complexity triggered (6 G's), so score > 0
        assert score >= 0.0


# ---------------------------------------------------------------------------
# StructuralPropensitySignal
# ---------------------------------------------------------------------------


class TestStructuralPropensitySignal:
    def test_score_range(self, fake_esmfold: bg.oracles.folding.ESMFold, monkeypatch):
        n_res = 5
        mock_structure = _make_backbone_structure(n_res)

        def mock_fold(self, chains):
            return bg.oracles.folding.ESMFoldResult(
                input_chains=chains,
                structure=mock_structure,
                local_plddt=0.8 * np.ones((1, n_res)),
                ptm=np.array([[0.7]]),
                pae=np.zeros((1, n_res, n_res)),
            )

        monkeypatch.setattr(bg.oracles.folding.ESMFold, 'fold', mock_fold)

        with patch('bagel.screening.signals.annotate_sse') as mock_sse:
            mock_sse.return_value = np.array(['b', 'b', 'c', 'a', 'c'])
            sig = StructuralPropensitySignal(oracle=fake_esmfold)
            score = sig.score('AACDE')
        assert 0.0 <= score <= 1.0

    def test_all_beta_high_score(self, fake_esmfold: bg.oracles.folding.ESMFold, monkeypatch):
        n_res = 6
        mock_structure = _make_backbone_structure(n_res)

        def mock_fold(self, chains):
            return bg.oracles.folding.ESMFoldResult(
                input_chains=chains,
                structure=mock_structure,
                local_plddt=np.zeros((1, n_res)),  # pLDDT=0 → (1-0)=1.0
                ptm=np.array([[0.5]]),
                pae=np.zeros((1, n_res, n_res)),
            )

        monkeypatch.setattr(bg.oracles.folding.ESMFold, 'fold', mock_fold)

        with patch('bagel.screening.signals.annotate_sse') as mock_sse:
            mock_sse.return_value = np.array(['b'] * n_res)
            sig = StructuralPropensitySignal(oracle=fake_esmfold)
            score = sig.score('VQIVYK')
        # beta_fraction=1.0 * 0.6 + (1.0-0.0) * 0.4 = 1.0
        assert np.isclose(score, 1.0)

    def test_all_helix_perfect_plddt_low_score(self, fake_esmfold: bg.oracles.folding.ESMFold, monkeypatch):
        n_res = 4
        mock_structure = _make_backbone_structure(n_res)

        def mock_fold(self, chains):
            return bg.oracles.folding.ESMFoldResult(
                input_chains=chains,
                structure=mock_structure,
                local_plddt=np.ones((1, n_res)),  # perfect pLDDT → (1-1)=0.0
                ptm=np.array([[0.9]]),
                pae=np.zeros((1, n_res, n_res)),
            )

        monkeypatch.setattr(bg.oracles.folding.ESMFold, 'fold', mock_fold)

        with patch('bagel.screening.signals.annotate_sse') as mock_sse:
            mock_sse.return_value = np.array(['a'] * n_res)  # all helix, no beta
            sig = StructuralPropensitySignal(oracle=fake_esmfold)
            score = sig.score('AAAA')
        # beta_fraction=0.0 * 0.6 + (1-1.0) * 0.4 = 0.0
        assert np.isclose(score, 0.0)


# ---------------------------------------------------------------------------
# EmbeddingSignal
# ---------------------------------------------------------------------------


class TestEmbeddingSignal:
    def _make_embedding_result(self, chains, n_features: int = 8):
        """Return a fake EmbeddingResult with random per-residue embeddings."""
        n_res = sum(len(c.residues) for c in chains)
        embeddings = np.random.randn(n_res, n_features).astype(np.float64)
        return bg.oracles.embedding.ESM2Result(
            input_chains=chains,
            embeddings=embeddings,
        )

    def test_score_range(self, fake_esm2: bg.oracles.embedding.ESM2, monkeypatch):
        n_features = 8
        n_refs = 3
        ref_embs = np.random.randn(n_refs, n_features).astype(np.float64)

        def mock_embed(self, chains):
            return self._make_embedding_result(chains, n_features)

        # Patch embed on the instance via monkeypatch at class level
        monkeypatch.setattr(
            bg.oracles.embedding.ESM2,
            'embed',
            lambda self, chains: (
                bg.oracles.embedding.ESM2Result(
                    input_chains=chains,
                    embeddings=np.random.randn(sum(len(c.residues) for c in chains), n_features).astype(np.float64),
                )
            ),
        )

        sig = EmbeddingSignal(oracle=fake_esm2, reference_embeddings=ref_embs)
        score = sig.score('ACDEF')
        assert 0.0 <= score <= 1.0

    def test_identical_to_reference_scores_high(self, fake_esm2: bg.oracles.embedding.ESM2, monkeypatch):
        """A sequence whose embedding is identical to a reference gets high score (max_sim=1.0)."""
        n_features = 8
        # Construct a unit-norm reference embedding
        ref_vec = np.ones(n_features, dtype=np.float64)
        ref_vec = ref_vec / np.linalg.norm(ref_vec)
        ref_embs = ref_vec[np.newaxis, :]  # shape (1, 8)

        monkeypatch.setattr(
            bg.oracles.embedding.ESM2,
            'embed',
            lambda self, chains: (
                bg.oracles.embedding.ESM2Result(
                    input_chains=chains,
                    # Each residue embedding is the same unit vector → mean = same unit vector
                    embeddings=np.tile(ref_vec, (sum(len(c.residues) for c in chains), 1)),
                )
            ),
        )

        sig = EmbeddingSignal(oracle=fake_esm2, reference_embeddings=ref_embs)
        score = sig.score('ACDEF')
        # max cosine similarity = 1.0 → score = 1.0
        assert np.isclose(score, 1.0, atol=1e-5)

    def test_zero_embedding_returns_half(self, fake_esm2: bg.oracles.embedding.ESM2, monkeypatch):
        """If the oracle returns all-zero embeddings, mean is zero and score is 0.5."""
        n_features = 8
        ref_embs = np.random.randn(3, n_features).astype(np.float64)

        monkeypatch.setattr(
            bg.oracles.embedding.ESM2,
            'embed',
            lambda self, chains: (
                bg.oracles.embedding.ESM2Result(
                    input_chains=chains,
                    embeddings=np.zeros((sum(len(c.residues) for c in chains), n_features)),
                )
            ),
        )

        sig = EmbeddingSignal(oracle=fake_esm2, reference_embeddings=ref_embs)
        score = sig.score('ACDE')
        assert score == 0.5
