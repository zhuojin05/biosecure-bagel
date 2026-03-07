"""Unit tests for BAGELPPScreener and pathogenic_potential helpers."""

from __future__ import annotations

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, array

import bagel as bg
from bagel.oracles.folding.base import FoldingResult
from bagel.screening.pathogenic_potential import (
    BAGELPPScreener,
    _compute_active_site_rmsd,
    _compute_rmsd,
    _get_template_sequence,
    _sequence_identity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHORT_SEQ = 'ACDEFGHIKL'  # 10 residues
SIMILAR_SEQ = 'ACDEFGHIKL'  # identical → blast_identity = 1.0
DISTANT_SEQ = 'MNPQRSTVWY'  # all different → low identity


def _make_ca_structure(n_residues: int, offset: float = 0.0) -> AtomArray:
    """CA-only AtomArray spread along x-axis."""
    atoms = [
        Atom(
            coord=[float(i) + offset, 0.0, 0.0],
            chain_id='A',
            res_id=i,
            res_name='ALA',
            atom_name='CA',
            element='C',
        )
        for i in range(n_residues)
    ]
    return array(atoms)


def _make_full_structure(n_residues: int, offset: float = 0.0) -> AtomArray:
    """N + CA + C + O backbone per residue."""
    atoms = []
    for i in range(n_residues):
        for atom_name, element in [('N', 'N'), ('CA', 'C'), ('C', 'C'), ('O', 'O')]:
            atoms.append(
                Atom(
                    coord=[float(i) + offset, 0.0, 0.0],
                    chain_id='A',
                    res_id=i,
                    res_name='ALA',
                    atom_name=atom_name,
                    element=element,
                )
            )
    return array(atoms)


# ---------------------------------------------------------------------------
# Helper function unit tests
# ---------------------------------------------------------------------------


class TestComputeRmsd:
    def test_identical_structures_zero_rmsd(self):
        struct = _make_ca_structure(5)
        assert _compute_rmsd(struct, struct) == pytest.approx(0.0, abs=1e-6)

    def test_different_shape_structure_nonzero_rmsd(self):
        """Structures with different shapes (not just translation) have nonzero RMSD after superimpose."""
        # Straight line vs bent: cannot be superimposed to zero
        atoms_straight = [
            Atom(coord=[float(i), 0.0, 0.0], chain_id='A', res_id=i, res_name='ALA', atom_name='CA', element='C')
            for i in range(5)
        ]
        atoms_bent = [
            Atom(coord=[float(i), float(i), 0.0], chain_id='A', res_id=i, res_name='ALA', atom_name='CA', element='C')
            for i in range(5)
        ]
        q = array(atoms_straight)
        t = array(atoms_bent)
        rmsd = _compute_rmsd(q, t)
        assert rmsd > 0.0

    def test_empty_structure_returns_zero(self):
        empty = AtomArray(0)
        assert _compute_rmsd(empty, empty) == 0.0


class TestComputeActiveSiteRmsd:
    def test_identical_structures_zero_active_rmsd(self):
        struct = _make_ca_structure(5)
        rmsd = _compute_active_site_rmsd(struct, struct, [0, 2, 4])
        assert rmsd == pytest.approx(0.0, abs=1e-6)

    def test_out_of_range_positions_returns_zero(self):
        struct = _make_ca_structure(3)
        assert _compute_active_site_rmsd(struct, struct, [10, 20]) == 0.0

    def test_empty_positions_returns_zero(self):
        struct = _make_ca_structure(5)
        assert _compute_active_site_rmsd(struct, struct, []) == 0.0


class TestSequenceIdentity:
    def test_identical_sequences_returns_one(self):
        assert _sequence_identity('ACDEF', 'ACDEF') == pytest.approx(1.0)

    def test_empty_sequences_return_zero(self):
        assert _sequence_identity('', '') == 0.0
        assert _sequence_identity('ACDEF', '') == 0.0

    def test_different_sequences_less_than_one(self):
        identity = _sequence_identity('ACDEF', 'MNPQR')
        assert identity < 1.0

    def test_returns_float_in_range(self):
        identity = _sequence_identity('ACDEFG', 'ACMNPQ')
        assert 0.0 <= identity <= 1.0


class TestGetTemplateSequence:
    def test_extracts_correct_length(self):
        struct = _make_ca_structure(5)
        seq = _get_template_sequence(struct)
        assert len(seq) == 5

    def test_extracts_alanine_for_ala_residues(self):
        struct = _make_ca_structure(3)  # res_name='ALA'
        seq = _get_template_sequence(struct)
        assert seq == 'AAA'


# ---------------------------------------------------------------------------
# BAGELPPScreener tests
# ---------------------------------------------------------------------------


class TestCalculatePathogenicPotential:
    def test_calculate_pp_returns_required_keys(self, fake_esmfold):
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[0, 2, 4],
            oracle=fake_esmfold,
        )
        result = screener.calculate_pathogenic_potential(SHORT_SEQ)
        expected_keys = {
            'pp_score',
            'tm_score_proxy',
            'function_retained',
            'blast_identity',
            'evasion_factor',
            'evades_blast',
            'above_efsa',
        }
        assert set(result.keys()) == expected_keys

    def test_identical_structure_max_tm_score_proxy(self, fake_esmfold):
        """When RMSD=0, tm_score_proxy=1.0."""
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[0],
            oracle=fake_esmfold,
        )
        with patch('bagel.screening.pathogenic_potential._compute_rmsd', return_value=0.0):
            result = screener.calculate_pathogenic_potential(SHORT_SEQ)
        assert result['tm_score_proxy'] == pytest.approx(1.0, abs=1e-6)

    def test_high_blast_identity_no_evasion(self, fake_esmfold):
        """Sequence identical to template sequence → identity ≥ 0.5 → evasion_factor=0."""
        # Template is ALA×N; SHORT_SEQ won't be identical but we test logic via mock
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[],
            oracle=fake_esmfold,
        )
        # Patch _sequence_identity to return 0.8
        with patch(
            'bagel.screening.pathogenic_potential._sequence_identity', return_value=0.8
        ):
            result = screener.calculate_pathogenic_potential(SHORT_SEQ)
        assert result['evasion_factor'] == pytest.approx(0.0)
        assert result['evades_blast'] is False

    def test_low_blast_identity_positive_evasion(self, fake_esmfold):
        """Low identity (< 0.5) → evasion_factor > 0."""
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[],
            oracle=fake_esmfold,
        )
        with patch(
            'bagel.screening.pathogenic_potential._sequence_identity', return_value=0.1
        ):
            result = screener.calculate_pathogenic_potential(SHORT_SEQ)
        assert result['evasion_factor'] > 0.0
        assert result['evades_blast'] is True

    def test_efsa_flag_requires_both_thresholds(self, fake_esmfold):
        """above_efsa requires tm_score_proxy > 0.8 AND function_retained > 0.7."""
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[],
            oracle=fake_esmfold,
        )

        with (
            patch('bagel.screening.pathogenic_potential._compute_rmsd', return_value=0.1),
            patch(
                'bagel.screening.pathogenic_potential._compute_active_site_rmsd',
                return_value=0.1,
            ),
            patch(
                'bagel.screening.pathogenic_potential._sequence_identity', return_value=0.9
            ),
        ):
            result = screener.calculate_pathogenic_potential(SHORT_SEQ)
        # RMSD=0.1 → tm=1/(1+0.1)≈0.91 > 0.8; active_rmsd=0.1 → exp(-0.1)≈0.90 > 0.7
        assert result['above_efsa'] is True

        # Only tm > 0.8 but function NOT > 0.7
        with (
            patch('bagel.screening.pathogenic_potential._compute_rmsd', return_value=0.1),
            patch(
                'bagel.screening.pathogenic_potential._compute_active_site_rmsd',
                return_value=5.0,
            ),
            patch(
                'bagel.screening.pathogenic_potential._sequence_identity', return_value=0.9
            ),
        ):
            result2 = screener.calculate_pathogenic_potential(SHORT_SEQ)
        assert result2['above_efsa'] is False

    def test_pp_score_above_one_for_similar_evading_sequence(self, fake_esmfold):
        """PP_structural > 1 when structurally similar and evading BLAST."""
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[],
            oracle=fake_esmfold,
        )
        with (
            patch('bagel.screening.pathogenic_potential._compute_rmsd', return_value=0.0),
            patch(
                'bagel.screening.pathogenic_potential._compute_active_site_rmsd',
                return_value=0.0,
            ),
            patch(
                'bagel.screening.pathogenic_potential._sequence_identity', return_value=0.0
            ),
        ):
            result = screener.calculate_pathogenic_potential(SHORT_SEQ)
        # tm=1, function=1, evasion=(0.5-0)*2=1.0 → pp=1*1*(1+1)=2.0
        assert result['pp_score'] == pytest.approx(2.0, rel=1e-3)


class TestTemplateLoading:
    def test_template_atomarray_no_fold_call(self, fake_esmfold):
        """When template is AtomArray, oracle.predict should NOT be called for template."""
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[],
            oracle=fake_esmfold,
        )
        original_predict = fake_esmfold.predict
        call_count_before = [0]

        def counting_predict(chains):
            call_count_before[0] += 1
            return original_predict(chains)

        fake_esmfold.predict = counting_predict
        _ = screener._get_template_atoms()
        assert call_count_before[0] == 0  # no oracle call for AtomArray template
        fake_esmfold.predict = original_predict

    def test_template_sequence_folds_lazily(self, fake_esmfold):
        """When template is a string, oracle is called exactly once on first access."""
        template_seq = SHORT_SEQ
        screener = BAGELPPScreener(
            toxin_template=template_seq,
            active_site_positions=[],
            oracle=fake_esmfold,
        )
        original_predict = fake_esmfold.predict
        call_count = [0]

        def counting_predict(chains):
            call_count[0] += 1
            return original_predict(chains)

        fake_esmfold.predict = counting_predict

        # First access — should call oracle once
        _ = screener._get_template_atoms()
        assert call_count[0] == 1

        # Second access — cached, no extra call
        _ = screener._get_template_atoms()
        assert call_count[0] == 1

        fake_esmfold.predict = original_predict


class TestRiskAssessment:
    def test_risk_assessment_returns_required_keys(self, fake_esmfold):
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[0, 2],
            oracle=fake_esmfold,
            n_steps=3,
        )
        result = screener.risk_assessment(SHORT_SEQ, n_variants=3)
        expected_keys = {
            'tps_auc',
            'tps_trajectory',
            'query_pp',
            'max_pp',
            'evasion_efficiency',
            'efsa_trigger_count',
            'pp_distribution',
            'risk_category',
        }
        assert set(result.keys()) == expected_keys

    def test_risk_assessment_pp_distribution_length(self, fake_esmfold):
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[],
            oracle=fake_esmfold,
            n_steps=3,
        )
        n_variants = 5
        result = screener.risk_assessment(SHORT_SEQ, n_variants=n_variants)
        assert len(result['pp_distribution']) == n_variants

    def test_risk_category_is_valid_value(self, fake_esmfold):
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[],
            oracle=fake_esmfold,
            n_steps=3,
        )
        result = screener.risk_assessment(SHORT_SEQ, n_variants=3)
        assert result['risk_category'] in ('HIGH', 'MEDIUM', 'LOW')

    def test_evasion_efficiency_in_range(self, fake_esmfold):
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[],
            oracle=fake_esmfold,
            n_steps=3,
        )
        result = screener.risk_assessment(SHORT_SEQ, n_variants=5)
        assert 0.0 <= result['evasion_efficiency'] <= 1.0

    def test_max_pp_geq_query_pp(self, fake_esmfold):
        template = _make_ca_structure(len(SHORT_SEQ))
        screener = BAGELPPScreener(
            toxin_template=template,
            active_site_positions=[],
            oracle=fake_esmfold,
            n_steps=3,
        )
        result = screener.risk_assessment(SHORT_SEQ, n_variants=3)
        assert result['max_pp'] >= result['query_pp']['pp_score']
