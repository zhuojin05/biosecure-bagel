"""Unit tests for TPSCallback and ToxinProximityWalk."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from biotite.structure import Atom, AtomArray, array

import bagel as bg
from bagel.screening.tps import TPSCallback, ToxinProximityWalk


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

TEST_SEQ = 'ACDEFGHIKL'


def _make_ca_structure(n_residues: int, offset: float = 0.0) -> AtomArray:
    """CA-only AtomArray with coords spread along x-axis."""
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


def _make_callback_context(energy: float) -> MagicMock:
    ctx = MagicMock()
    ctx.metrics = {'system_energy': energy}
    return ctx


# ---------------------------------------------------------------------------
# TPSCallback unit tests
# ---------------------------------------------------------------------------


class TestTPSCallback:
    def test_tps_callback_records_trajectory(self):
        cb = TPSCallback()
        for e in [5.0, 4.0, 3.0]:
            cb.on_step_end(_make_callback_context(e))
        assert cb.trajectory == [5.0, 4.0, 3.0]

    def test_trajectory_grows_at_each_step(self):
        cb = TPSCallback()
        for i, e in enumerate([10.0, 8.0, 6.0, 4.0]):
            cb.on_step_end(_make_callback_context(e))
            assert len(cb.trajectory) == i + 1

    def test_tps_flat_trajectory_scores_zero(self):
        cb = TPSCallback()
        for _ in range(5):
            cb.on_step_end(_make_callback_context(3.0))
        assert cb.tps() == pytest.approx(0.0)

    def test_tps_drop_scores_positive(self):
        cb = TPSCallback()
        cb.on_step_end(_make_callback_context(4.0))
        cb.on_step_end(_make_callback_context(2.0))
        assert cb.tps() > 0.0

    def test_tps_full_drop_scores_one(self):
        cb = TPSCallback()
        cb.on_step_end(_make_callback_context(4.0))
        cb.on_step_end(_make_callback_context(0.0))
        assert cb.tps() == pytest.approx(1.0)

    def test_tps_zero_initial_energy_returns_zero(self):
        cb = TPSCallback()
        cb.on_step_end(_make_callback_context(0.0))
        cb.on_step_end(_make_callback_context(0.0))
        assert cb.tps() == 0.0

    def test_tps_empty_trajectory_returns_zero(self):
        cb = TPSCallback()
        assert cb.tps() == 0.0
        assert cb.tps_auc() == 0.0

    def test_tps_auc_flat_trajectory_scores_zero(self):
        cb = TPSCallback()
        for _ in range(5):
            cb.on_step_end(_make_callback_context(2.0))
        # sum(e/e0) = N, so 1 - N/N = 0
        assert cb.tps_auc() == pytest.approx(0.0)

    def test_tps_auc_faster_drop_scores_higher(self):
        """Early energy drop should give higher AUC than late drop."""
        # Early drop: energy falls immediately
        cb_early = TPSCallback()
        cb_early.on_step_end(_make_callback_context(4.0))
        cb_early.on_step_end(_make_callback_context(1.0))
        cb_early.on_step_end(_make_callback_context(1.0))
        cb_early.on_step_end(_make_callback_context(1.0))

        # Late drop: energy falls at the end
        cb_late = TPSCallback()
        cb_late.on_step_end(_make_callback_context(4.0))
        cb_late.on_step_end(_make_callback_context(4.0))
        cb_late.on_step_end(_make_callback_context(4.0))
        cb_late.on_step_end(_make_callback_context(1.0))

        assert cb_early.tps_auc() > cb_late.tps_auc()

    def test_tps_auc_in_range(self):
        cb = TPSCallback()
        energies = [10.0, 8.0, 6.0, 4.0, 2.0]
        for e in energies:
            cb.on_step_end(_make_callback_context(e))
        auc = cb.tps_auc()
        assert 0.0 <= auc <= 1.0

    def test_tps_auc_zero_initial_energy_returns_zero(self):
        cb = TPSCallback()
        cb.on_step_end(_make_callback_context(0.0))
        assert cb.tps_auc() == 0.0


# ---------------------------------------------------------------------------
# ToxinProximityWalk integration tests (oracle mocked via fake_esmfold)
# ---------------------------------------------------------------------------


class TestToxinProximityWalk:
    def test_toxin_proximity_walk_returns_required_keys(self, fake_esmfold):
        template = _make_ca_structure(len(TEST_SEQ))
        walk = ToxinProximityWalk(
            oracle=fake_esmfold,
            template_atoms=template,
            n_steps=5,
            initial_temperature=1.0,
            final_temperature=0.1,
        )
        result = walk.run(TEST_SEQ)
        assert set(result.keys()) == {'tps', 'tps_auc', 'trajectory', 'final_energy'}

    def test_toxin_proximity_walk_trajectory_length(self, fake_esmfold):
        n_steps = 7
        template = _make_ca_structure(len(TEST_SEQ))
        walk = ToxinProximityWalk(
            oracle=fake_esmfold,
            template_atoms=template,
            n_steps=n_steps,
        )
        result = walk.run(TEST_SEQ)
        assert len(result['trajectory']) == n_steps

    def test_toxin_proximity_walk_tps_in_range(self, fake_esmfold):
        template = _make_ca_structure(len(TEST_SEQ))
        walk = ToxinProximityWalk(
            oracle=fake_esmfold,
            template_atoms=template,
            n_steps=10,
        )
        result = walk.run(TEST_SEQ)
        # Both scores should be in a sensible numerical range
        assert isinstance(result['tps'], float)
        assert isinstance(result['tps_auc'], float)
        assert not np.isnan(result['tps'])
        assert not np.isnan(result['tps_auc'])

    def test_toxin_proximity_walk_final_energy_matches_trajectory(self, fake_esmfold):
        template = _make_ca_structure(len(TEST_SEQ))
        walk = ToxinProximityWalk(
            oracle=fake_esmfold,
            template_atoms=template,
            n_steps=5,
        )
        result = walk.run(TEST_SEQ)
        assert result['final_energy'] == result['trajectory'][-1]
