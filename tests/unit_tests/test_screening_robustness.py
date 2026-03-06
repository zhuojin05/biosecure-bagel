"""Unit tests for LocalRobustnessProbe."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from bagel.screening.robustness import LocalRobustnessProbe, _sequence_to_system
from bagel.screening.signals import AggregationSignal

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------

_SEQ = 'MQIFVKTLTGKTITLE'  # short generic sequence


class _ConstantSignal(AggregationSignal):
    """Always returns the same score."""

    def __init__(self, value: float) -> None:
        self._value = value

    def score(self, sequence: str) -> float:
        return self._value


class _NoisySignal(AggregationSignal):
    """Returns a random score drawn from a uniform distribution."""

    def __init__(self, rng: np.random.Generator) -> None:
        self._rng = rng

    def score(self, sequence: str) -> float:
        return float(self._rng.uniform(0.0, 1.0))


# ---------------------------------------------------------------------------
# _sequence_to_system
# ---------------------------------------------------------------------------


def test_sequence_to_system_builds_valid_system():
    system = _sequence_to_system(_SEQ)
    assert len(system.states) == 1
    state = system.states[0]
    assert len(state.energy_terms) == 0
    assert state.chains[0].sequence == _SEQ


def test_sequence_to_system_all_mutable():
    system = _sequence_to_system(_SEQ)
    chain = system.states[0].chains[0]
    assert all(r.mutable for r in chain.residues)


# ---------------------------------------------------------------------------
# generate_local_variants
# ---------------------------------------------------------------------------


def test_generate_local_variants_count():
    probe = LocalRobustnessProbe()
    variants = probe.generate_local_variants(_SEQ, n_variants=10)
    assert len(variants) == 10


def test_generate_local_variants_are_same_length():
    probe = LocalRobustnessProbe()
    variants = probe.generate_local_variants(_SEQ, n_variants=20)
    for v in variants:
        assert len(v) == len(_SEQ)


def test_generate_local_variants_differ_from_original():
    probe = LocalRobustnessProbe(n_mutations=1)
    # With enough variants, at least one should differ from the original
    variants = probe.generate_local_variants(_SEQ, n_variants=30)
    assert any(v != _SEQ for v in variants), 'All variants identical to original (very unlikely)'


def test_generate_local_variants_single_substitution():
    """Each variant should differ from the original by exactly 1 residue."""
    probe = LocalRobustnessProbe(n_mutations=1)
    variants = probe.generate_local_variants(_SEQ, n_variants=20)
    for v in variants:
        diff = sum(a != b for a, b in zip(v, _SEQ))
        assert diff == 1, f'Expected exactly 1 substitution, got {diff}'


# ---------------------------------------------------------------------------
# probe_robustness
# ---------------------------------------------------------------------------


def test_probe_robustness_keys_match_signals():
    probe = LocalRobustnessProbe()
    signals = {'sig_a': _ConstantSignal(0.5), 'sig_b': _ConstantSignal(0.8)}
    result = probe.probe_robustness(_SEQ, signals, n_perturbations=5)
    assert set(result.keys()) == {'sig_a', 'sig_b'}


def test_probe_robustness_constant_signal_returns_one():
    """A signal that always returns the same value has std=0 → robustness=1.0."""
    probe = LocalRobustnessProbe()
    signals = {'constant': _ConstantSignal(0.7)}
    result = probe.probe_robustness(_SEQ, signals, n_perturbations=10)
    assert np.isclose(result['constant'], 1.0), f'Expected 1.0, got {result["constant"]}'


def test_probe_robustness_range():
    """Robustness scores should be in [0, 1]."""
    rng = np.random.default_rng(42)
    probe = LocalRobustnessProbe()
    signals = {
        'noisy': _NoisySignal(rng),
        'constant': _ConstantSignal(0.3),
    }
    result = probe.probe_robustness(_SEQ, signals, n_perturbations=15)
    for name, rob in result.items():
        assert 0.0 <= rob <= 1.0, f'Signal {name!r} robustness {rob} out of [0,1]'


def test_probe_robustness_zero_mean_signal():
    """Signal that always returns 0 → mean=0, robustness formula denominator = 1e-6."""
    probe = LocalRobustnessProbe()
    signals = {'zero': _ConstantSignal(0.0)}
    result = probe.probe_robustness(_SEQ, signals, n_perturbations=5)
    # std=0, mean=0, robustness = max(0, 1 - 0/1e-6) = max(0, 1-0) = 1.0
    assert np.isclose(result['zero'], 1.0)


def test_probe_robustness_uses_correct_n_perturbations():
    """Signal score() should be called exactly n_perturbations times."""
    probe = LocalRobustnessProbe()
    mock_signal = MagicMock(spec=AggregationSignal)
    mock_signal.score.return_value = 0.5
    signals = {'mock': mock_signal}
    n = 12
    probe.probe_robustness(_SEQ, signals, n_perturbations=n)
    assert mock_signal.score.call_count == n
