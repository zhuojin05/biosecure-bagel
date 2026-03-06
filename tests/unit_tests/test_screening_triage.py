"""Unit tests for MultiSignalTriageEngine."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from bagel.screening.signals import AggregationSignal
from bagel.screening.triage import MultiSignalTriageEngine

_SEQ = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'  # Aβ42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mock_signal(value: float) -> AggregationSignal:
    sig = MagicMock(spec=AggregationSignal)
    sig.score.return_value = value
    return sig


def _mock_probe_robustness(robustness_map: dict) -> MagicMock:
    probe = MagicMock()
    probe.probe_robustness.return_value = robustness_map
    return probe


# ---------------------------------------------------------------------------
# Instantiation
# ---------------------------------------------------------------------------


def test_engine_instantiates_with_no_oracles():
    engine = MultiSignalTriageEngine()
    assert 'homology' in engine.fast_signals
    assert 'motif' in engine.fast_signals
    assert len(engine.slow_signals) == 0


def test_engine_with_esmfold_adds_structure_signal(fake_esmfold):
    engine = MultiSignalTriageEngine(esmfold=fake_esmfold)
    assert 'structure' in engine.slow_signals


def test_engine_with_esm2_adds_embedding_signal(fake_esm2, tmp_path):
    # Create a dummy reference embeddings file so EmbeddingSignal doesn't raise
    import numpy as np

    ref = np.random.randn(5, 8).astype(np.float64)
    ref_path = tmp_path / 'reference_embeddings.npy'
    np.save(ref_path, ref)

    with patch('bagel.screening.signals._REFERENCE_EMBEDDINGS_PATH', ref_path):
        engine = MultiSignalTriageEngine(esm2=fake_esm2)
    assert 'embedding' in engine.slow_signals


# ---------------------------------------------------------------------------
# assess_risk — combined score formula
# ---------------------------------------------------------------------------


def test_assess_risk_combined_score_formula():
    """
    Test the combined score formula:
        combined = mean(signal_scores[n] * (1 + robustness_scores[n]))
    using known mock values.
    """
    engine = MultiSignalTriageEngine(n_perturbations=5)

    # Override fast signals with known-value mocks
    engine.fast_signals = {
        'homology': _mock_signal(0.6),
        'motif': _mock_signal(0.4),
    }

    # Override the probe to return known robustness values
    engine.probe = _mock_probe_robustness({'homology': 0.8, 'motif': 0.5})

    result = engine.assess_risk(_SEQ)

    expected_combined = np.mean(
        [
            0.6 * (1.0 + 0.8),  # homology: 0.6 * 1.8 = 1.08
            0.4 * (1.0 + 0.5),  # motif: 0.4 * 1.5 = 0.6
        ]
    )
    assert np.isclose(result['combined_score'], expected_combined), (
        f'Expected {expected_combined}, got {result["combined_score"]}'
    )


def test_assess_risk_slow_signal_robustness_fixed_at_one(fake_esmfold):
    """Slow signals always get robustness=1.0."""
    engine = MultiSignalTriageEngine(n_perturbations=3)

    # Replace fast signals with constants
    engine.fast_signals = {'homology': _mock_signal(0.5), 'motif': _mock_signal(0.5)}
    # Add a mock slow signal
    engine.slow_signals = {'structure': _mock_signal(0.9)}

    engine.probe = _mock_probe_robustness({'homology': 0.5, 'motif': 0.5})

    result = engine.assess_risk(_SEQ)
    assert result['robustness_scores']['structure'] == 1.0


# ---------------------------------------------------------------------------
# assess_risk — risk level thresholds
# ---------------------------------------------------------------------------


def test_risk_level_high():
    engine = MultiSignalTriageEngine(n_perturbations=3)
    engine.fast_signals = {'homology': _mock_signal(0.9), 'motif': _mock_signal(0.9)}
    engine.probe = _mock_probe_robustness({'homology': 1.0, 'motif': 1.0})
    result = engine.assess_risk(_SEQ)
    # combined = mean([0.9*2, 0.9*2]) = 1.8 > 1.2 → HIGH
    assert result['risk_level'] == 'HIGH'


def test_risk_level_medium():
    engine = MultiSignalTriageEngine(n_perturbations=3)
    # combined ≈ mean([0.4*2, 0.4*2]) = 0.8 → MEDIUM
    engine.fast_signals = {'homology': _mock_signal(0.4), 'motif': _mock_signal(0.4)}
    engine.probe = _mock_probe_robustness({'homology': 1.0, 'motif': 1.0})
    result = engine.assess_risk(_SEQ)
    assert result['risk_level'] == 'MEDIUM'


def test_risk_level_low():
    engine = MultiSignalTriageEngine(n_perturbations=3)
    # combined = mean([0.1*1.0, 0.1*1.0]) = 0.1 → LOW
    engine.fast_signals = {'homology': _mock_signal(0.1), 'motif': _mock_signal(0.1)}
    engine.probe = _mock_probe_robustness({'homology': 0.0, 'motif': 0.0})
    result = engine.assess_risk(_SEQ)
    assert result['risk_level'] == 'LOW'


# ---------------------------------------------------------------------------
# assess_risk — output structure
# ---------------------------------------------------------------------------


def test_assess_risk_returns_required_keys():
    engine = MultiSignalTriageEngine(n_perturbations=3)
    engine.fast_signals = {'homology': _mock_signal(0.5), 'motif': _mock_signal(0.5)}
    engine.probe = _mock_probe_robustness({'homology': 0.8, 'motif': 0.8})
    result = engine.assess_risk(_SEQ)
    assert 'combined_score' in result
    assert 'risk_level' in result
    assert 'signal_scores' in result
    assert 'robustness_scores' in result


def test_assess_risk_signal_scores_match_signal_names():
    engine = MultiSignalTriageEngine(n_perturbations=3)
    engine.fast_signals = {'homology': _mock_signal(0.7), 'motif': _mock_signal(0.3)}
    engine.probe = _mock_probe_robustness({'homology': 1.0, 'motif': 1.0})
    result = engine.assess_risk(_SEQ)
    assert result['signal_scores']['homology'] == pytest.approx(0.7)
    assert result['signal_scores']['motif'] == pytest.approx(0.3)


def test_assess_risk_robustness_probe_called_with_fast_signals_only():
    engine = MultiSignalTriageEngine(n_perturbations=5)
    engine.fast_signals = {'homology': _mock_signal(0.5), 'motif': _mock_signal(0.5)}
    engine.slow_signals = {'structure': _mock_signal(0.8)}
    engine.probe = _mock_probe_robustness({'homology': 0.9, 'motif': 0.7})

    engine.assess_risk(_SEQ)

    # probe.probe_robustness should be called with fast_signals only
    call_args = engine.probe.probe_robustness.call_args
    passed_signals = call_args[0][1] if call_args[0] else call_args[1]['signals']
    assert 'homology' in passed_signals
    assert 'motif' in passed_signals
    assert 'structure' not in passed_signals
