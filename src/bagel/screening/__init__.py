"""
BAGEL Screening Module
======================

Defensive aggregation / prion risk assessment via multi-signal triage.

Public API
----------
MultiSignalTriageEngine : main entry point for risk assessment
SequenceHomologySignal  : BLOSUM62 alignment against reference corpus
MotifDetectionSignal    : hydrophobicity, hexapeptide scan, low-complexity
StructuralPropensitySignal : ESMFold beta-sheet / pLDDT signal (oracle-dep.)
EmbeddingSignal         : ESM2 mean-embedding cosine similarity (oracle-dep.)
LocalRobustnessProbe    : perturbation-based signal stability assessment
"""

from .signals import (
    AggregationSignal,
    EmbeddingSignal,
    MotifDetectionSignal,
    SequenceHomologySignal,
    StructuralPropensitySignal,
)
from .robustness import LocalRobustnessProbe
from .triage import MultiSignalTriageEngine
from .tps import TPSCallback, ToxinProximityWalk
from .pathogenic_potential import BAGELPPScreener

__all__ = [
    'AggregationSignal',
    'BAGELPPScreener',
    'EmbeddingSignal',
    'LocalRobustnessProbe',
    'MotifDetectionSignal',
    'MultiSignalTriageEngine',
    'SequenceHomologySignal',
    'StructuralPropensitySignal',
    'TPSCallback',
    'ToxinProximityWalk',
]
