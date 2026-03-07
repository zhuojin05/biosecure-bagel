"""
Modal smoke test — validates all bug-fixed components with minimal GPU calls.

Usage
-----
    cd /Users/zhuojin/Documents/GitHub/biosecure-bagel-fork
    uv run python scripts/smoke_test_modal.py

Expected wall-clock time: 30–90 s (GPU cold-start included).
Expected GPU time: ~5 ESMFold calls + ~2 ESM2 calls ≈ 5–10 GPU-seconds.
"""

from __future__ import annotations

import math
import sys
import pathlib as pl

# Ensure project root is importable when invoked from any working directory.
_ROOT = pl.Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / 'src'))
sys.path.insert(0, str(_ROOT))

# ---------------------------------------------------------------------------
# Test sequences
# ---------------------------------------------------------------------------

# 14-aa PrP fragment — shorter than abeta42, exercises the tps.py atom-count fix.
QUERY = 'KTNMKHMAGAAAAG'

# abeta42 — used as the toxin template throughout.
ABETA42 = 'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA'

_PASS = 'PASS'


def _fail(reason: str) -> None:
    print(f'FAIL: {reason}')
    sys.exit(1)


def _check_finite(value: float, name: str) -> None:
    if not math.isfinite(value):
        _fail(f'{name} is not finite: {value}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import bagel.oracles as bg_oracles
    from bagel.screening.signals import _sequence_to_chain
    from bagel.screening.pathogenic_potential import BAGELPPScreener
    from bagel.screening.tps import ToxinProximityWalk
    from bagel.screening.triage import MultiSignalTriageEngine

    # ------------------------------------------------------------------
    # Initialise oracles (single Modal app context shared across all steps)
    # ------------------------------------------------------------------
    print('Initialising oracles (Modal) ...')
    esmfold = bg_oracles.ESMFold(use_modal=True)
    # ESM2 must share ESMFold's app context — Modal allows only one running
    # context per process (fixes the Modal context-conflict bug).
    esm2 = bg_oracles.ESM2(use_modal=True, modal_app_context=esmfold.modal_app_context)
    print()

    # ------------------------------------------------------------------
    # [1/5] ESMFold fold
    # ------------------------------------------------------------------
    print('[1/5] ESMFold fold ...', end=' ', flush=True)
    try:
        chain = _sequence_to_chain(QUERY)
        fold_result = esmfold.fold([chain])
        atoms = fold_result.structure
        if atoms is None or len(atoms) == 0:
            _fail('ESMFold returned empty AtomArray')
    except Exception as exc:
        _fail(f'ESMFold.fold raised {type(exc).__name__}: {exc}')
    print(_PASS)

    # ------------------------------------------------------------------
    # [2/5] ESM2 embed (shared context)
    # ------------------------------------------------------------------
    print('[2/5] ESM2 embed (shared context) ...', end=' ', flush=True)
    try:
        embed_result = esm2.embed([chain])
        emb = embed_result.embeddings
        if emb is None or emb.size == 0:
            _fail('ESM2 returned empty embeddings')
    except Exception as exc:
        _fail(f'ESM2.embed raised {type(exc).__name__}: {exc}')
    print(_PASS)

    # ------------------------------------------------------------------
    # [3/5] BAGELPPScreener.calculate_pathogenic_potential
    # ------------------------------------------------------------------
    print('[3/5] BAGELPPScreener.calculate_pathogenic_potential ...', end=' ', flush=True)
    try:
        screener = BAGELPPScreener(
            toxin_template=ABETA42,
            active_site_positions=[17, 18, 19, 20],  # KLVFF core of abeta42
            oracle=esmfold,
        )
        pp_result = screener.calculate_pathogenic_potential(QUERY)
        _check_finite(pp_result['pp_score'], 'pp_score')
    except Exception as exc:
        _fail(f'BAGELPPScreener.calculate_pathogenic_potential raised {type(exc).__name__}: {exc}')
    print(_PASS)

    # ------------------------------------------------------------------
    # [4/5] ToxinProximityWalk (n_steps=5, query shorter than template)
    # ------------------------------------------------------------------
    print('[4/5] ToxinProximityWalk (n_steps=5, short query) ...', end=' ', flush=True)
    try:
        # Reuse the already-folded template atoms from the screener cache.
        template_atoms = screener._get_template_atoms()
        tps_walk = ToxinProximityWalk(
            oracle=esmfold,
            template_atoms=template_atoms,
            n_steps=5,
        )
        tps_result = tps_walk.run(QUERY)
        _check_finite(tps_result['tps_auc'], 'tps_auc')
    except Exception as exc:
        _fail(f'ToxinProximityWalk.run raised {type(exc).__name__}: {exc}')
    print(_PASS)

    # ------------------------------------------------------------------
    # [5/5] MultiSignalTriageEngine 4-signal
    # ------------------------------------------------------------------
    print('[5/5] MultiSignalTriageEngine 4-signal ...', end=' ', flush=True)
    try:
        engine = MultiSignalTriageEngine(
            esmfold=esmfold,
            esm2=esm2,
            n_perturbations=3,  # minimal perturbations to keep smoke test fast
        )
        triage = engine.assess_risk(QUERY)
        _check_finite(triage['combined_score'], 'combined_score')
    except Exception as exc:
        _fail(f'MultiSignalTriageEngine.assess_risk raised {type(exc).__name__}: {exc}')
    print(_PASS)

    # ------------------------------------------------------------------
    print()
    print('All smoke tests passed. Safe to run_overnight.sh.')


if __name__ == '__main__':
    main()
