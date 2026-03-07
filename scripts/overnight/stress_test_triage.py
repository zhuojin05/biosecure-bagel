"""
False-positive stress test: 50 random sequences through MultiSignalTriageEngine.

Generates 50 random amino-acid sequences (lengths 30-150 aa) and runs each
through the full 4-signal triage engine (ESMFold + ESM2).  Logs the complete
score distribution and computes the false-positive rate at each risk threshold.

Usage
-----
    uv run python scripts/overnight/stress_test_triage.py

Prerequisites
-------------
    modal token set --token-id <ID> --token-secret <SECRET>

Estimated run time: ~5-10 minutes on Modal (100 oracle calls).
"""

from __future__ import annotations

import pathlib as pl
import sys

import numpy as np

# Ensure project root is importable when running from scripts/overnight/
sys.path.insert(0, str(pl.Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(pl.Path(__file__).parent.parent.parent))

import bagel as bg
import bagel.oracles as bg_oracles
from bagel.screening import MultiSignalTriageEngine


_OUTPUT_DIR = pl.Path(__file__).parent.parent.parent / 'validation' / 'results'

N_SEQUENCES = 50
MIN_LENGTH = 30
MAX_LENGTH = 150
SEED = 42


def run_stress_test(output_dir: pl.Path = _OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    print('Initialising oracles (Modal) ...')
    esmfold = bg_oracles.ESMFold(use_modal=True)
    # Reuse the app context already started by ESMFold; Modal allows only one
    # running app context per process, so ESM2 must not start its own.
    esm2 = bg_oracles.ESM2(use_modal=True, modal_app_context=esmfold.modal_app_context)

    engine = MultiSignalTriageEngine(esmfold=esmfold, esm2=esm2, n_perturbations=30)

    aas = list(bg.constants.aa_dict.keys())
    rng = np.random.default_rng(SEED)

    rows: list[dict] = []
    print(f'\nGenerating and scoring {N_SEQUENCES} random sequences ...')
    for i in range(N_SEQUENCES):
        length = int(rng.integers(MIN_LENGTH, MAX_LENGTH + 1))
        seq = ''.join(rng.choice(aas, size=length))
        print(f'  [{i+1}/{N_SEQUENCES}] length={length} ...')
        try:
            result = engine.assess_risk(seq)
            row = {
                'seq_id': i,
                'length': length,
                'seq': seq,
                **result['signal_scores'],
                'combined': result['combined_score'],
                'risk': result['risk_level'],
            }
            rows.append(row)
            print(
                f'    combined={result["combined_score"]:.3f}'
                f'  risk={result["risk_level"]}'
            )
        except Exception as exc:
            print(f'    ERROR: {exc}')
            rows.append({
                'seq_id': i,
                'length': length,
                'seq': seq,
                'combined': float('nan'),
                'risk': 'ERROR',
            })

    # ------------------------------------------------------------------
    # Save score CSV
    # ------------------------------------------------------------------
    if not rows:
        print('No results to save.')
        return

    # Collect all field names (signal_scores keys vary if errors occurred)
    all_keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in all_keys:
                all_keys.append(k)

    csv_lines = [','.join(all_keys)]
    for row in rows:
        csv_lines.append(','.join(str(row.get(k, 'nan')) for k in all_keys))

    out_csv = output_dir / 'stress_test_scores.csv'
    out_csv.write_text('\n'.join(csv_lines))
    print(f'\nStress-test scores saved to {out_csv}')

    # ------------------------------------------------------------------
    # False-positive rate summary
    # ------------------------------------------------------------------
    valid_rows = [r for r in rows if r.get('risk') not in ('ERROR', None)]
    total = len(valid_rows)
    if total == 0:
        print('No valid rows for FPR summary.')
        return

    risk_levels = ['HIGH', 'MEDIUM', 'LOW']
    print('\n--- False-positive rate summary (random sequences, expected label=0) ---')
    print(f'  Total evaluated : {total}')
    for level in risk_levels:
        count = sum(1 for r in valid_rows if r['risk'] == level)
        fpr = count / total
        print(f'  {level:<8s} : {count:>3d} / {total}  ({fpr:.1%})')

    combined_scores = [r['combined'] for r in valid_rows if not np.isnan(r.get('combined', float('nan')))]
    if combined_scores:
        print(f'\n  combined score — mean={np.mean(combined_scores):.3f}'
              f'  std={np.std(combined_scores):.3f}'
              f'  min={np.min(combined_scores):.3f}'
              f'  max={np.max(combined_scores):.3f}')

    print(f'\nAll stress-test results saved to {output_dir}')


if __name__ == '__main__':
    run_stress_test()
