"""
PP-score batch job: run BAGELPPScreener over the full validation dataset.

Scores every sequence against 3 canonical amyloid templates (Abeta42,
alpha-synuclein NAC, PrP 106-126), saves a score matrix and per-template
AUROC.  This is the first time BAGELPPScreener has been run on labelled data.

Usage
-----
    uv run python scripts/overnight/pp_screening_batch.py

Prerequisites
-------------
    modal token set --token-id <ID> --token-secret <SECRET>

Estimated run time: ~2-5 hours on Modal (8,325 ESMFold calls).
"""

from __future__ import annotations

import pathlib as pl
import sys

import numpy as np

# Ensure project root is importable when running from scripts/overnight/
sys.path.insert(0, str(pl.Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(pl.Path(__file__).parent.parent.parent))

try:
    from sklearn.metrics import roc_auc_score
except ImportError as exc:
    raise ImportError(
        'scikit-learn is required. Install with: uv sync --extra dev'
    ) from exc

import bagel.oracles as bg_oracles
from bagel.screening.pathogenic_potential import BAGELPPScreener
from validation.aggregation_dataset import AggregationTestDataset


_OUTPUT_DIR = pl.Path(__file__).parent.parent.parent / 'validation' / 'results'

# ---------------------------------------------------------------------------
# Template definitions
# Template sequences match entries in AggregationTestDataset (positives) so
# self-comparisons are skipped automatically.
# ---------------------------------------------------------------------------
TEMPLATES: list[tuple[str, str, list[int]]] = [
    (
        'abeta42',
        'DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA',
        [17, 18, 19, 20, 21, 22],   # KLVFFA amyloid core
    ),
    (
        'asyn_NAC',
        'EQVTNVGGAVVTGVTAVAQKTVEGAGSIAAATGFVKK',
        [8, 9, 10, 11, 12, 13],     # VGGAVVT hydrophobic core
    ),
    (
        'PrP_106-126',
        'KTNMKHMAGAAAAGAVVGGLGGYMLGSAMSRP',
        [14, 15, 16, 17, 18, 19],   # AGAVVGG amyloid core
    ),
]


def _safe_auroc(y_true: np.ndarray, scores: list[float], label: str = '') -> float:
    finite_mask = np.isfinite(scores)
    n_dropped = int(np.sum(~finite_mask))
    if n_dropped:
        print(f'  WARNING [{label}]: dropping {n_dropped}/{len(scores)} non-finite scores before AUROC')
    y_f = y_true[finite_mask]
    s_f = np.array(scores)[finite_mask]
    if len(y_f) < 2 or len(np.unique(y_f)) < 2:
        print(f'  WARNING [{label}]: not enough classes for AUROC after filtering (n={len(y_f)})')
        return float('nan')
    try:
        return float(roc_auc_score(y_f, s_f))
    except Exception as exc:
        print(f'  WARNING [{label}]: AUROC computation failed: {exc}')
        return float('nan')


def run_pp_screening(
    output_dir: pl.Path = _OUTPUT_DIR,
    n_steps: int = 100,
    n_variants: int = 10,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AggregationTestDataset()
    print(
        f'Dataset: {len(dataset.positives)} positives, {len(dataset.negatives)} negatives'
        f' ({len(dataset)} total)'
    )

    print('Initialising ESMFold oracle (Modal) ...')
    esmfold = bg_oracles.ESMFold(use_modal=True)

    all_rows: list[dict] = []

    for template_name, template_seq, active_sites in TEMPLATES:
        print(f'\n=== Template: {template_name} ===')
        screener = BAGELPPScreener(
            toxin_template=template_seq,
            active_site_positions=active_sites,
            oracle=esmfold,
            n_steps=n_steps,
        )

        # Pre-fold the template once so the TPS walk reuses the cached atoms
        print(f'  Pre-folding template ...')
        _ = screener._get_template_atoms()
        _ = screener._get_tps_walk()

        for i, entry in enumerate(dataset):
            # Skip self-comparison
            if entry.sequence == template_seq:
                print(f'  [{i+1}/{len(dataset)}] {entry.name} — SKIP (self)')
                continue

            print(f'  [{i+1}/{len(dataset)}] {entry.name} ...')
            try:
                result = screener.risk_assessment(entry.sequence, n_variants=n_variants)
                row = {
                    'template': template_name,
                    'name': entry.name,
                    'label': entry.label,
                    'group': entry.group,
                    'pp_score': result['query_pp']['pp_score'],
                    'max_pp': result['max_pp'],
                    'tps_auc': result['tps_auc'],
                    'evasion_efficiency': result['evasion_efficiency'],
                    'efsa_trigger_count': result['efsa_trigger_count'],
                    'risk_category': result['risk_category'],
                    'tm_score_proxy': result['query_pp']['tm_score_proxy'],
                    'function_retained': result['query_pp']['function_retained'],
                    'blast_identity': result['query_pp']['blast_identity'],
                    'evasion_factor': result['query_pp']['evasion_factor'],
                    'evades_blast': int(result['query_pp']['evades_blast']),
                    'above_efsa': int(result['query_pp']['above_efsa']),
                }
                all_rows.append(row)
                print(
                    f'    pp={row["pp_score"]:.3f}  max_pp={row["max_pp"]:.3f}'
                    f'  tps_auc={row["tps_auc"]:.3f}'
                    f'  risk={row["risk_category"]}'
                )
            except Exception as exc:
                print(
                    f'    ERROR scoring {entry.name!r} against template {template_name!r}: '
                    f'{type(exc).__name__}: {exc}'
                )

        # Reset screener TPS cache between templates (new template, new walk)
        screener._tps_walk = None
        screener._template_atoms = None

    # ------------------------------------------------------------------
    # Save full score matrix
    # ------------------------------------------------------------------
    if not all_rows:
        print('No results to save.')
        return

    cols = list(all_rows[0].keys())
    numeric_cols = [c for c in cols if c not in ('template', 'name', 'group', 'risk_category')]
    for row in all_rows:
        bad = [c for c in numeric_cols if not np.isfinite(float(row[c])) if isinstance(row[c], (int, float))]
        if bad:
            raise ValueError(
                f'Non-finite value(s) in row for {row["name"]!r} / {row["template"]!r}: {bad}'
            )
    csv_lines = [','.join(cols)]
    for row in all_rows:
        csv_lines.append(','.join(str(row[c]) for c in cols))
    (output_dir / 'pp_scores.csv').write_text('\n'.join(csv_lines))
    print(f'\nPP scores saved to {output_dir / "pp_scores.csv"}')

    # ------------------------------------------------------------------
    # Per-template AUROC
    # ------------------------------------------------------------------
    auroc_rows = ['template,signal,auroc']
    for template_name, _, _ in TEMPLATES:
        t_rows = [r for r in all_rows if r['template'] == template_name]
        if not t_rows:
            continue
        y_true = np.array([r['label'] for r in t_rows])
        for sig in ('pp_score', 'max_pp', 'tps_auc'):
            scores = [r[sig] for r in t_rows]
            auroc = _safe_auroc(y_true, np.array(scores), label=f'{template_name}/{sig}')
            print(f'  {template_name:<14s}  {sig:<12s}  AUROC={auroc:.3f}')
            auroc_rows.append(f'{template_name},{sig},{auroc:.4f}')

    (output_dir / 'pp_auroc.csv').write_text('\n'.join(auroc_rows))
    print(f'PP AUROC saved to {output_dir / "pp_auroc.csv"}')
    print(f'\nAll PP results saved to {output_dir}')


if __name__ == '__main__':
    run_pp_screening()
