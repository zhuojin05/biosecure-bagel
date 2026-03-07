"""
PP-score extended template panel: all 10 known-positive sequences as templates.

Runs BAGELPPScreener with every positive from AggregationTestDataset as a
template (reduced depth: n_steps=50, n_variants=5) to produce a comprehensive
amyloid cross-screening matrix.

Usage
-----
    uv run python scripts/overnight/pp_extended_templates.py

Prerequisites
-------------
    modal token set --token-id <ID> --token-secret <SECRET>

Estimated run time: ~3-7 hours on Modal (~13,440 ESMFold calls).
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
# Extended template definitions — all 10 positives from AggregationTestDataset.
# Sequences must match exactly so self-comparisons are auto-skipped.
# Active-site positions mark the dominant aggregation core (0-based).
# ---------------------------------------------------------------------------
TEMPLATES: list[tuple[str, str, list[int]]] = [
    # --- original 3 from pp_screening_batch.py ---
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
    # --- 7 additional positives ---
    (
        'tau_PHF6',
        'VQIVYKPVDLSKVTSKCGSLGNIHHKPGGG',
        [0, 1, 2, 3, 4, 5],         # VQIVYK PHF6 motif
    ),
    (
        'huntingtin_polyQ',
        'MATLEKLMKAFESLKSFQQQQQQQQQQQQQQQQQQQQQQQ',
        [21, 22, 23, 24, 25, 26],   # polyQ run
    ),
    (
        'IAPP_amylin',
        'KCNTATCATQRLANFLVHSSNNFGAILSSTNVGSNTY',
        [23, 24, 25, 26, 27],       # FGAIL core
    ),
    (
        'beta2_microglobulin_K3',
        'IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKN',
        [0, 1, 2, 3, 4, 5],         # IQRTPK N-term
    ),
    (
        'TTR_L55P_fragment',
        'GPTGTGESKCPLMVKVLDAVRGSPAINVAVHVFRKAADDTWEPFASGK',
        [14, 15, 16, 17, 18, 19],   # beta-strand
    ),
    (
        'FUS_QGSY_region',
        'MASNDYTQQATQSYGAYPTQPGQGYSQQSSQPYGQQSYSGYSQSTDTSGYGQ',
        [24, 25, 26, 27, 28, 29],   # SYSGYS repeat
    ),
    (
        'TDP43_GRR',
        'GRFGGGNSSSGPSGNNQNQGNMQREPNQAFGSGNNSYSGSNSGAAIGWGSASNAGSGSGFNGGFGSSMDS',
        [0, 1, 2, 3, 4, 5, 6, 7],   # GFGGGNSS
    ),
]


def _safe_auroc(y_true: np.ndarray, scores: list[float]) -> float:
    try:
        return float(roc_auc_score(y_true, scores))
    except Exception:
        return float('nan')


def run_pp_extended(
    output_dir: pl.Path = _OUTPUT_DIR,
    n_steps: int = 50,
    n_variants: int = 5,
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

        print('  Pre-folding template ...')
        _ = screener._get_template_atoms()
        _ = screener._get_tps_walk()

        for i, entry in enumerate(dataset):
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
                print(f'    ERROR: {exc}')
                all_rows.append({
                    'template': template_name,
                    'name': entry.name,
                    'label': entry.label,
                    'group': entry.group,
                    'pp_score': float('nan'),
                    'max_pp': float('nan'),
                    'tps_auc': float('nan'),
                    'evasion_efficiency': float('nan'),
                    'efsa_trigger_count': float('nan'),
                    'risk_category': 'ERROR',
                    'tm_score_proxy': float('nan'),
                    'function_retained': float('nan'),
                    'blast_identity': float('nan'),
                    'evasion_factor': float('nan'),
                    'evades_blast': -1,
                    'above_efsa': -1,
                })

        screener._tps_walk = None
        screener._template_atoms = None

    # ------------------------------------------------------------------
    # Save full score matrix
    # ------------------------------------------------------------------
    if not all_rows:
        print('No results to save.')
        return

    cols = list(all_rows[0].keys())
    csv_lines = [','.join(cols)]
    for row in all_rows:
        csv_lines.append(','.join(str(row[c]) for c in cols))
    (output_dir / 'pp_extended_scores.csv').write_text('\n'.join(csv_lines))
    print(f'\nExtended PP scores saved to {output_dir / "pp_extended_scores.csv"}')

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
            auroc = _safe_auroc(y_true, scores)
            print(f'  {template_name:<25s}  {sig:<12s}  AUROC={auroc:.3f}')
            auroc_rows.append(f'{template_name},{sig},{auroc:.4f}')

    (output_dir / 'pp_extended_auroc.csv').write_text('\n'.join(auroc_rows))
    print(f'Extended PP AUROC saved to {output_dir / "pp_extended_auroc.csv"}')
    print(f'\nAll extended PP results saved to {output_dir}')


if __name__ == '__main__':
    run_pp_extended()
