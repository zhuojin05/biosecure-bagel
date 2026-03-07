"""
4-signal validation pipeline for the aggregation screening module.

Extends scripts/validate_aggregation_screen.py to include GPU-backed oracle
signals (ESMFold structural propensity + ESM2 embedding) in addition to the
two fast sequence-only signals.  Runs the full labelled dataset through
MultiSignalTriageEngine with all 4 signals and saves updated metrics.

Usage
-----
    uv run python scripts/overnight/validate_full_signals.py

Prerequisites
-------------
    modal token set --token-id <ID> --token-secret <SECRET>
"""

from __future__ import annotations

import pathlib as pl
import sys

import numpy as np

# Ensure project root is importable when running from scripts/overnight/
sys.path.insert(0, str(pl.Path(__file__).parent.parent.parent / 'src'))
sys.path.insert(0, str(pl.Path(__file__).parent.parent.parent))

try:
    from sklearn.metrics import (
        average_precision_score,
        precision_recall_curve,
        roc_auc_score,
        roc_curve,
    )
except ImportError as exc:
    raise ImportError(
        'scikit-learn is required. Install with: uv sync --extra dev'
    ) from exc

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import bagel.oracles as bg_oracles
from bagel.screening import MultiSignalTriageEngine
from validation.aggregation_dataset import AggregationTestDataset


_OUTPUT_DIR = pl.Path(__file__).parent.parent.parent / 'validation' / 'results'


def _safe_auroc(y_true: np.ndarray, scores: list[float], label: str = '') -> float:
    arr = np.asarray(scores, dtype=float)
    finite_mask = np.isfinite(arr)
    n_dropped = int(np.sum(~finite_mask))
    if n_dropped:
        print(f'  WARNING [{label}]: dropping {n_dropped}/{len(arr)} non-finite scores before AUROC')
    y_f, s_f = y_true[finite_mask], arr[finite_mask]
    if len(y_f) < 2 or len(np.unique(y_f)) < 2:
        print(f'  WARNING [{label}]: not enough classes after filtering (n={len(y_f)})')
        return float('nan')
    try:
        return float(roc_auc_score(y_f, s_f))
    except Exception as exc:
        print(f'  WARNING [{label}]: AUROC failed: {exc}')
        return float('nan')


def _safe_auprc(y_true: np.ndarray, scores: list[float], label: str = '') -> float:
    arr = np.asarray(scores, dtype=float)
    finite_mask = np.isfinite(arr)
    n_dropped = int(np.sum(~finite_mask))
    if n_dropped:
        print(f'  WARNING [{label}]: dropping {n_dropped}/{len(arr)} non-finite scores before AUPRC')
    y_f, s_f = y_true[finite_mask], arr[finite_mask]
    if len(y_f) < 2 or len(np.unique(y_f)) < 2:
        print(f'  WARNING [{label}]: not enough classes after filtering (n={len(y_f)})')
        return float('nan')
    try:
        return float(average_precision_score(y_f, s_f))
    except Exception as exc:
        print(f'  WARNING [{label}]: AUPRC failed: {exc}')
        return float('nan')


def run_validation(output_dir: pl.Path = _OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AggregationTestDataset()
    print(f'Dataset: {len(dataset.positives)} positives, {len(dataset.negatives)} negatives')

    # ------------------------------------------------------------------
    # Instantiate oracles (Modal GPU)
    # ------------------------------------------------------------------
    print('Initialising oracles (Modal) ...')
    esmfold = bg_oracles.ESMFold(use_modal=True)
    # Reuse the app context already started by ESMFold; Modal allows only one
    # running app context per process, so ESM2 must not start its own.
    esm2 = bg_oracles.ESM2(use_modal=True, modal_app_context=esmfold.modal_app_context)

    # ------------------------------------------------------------------
    # Instantiate engine with all 4 signals
    # ------------------------------------------------------------------
    engine = MultiSignalTriageEngine(esmfold=esmfold, esm2=esm2, n_perturbations=30)

    labels: list[int] = []
    combined_scores: list[float] = []
    homology_scores: list[float] = []
    motif_scores: list[float] = []
    structure_scores: list[float] = []
    embedding_scores: list[float] = []
    names: list[str] = []

    print('Scoring sequences ...')
    for i, entry in enumerate(dataset):
        print(f'  [{i+1}/{len(dataset)}] {entry.name} ...')
        try:
            result = engine.assess_risk(entry.sequence)
        except Exception as exc:
            print(
                f'    ERROR scoring {entry.name!r}: {type(exc).__name__}: {exc}'
            )
            continue
        labels.append(entry.label)
        combined_scores.append(result['combined_score'])
        homology_scores.append(result['signal_scores']['homology'])
        motif_scores.append(result['signal_scores']['motif'])
        structure_scores.append(result['signal_scores'].get('structure', float('nan')))
        embedding_scores.append(result['signal_scores'].get('embedding', float('nan')))
        names.append(entry.name)
        print(
            f'    label={entry.label}  combined={result["combined_score"]:.3f}'
            f'  homology={result["signal_scores"]["homology"]:.3f}'
            f'  motif={result["signal_scores"]["motif"]:.3f}'
            f'  structure={result["signal_scores"].get("structure", float("nan")):.3f}'
            f'  embedding={result["signal_scores"].get("embedding", float("nan")):.3f}'
            f'  risk={result["risk_level"]}'
        )

    y_true = np.array(labels)

    # ------------------------------------------------------------------
    # Save per-sequence scores
    # ------------------------------------------------------------------
    header = 'name,label,combined,homology,motif,structure,embedding'
    rows = [header]
    for n, l, c, h, m, s, e in zip(
        names, labels, combined_scores, homology_scores,
        motif_scores, structure_scores, embedding_scores
    ):
        rows.append(f'{n},{l},{c:.4f},{h:.4f},{m:.4f},{s:.4f},{e:.4f}')
    (output_dir / 'scores_full.csv').write_text('\n'.join(rows))
    print(f'\nScores saved to {output_dir / "scores_full.csv"}')

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    signal_data = {
        'combined': combined_scores,
        'homology': homology_scores,
        'motif': motif_scores,
        'structure': structure_scores,
        'embedding': embedding_scores,
    }

    print('\n--- Results ---')
    metrics_rows = ['signal,auroc,auprc']
    for sig_name, scores in signal_data.items():
        auroc = _safe_auroc(y_true, scores, label=sig_name)
        auprc = _safe_auprc(y_true, scores, label=sig_name)
        print(f'  {sig_name:<12s}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}')
        metrics_rows.append(f'{sig_name},{auroc:.4f},{auprc:.4f}')

    (output_dir / 'metrics_full.csv').write_text('\n'.join(metrics_rows))
    print(f'Metrics saved to {output_dir / "metrics_full.csv"}')

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # ROC curves
        ax = axes[0]
        for sig_name, scores in signal_data.items():
            auroc = _safe_auroc(y_true, scores, label=sig_name)
            try:
                arr = np.asarray(scores, dtype=float)
                mask = np.isfinite(arr)
                fpr, tpr, _ = roc_curve(y_true[mask], arr[mask])
                ax.plot(fpr, tpr, label=f'{sig_name} (AUROC={auroc:.2f})')
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curves (4-signal)')
        ax.legend(fontsize=7)

        # PR curves
        ax = axes[1]
        for sig_name, scores in signal_data.items():
            auprc = _safe_auprc(y_true, scores, label=sig_name)
            try:
                arr = np.asarray(scores, dtype=float)
                mask = np.isfinite(arr)
                prec, rec, _ = precision_recall_curve(y_true[mask], arr[mask])
                ax.plot(rec, prec, label=f'{sig_name} (AUPRC={auprc:.2f})')
            except Exception:
                pass
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR Curves (4-signal)')
        ax.legend(fontsize=7)

        fig.tight_layout()
        plot_path = output_dir / 'roc_pr_curves_full.png'
        fig.savefig(plot_path, dpi=150)
        print(f'Plot saved to {plot_path}')
        plt.close(fig)

    print(f'\nAll results saved to {output_dir}')


if __name__ == '__main__':
    run_validation()
