"""
End-to-end validation pipeline for the aggregation screening module.

Computes AUROC and AUPRC for fast signals (no GPU required), with and without
robustness weighting, and saves results + ROC/PR curve plots.

Usage
-----
    uv run python scripts/validate_aggregation_screen.py
"""

from __future__ import annotations

import pathlib as pl
import sys

import numpy as np

# Ensure project root is importable when running from scripts/
sys.path.insert(0, str(pl.Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(pl.Path(__file__).parent.parent))

try:
    from sklearn.metrics import auc, average_precision_score, roc_auc_score, roc_curve, precision_recall_curve
except ImportError as exc:
    raise ImportError(
        'scikit-learn is required for validation. Install with: uv sync --extra dev'
    ) from exc

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

from bagel.screening import MultiSignalTriageEngine
from bagel.screening.signals import SequenceHomologySignal, MotifDetectionSignal
from validation.aggregation_dataset import AggregationTestDataset


_OUTPUT_DIR = pl.Path(__file__).parent.parent / 'validation' / 'results'


def run_validation(output_dir: pl.Path = _OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = AggregationTestDataset()
    print(f'Dataset: {len(dataset.positives)} positives, {len(dataset.negatives)} negatives')

    # ------------------------------------------------------------------
    # Instantiate engine (fast signals only — no GPU required)
    # ------------------------------------------------------------------
    engine = MultiSignalTriageEngine(n_perturbations=30)

    labels: list[int] = []
    combined_scores: list[float] = []
    homology_scores: list[float] = []
    motif_scores: list[float] = []
    names: list[str] = []

    print('Scoring sequences ...')
    for entry in dataset:
        result = engine.assess_risk(entry.sequence)
        labels.append(entry.label)
        combined_scores.append(result['combined_score'])
        homology_scores.append(result['signal_scores']['homology'])
        motif_scores.append(result['signal_scores']['motif'])
        names.append(entry.name)
        print(
            f'  {entry.name:<40s} label={entry.label}  '
            f'combined={result["combined_score"]:.3f}  '
            f'risk={result["risk_level"]}'
        )

    y_true = np.array(labels)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    results_rows: list[str] = ['name,label,combined,homology,motif']
    for n, l, c, h, m in zip(names, labels, combined_scores, homology_scores, motif_scores):
        results_rows.append(f'{n},{l},{c:.4f},{h:.4f},{m:.4f}')
    (output_dir / 'scores.csv').write_text('\n'.join(results_rows))

    def _safe_auroc(scores: list[float]) -> float:
        try:
            return float(roc_auc_score(y_true, scores))
        except Exception:
            return float('nan')

    def _safe_auprc(scores: list[float]) -> float:
        try:
            return float(average_precision_score(y_true, scores))
        except Exception:
            return float('nan')

    auroc_combined = _safe_auroc(combined_scores)
    auroc_homology = _safe_auroc(homology_scores)
    auroc_motif = _safe_auroc(motif_scores)

    auprc_combined = _safe_auprc(combined_scores)
    auprc_homology = _safe_auprc(homology_scores)
    auprc_motif = _safe_auprc(motif_scores)

    print('\n--- Results ---')
    print(f'Combined  AUROC={auroc_combined:.3f}  AUPRC={auprc_combined:.3f}')
    print(f'Homology  AUROC={auroc_homology:.3f}  AUPRC={auprc_homology:.3f}')
    print(f'Motif     AUROC={auroc_motif:.3f}  AUPRC={auprc_motif:.3f}')

    metrics_lines = [
        'signal,auroc,auprc',
        f'combined,{auroc_combined:.4f},{auprc_combined:.4f}',
        f'homology,{auroc_homology:.4f},{auprc_homology:.4f}',
        f'motif,{auroc_motif:.4f},{auprc_motif:.4f}',
    ]
    (output_dir / 'metrics.csv').write_text('\n'.join(metrics_lines))

    # ------------------------------------------------------------------
    # Ablation: fast signals without vs with robustness weighting
    # ------------------------------------------------------------------
    print('\n--- Ablation: homology+motif mean (no robustness) ---')
    raw_scores = [(h + m) / 2 for h, m in zip(homology_scores, motif_scores)]
    auroc_raw = _safe_auroc(raw_scores)
    auprc_raw = _safe_auprc(raw_scores)
    print(f'  AUROC={auroc_raw:.3f}  AUPRC={auprc_raw:.3f}')
    print('--- Combined (with robustness weighting) ---')
    print(f'  AUROC={auroc_combined:.3f}  AUPRC={auprc_combined:.3f}')

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    if HAS_MPL:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # ROC curves
        ax = axes[0]
        for scores, label in [
            (combined_scores, f'Combined (AUROC={auroc_combined:.2f})'),
            (homology_scores, f'Homology (AUROC={auroc_homology:.2f})'),
            (motif_scores, f'Motif (AUROC={auroc_motif:.2f})'),
        ]:
            try:
                fpr, tpr, _ = roc_curve(y_true, scores)
                ax.plot(fpr, tpr, label=label)
            except Exception:
                pass
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3)
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.set_title('ROC Curves')
        ax.legend(fontsize=8)

        # PR curves
        ax = axes[1]
        for scores, label in [
            (combined_scores, f'Combined (AUPRC={auprc_combined:.2f})'),
            (homology_scores, f'Homology (AUPRC={auprc_homology:.2f})'),
            (motif_scores, f'Motif (AUPRC={auprc_motif:.2f})'),
        ]:
            try:
                prec, rec, _ = precision_recall_curve(y_true, scores)
                ax.plot(rec, prec, label=label)
            except Exception:
                pass
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('PR Curves')
        ax.legend(fontsize=8)

        fig.tight_layout()
        plot_path = output_dir / 'roc_pr_curves.png'
        fig.savefig(plot_path, dpi=150)
        print(f'\nPlot saved to {plot_path}')
        plt.close(fig)

    print(f'\nResults saved to {output_dir}')


if __name__ == '__main__':
    run_validation()
