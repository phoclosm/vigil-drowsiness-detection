"""Reproducible evaluation and head-to-head model comparison for Vigil M2."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vigil.ml.baseline.classifier import RuleBasedBaseline
from vigil.ml.baseline.evaluation import (
    BaselineEvaluationReport,
    compute_metrics,
    evaluate_baseline,
)
from vigil.ml.data.schema import SampleRecord
from vigil.ml.dataset.dataset import FatigueWindowDataset
from vigil.ml.dataset.normalizer import FeatureNormalizer
from vigil.ml.dataset.splitter import DatasetGroupSplitResult, split_records_by_group
from vigil.ml.dataset.windowing import extract_windows
from vigil.ml.models.export import LoadedModelBundle, load_model_bundle
from vigil.ml.models.gru import GRUTemporalClassifier

try:
    import torch
    HAS_TORCH = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    HAS_TORCH = False


@dataclass(frozen=True)
class ModelComparisonReport:
    """Comparative evaluation report contrasting the learned GRU model and D3 rule baseline.

    Attributes:
        gru_report: Evaluation metrics of the learned GRU sequence classifier on held-out windows.
        baseline_matched_report: Evaluation metrics of the D3 rule baseline on the identical window endpoint samples.
        baseline_full_report: Frame-by-frame evaluation metrics of the D3 rule baseline across all non-ambiguous samples.
        split_result: Metadata describing the subject/session grouped split.
        window_count: Number of evaluation windows evaluated.
        is_synthetic: Whether the evaluation was executed on synthetic test fixtures.
    """

    gru_report: BaselineEvaluationReport
    baseline_matched_report: BaselineEvaluationReport
    baseline_full_report: BaselineEvaluationReport
    split_result: DatasetGroupSplitResult
    window_count: int
    is_synthetic: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Convert the comparative report into a nested dictionary."""
        return {
            "is_synthetic": self.is_synthetic,
            "window_count": self.window_count,
            "gru_model": self.gru_report.to_dict(),
            "rule_baseline_matched": self.baseline_matched_report.to_dict(),
            "rule_baseline_full": self.baseline_full_report.to_dict(),
            "split_result": {
                "strategy": self.split_result.strategy,
                "train_subjects": self.split_result.train_subjects,
                "val_subjects": self.split_result.val_subjects,
                "train_sessions": self.split_result.train_sessions,
                "val_sessions": self.split_result.val_sessions,
                "limitation_note": self.split_result.limitation_note,
            },
        }

    def to_markdown(self) -> str:
        """Render a formatted markdown comparison table."""
        lines = [
            "### M2 Model Comparison Report: Learned GRU vs. D3 Rule Baseline",
            "",
        ]

        if self.is_synthetic:
            lines.extend([
                "> [!IMPORTANT]",
                (
                    "> **Disclaimer on Synthetic Fixtures**: This evaluation was executed using synthetic "
                    "unit-test fixtures because no real labeled driver recordings are present in the repository. "
                    "These metrics validate code mechanics, tensor contracts, and metric pipelines; they "
                    "do NOT establish real-world driver drowsiness detection accuracy."
                ),
                "",
            ])

        lines.extend([
            f"- **Evaluation Partition Strategy**: `{self.split_result.strategy}`",
            f"- **Evaluated Windows ($W=60$)**: {self.window_count}",
            f"- **Validation Sessions**: {self.split_result.val_sessions}",
            f"- **Validation Subjects**: {self.split_result.val_subjects}",
            "",
            "#### Head-to-Head Performance Comparison",
            "| Metric | Learned GRU (Windows) | Rule Baseline (Matched Timesteps) | Rule Baseline (All Frames) |",
            "|---|---|---|---|",
            f"| **Overall Accuracy** | {self.gru_report.accuracy:.4f} | {self.baseline_matched_report.accuracy:.4f} | {self.baseline_full_report.accuracy:.4f} |",
            f"| **Macro Precision** | {self.gru_report.macro_precision:.4f} | {self.baseline_matched_report.macro_precision:.4f} | {self.baseline_full_report.macro_precision:.4f} |",
            f"| **Macro Recall** | {self.gru_report.macro_recall:.4f} | {self.baseline_matched_report.macro_recall:.4f} | {self.baseline_full_report.macro_recall:.4f} |",
            f"| **Macro F1 Score** | {self.gru_report.macro_f1:.4f} | {self.baseline_matched_report.macro_f1:.4f} | {self.baseline_full_report.macro_f1:.4f} |",
            f"| **Prediction Coverage** | {self.gru_report.coverage:.4f} ({self.gru_report.evaluated_samples}/{self.gru_report.eligible_samples}) | {self.baseline_matched_report.coverage:.4f} ({self.baseline_matched_report.evaluated_samples}/{self.baseline_matched_report.eligible_samples}) | {self.baseline_full_report.coverage:.4f} ({self.baseline_full_report.evaluated_samples}/{self.baseline_full_report.eligible_samples}) |",
            f"| **Skipped Predictions** | {self.gru_report.skipped_predictions} | {self.baseline_matched_report.skipped_predictions} | {self.baseline_full_report.skipped_predictions} |",
            "",
            "#### GRU Confusion Matrix (Canonical Order: 0: ACTIVE, 1: DROWSY, 2: SLEEPING)",
            "| True \\ Pred | ACTIVE (0) | DROWSY (1) | SLEEPING (2) |",
            "|---|---|---|---|",
            f"| **ACTIVE (0)** | {self.gru_report.confusion_matrix[0][0]} | {self.gru_report.confusion_matrix[0][1]} | {self.gru_report.confusion_matrix[0][2]} |",
            f"| **DROWSY (1)** | {self.gru_report.confusion_matrix[1][0]} | {self.gru_report.confusion_matrix[1][1]} | {self.gru_report.confusion_matrix[1][2]} |",
            f"| **SLEEPING (2)** | {self.gru_report.confusion_matrix[2][0]} | {self.gru_report.confusion_matrix[2][1]} | {self.gru_report.confusion_matrix[2][2]} |",
            "",
            "#### Rule Baseline (Matched) Confusion Matrix",
            "| True \\ Pred | ACTIVE (0) | DROWSY (1) | SLEEPING (2) |",
            "|---|---|---|---|",
            f"| **ACTIVE (0)** | {self.baseline_matched_report.confusion_matrix[0][0]} | {self.baseline_matched_report.confusion_matrix[0][1]} | {self.baseline_matched_report.confusion_matrix[0][2]} |",
            f"| **DROWSY (1)** | {self.baseline_matched_report.confusion_matrix[1][0]} | {self.baseline_matched_report.confusion_matrix[1][1]} | {self.baseline_matched_report.confusion_matrix[1][2]} |",
            f"| **SLEEPING (2)** | {self.baseline_matched_report.confusion_matrix[2][0]} | {self.baseline_matched_report.confusion_matrix[2][1]} | {self.baseline_matched_report.confusion_matrix[2][2]} |",
        ])

        if self.split_result.limitation_note:
            lines.extend([
                "",
                "> [!WARNING]",
                f"> **Partitioning Limitation**: {self.split_result.limitation_note}",
            ])

        return "\n".join(lines)


def compare_models(
    records: list[SampleRecord],
    model: GRUTemporalClassifier,
    normalizer: FeatureNormalizer,
    baseline_classifier: RuleBasedBaseline | None = None,
    train_ratio: float = 0.75,
    seed: int = 42,
    is_synthetic: bool = False,
) -> ModelComparisonReport:
    """Compare learned GRU sequence classifier and D3 rule baseline on held-out records.

    Data Hygiene:
        - Partition records by subject-first/session-fallback grouping before windowing.
        - Evaluate both models strictly on the held-out validation/test partition.
        - Evaluate the rule baseline both on all eligible frames and matched to exact window endpoints.

    Args:
        records: List of SampleRecords to partition and evaluate.
        model: Trained GRUTemporalClassifier instance.
        normalizer: Fitted FeatureNormalizer instance.
        baseline_classifier: Optional RuleBasedBaseline (default uses standard thresholds).
        train_ratio: Grouped split proportion (default 0.75).
        seed: Deterministic seed (default 42).
        is_synthetic: Flag indicating whether records are synthetic test fixtures.

    Returns:
        ModelComparisonReport detailing performance metrics and confusion matrices.

    Raises:
        ImportError: If PyTorch is unavailable.
        ValueError: If records are empty or partitions lack usable evaluation windows.
    """
    if not HAS_TORCH or torch is None:
        raise ImportError("PyTorch is required for model comparison. Install it with 'pip install torch'.")

    if not records:
        raise ValueError("No sample records provided for model comparison.")

    if not normalizer.is_fitted:
        raise ValueError("FeatureNormalizer must be fitted prior to model comparison.")

    if baseline_classifier is None:
        baseline_classifier = RuleBasedBaseline()

    # 1. Subject-first / session-fallback grouped split
    _, val_records, split_result = split_records_by_group(
        records,
        train_ratio=train_ratio,
        seed=seed,
    )

    if not val_records:
        raise ValueError(
            f"Held-out validation partition is unavailable: {split_result.limitation_note}"
        )

    # 2. Extract validation windows using the fitted normalizer
    val_windows = extract_windows(val_records, normalizer=normalizer)
    if not val_windows:
        raise ValueError(
            "Validation partition yielded 0 usable windows (segments may be shorter than W=60 "
            "or overlap ambiguous annotations)."
        )

    # 3. Evaluate GRU model on validation windows
    model.eval()
    val_dataset = FatigueWindowDataset(val_windows)

    gru_y_true: list[int] = []
    gru_y_pred: list[int | None] = []

    with torch.no_grad():
        for i in range(len(val_dataset)):
            features, target = val_dataset[i]
            batch_x = features.unsqueeze(0)  # (1, 60, 8)
            logits = model(batch_x)
            pred_id = int(torch.argmax(logits, dim=-1).item())
            gru_y_true.append(int(target.item()))
            gru_y_pred.append(pred_id)

    gru_report = compute_metrics(
        y_true=gru_y_true,
        y_pred=gru_y_pred,
        total_samples=len(gru_y_true),
        ambiguous_count=0,
    )

    # 4. Evaluate D3 rule baseline on matched window endpoints
    # Index validation records by (session_id, frame_index)
    rec_index: dict[tuple[str, int], SampleRecord] = {
        (r.session_id, r.frame_index): r for r in val_records
    }

    matched_base_true: list[int] = []
    matched_base_pred: list[int | None] = []

    for w in val_windows:
        endpoint_key = (w.session_id, w.end_frame_index)
        sample = rec_index.get(endpoint_key)
        if sample is None:
            continue

        matched_base_true.append(w.target)
        rule_res = baseline_classifier.classify(sample)
        matched_base_pred.append(rule_res.label_id if rule_res.has_prediction else None)

    baseline_matched_report = compute_metrics(
        y_true=matched_base_true,
        y_pred=matched_base_pred,
        total_samples=len(matched_base_true),
        ambiguous_count=0,
    )

    # 5. Full frame-by-frame baseline evaluation on all eligible validation records
    baseline_full_report = evaluate_baseline(val_records, baseline_classifier)

    return ModelComparisonReport(
        gru_report=gru_report,
        baseline_matched_report=baseline_matched_report,
        baseline_full_report=baseline_full_report,
        split_result=split_result,
        window_count=len(val_windows),
        is_synthetic=is_synthetic,
    )


def evaluate_recording_directory(
    directory_path: str | Path,
    bundle_path: str | Path | LoadedModelBundle | None = None,
) -> ModelComparisonReport | None:
    """Evaluate and compare models on recordings located in a directory.

    If the directory does not exist or contains no `.jsonl` session files, this function
    returns None and reports the absence of real recording data cleanly without fabricating results.

    Args:
        directory_path: Directory containing recorded session `.jsonl` files.
        bundle_path: Optional model bundle or path to bundle.

    Returns:
        ModelComparisonReport if valid sessions exist, else None.
    """
    dir_path = Path(directory_path)
    if not dir_path.exists():
        return None

    # Search for jsonl session files
    sample_files = list(dir_path.glob("**/samples.jsonl")) + list(dir_path.glob("*.jsonl"))
    if not sample_files:
        return None

    records: list[SampleRecord] = []
    for f in sample_files:
        with open(f, encoding="utf-8") as fh:
            for line in fh:
                line_str = line.strip()
                if not line_str:
                    continue
                try:
                    data = json.loads(line_str)
                    records.append(SampleRecord.from_dict(data))
                except (json.JSONDecodeError, ValueError, KeyError):
                    continue

    if not records:
        return None

    if bundle_path is None:
        raise ValueError("A trained model bundle is required to evaluate recordings.")

    if isinstance(bundle_path, LoadedModelBundle):
        bundle = bundle_path
    else:
        bundle = load_model_bundle(bundle_path)

    return compare_models(
        records=records,
        model=bundle.model,
        normalizer=bundle.normalizer,
        is_synthetic=False,
    )
