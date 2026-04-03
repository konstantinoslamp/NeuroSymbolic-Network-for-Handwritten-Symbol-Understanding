"""
Full Evaluation Suite for Neuro-Symbolic Arithmetic System

Implements:
  - Per-class accuracy (digits 0-9 and operators separately)
  - Expression-level accuracy (full d op d correct?)
  - Result-level accuracy (arithmetic output correct?)
  - Abduction rate (% of steps requiring abduction)
  - Calibration: ECE (Expected Calibration Error)
  - Confusion matrices
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import defaultdict


# ---------------------------------------------------------------------------
# Symbol constants
# ---------------------------------------------------------------------------

DIGIT_CLASSES = list(range(10))        # 0-9
OPERATOR_CLASSES = list(range(10, 14)) # +, -, ×, ÷
ALL_CLASSES = list(range(14))

IDX_TO_SYMBOL = {i: str(i) for i in range(10)}
IDX_TO_SYMBOL.update({10: '+', 11: '-', 12: '×', 13: '÷'})
SYMBOL_TO_IDX = {v: k for k, v in IDX_TO_SYMBOL.items()}


# ---------------------------------------------------------------------------
# Per-Class Accuracy
# ---------------------------------------------------------------------------

class PerClassAccuracy:
    """Tracks per-class accuracy for digits and operators separately."""

    def __init__(self):
        self.correct = defaultdict(int)
        self.total = defaultdict(int)

    def update(self, predicted_class: int, true_class: int):
        self.total[true_class] += 1
        if predicted_class == true_class:
            self.correct[true_class] += 1

    def update_batch(self, predicted: np.ndarray, true: np.ndarray):
        """Update with batch of predictions. Both arrays are 1D class indices."""
        for p, t in zip(predicted.flatten(), true.flatten()):
            self.update(int(p), int(t))

    def get_accuracy(self, class_idx: int) -> float:
        if self.total[class_idx] == 0:
            return 0.0
        return self.correct[class_idx] / self.total[class_idx]

    def get_digit_accuracy(self) -> Dict[str, float]:
        """Per-digit accuracy (0-9)."""
        return {IDX_TO_SYMBOL[c]: self.get_accuracy(c) for c in DIGIT_CLASSES}

    def get_operator_accuracy(self) -> Dict[str, float]:
        """Per-operator accuracy (+, -, ×, ÷)."""
        return {IDX_TO_SYMBOL[c]: self.get_accuracy(c) for c in OPERATOR_CLASSES}

    def get_overall_digit_accuracy(self) -> float:
        total = sum(self.total[c] for c in DIGIT_CLASSES)
        correct = sum(self.correct[c] for c in DIGIT_CLASSES)
        return correct / max(total, 1)

    def get_overall_operator_accuracy(self) -> float:
        total = sum(self.total[c] for c in OPERATOR_CLASSES)
        correct = sum(self.correct[c] for c in OPERATOR_CLASSES)
        return correct / max(total, 1)

    def summary(self) -> Dict:
        return {
            'digit_accuracy': self.get_digit_accuracy(),
            'operator_accuracy': self.get_operator_accuracy(),
            'overall_digit_accuracy': self.get_overall_digit_accuracy(),
            'overall_operator_accuracy': self.get_overall_operator_accuracy(),
        }


# ---------------------------------------------------------------------------
# Expression-Level Accuracy
# ---------------------------------------------------------------------------

class ExpressionAccuracy:
    """
    Tracks whether the FULL expression (d1 op d2) is recognized correctly.
    All 3 symbols must match for the expression to count as correct.
    """

    def __init__(self):
        self.total = 0
        self.correct = 0
        self.per_operator = defaultdict(lambda: {'correct': 0, 'total': 0})

    def update(self, predicted_symbols: List[str], true_symbols: List[str]):
        """
        Args:
            predicted_symbols: e.g. ['3', '+', '5']
            true_symbols: e.g. ['3', '+', '5']
        """
        self.total += 1
        is_correct = predicted_symbols == true_symbols
        if is_correct:
            self.correct += 1

        # Track per operator type
        if len(true_symbols) >= 3:
            op = true_symbols[1]
            self.per_operator[op]['total'] += 1
            if is_correct:
                self.per_operator[op]['correct'] += 1

    def get_accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    def get_per_operator_accuracy(self) -> Dict[str, float]:
        result = {}
        for op, counts in self.per_operator.items():
            result[op] = counts['correct'] / max(counts['total'], 1)
        return result

    def summary(self) -> Dict:
        return {
            'expression_accuracy': self.get_accuracy(),
            'total_expressions': self.total,
            'correct_expressions': self.correct,
            'per_operator': self.get_per_operator_accuracy(),
        }


# ---------------------------------------------------------------------------
# Result-Level Accuracy
# ---------------------------------------------------------------------------

class ResultAccuracy:
    """
    Tracks whether the arithmetic OUTPUT is correct,
    even if the recognized expression differs from ground truth.
    (e.g., predicting 2+3 instead of 1+4 is still result-correct if target is 5)
    """

    def __init__(self, tolerance: float = 0.01):
        self.total = 0
        self.correct = 0
        self.tolerance = tolerance
        self.per_operator = defaultdict(lambda: {'correct': 0, 'total': 0})

    def update(self, predicted_result: Optional[float], true_result: Optional[float],
               operator: str = None):
        """
        Args:
            predicted_result: computed result from predicted expression (None if invalid)
            true_result: ground truth result (None if invalid expression)
        """
        if true_result is None:
            # Skip invalid ground truth expressions
            return

        self.total += 1
        is_correct = (
            predicted_result is not None
            and abs(predicted_result - true_result) < self.tolerance
        )
        if is_correct:
            self.correct += 1

        if operator:
            self.per_operator[operator]['total'] += 1
            if is_correct:
                self.per_operator[operator]['correct'] += 1

    def get_accuracy(self) -> float:
        return self.correct / max(self.total, 1)

    def summary(self) -> Dict:
        per_op = {}
        for op, counts in self.per_operator.items():
            per_op[op] = counts['correct'] / max(counts['total'], 1)
        return {
            'result_accuracy': self.get_accuracy(),
            'total_evaluated': self.total,
            'correct_results': self.correct,
            'per_operator': per_op,
        }


# ---------------------------------------------------------------------------
# Abduction Rate
# ---------------------------------------------------------------------------

class AbductionTracker:
    """Tracks how often abduction is invoked and its effectiveness."""

    def __init__(self):
        self.total_steps = 0
        self.abduction_used = 0
        self.abduction_helped = 0  # abduction led to correct result
        self.abduction_paths_count = []  # num valid paths per abduction

    def update(self, used_abduction: bool, abduction_helped: bool = False,
               num_paths: int = 0):
        self.total_steps += 1
        if used_abduction:
            self.abduction_used += 1
            self.abduction_paths_count.append(num_paths)
            if abduction_helped:
                self.abduction_helped += 1

    def get_rate(self) -> float:
        return self.abduction_used / max(self.total_steps, 1)

    def get_effectiveness(self) -> float:
        return self.abduction_helped / max(self.abduction_used, 1)

    def get_avg_paths(self) -> float:
        if not self.abduction_paths_count:
            return 0.0
        return np.mean(self.abduction_paths_count)

    def summary(self) -> Dict:
        return {
            'abduction_rate': self.get_rate(),
            'abduction_effectiveness': self.get_effectiveness(),
            'total_steps': self.total_steps,
            'abductions_used': self.abduction_used,
            'abductions_helped': self.abduction_helped,
            'avg_valid_paths': self.get_avg_paths(),
        }


# ---------------------------------------------------------------------------
# Expected Calibration Error (ECE)
# ---------------------------------------------------------------------------

class CalibrationMetrics:
    """
    Computes Expected Calibration Error (ECE) on neural outputs.

    ECE = sum_b (|B_b|/N) * |acc(B_b) - conf(B_b)|

    where B_b are confidence bins, acc is accuracy within the bin,
    and conf is average confidence within the bin.
    """

    def __init__(self, num_bins: int = 15):
        self.num_bins = num_bins
        self.confidences = []
        self.predictions = []
        self.true_labels = []

    def update(self, confidence: float, predicted_class: int, true_class: int):
        self.confidences.append(confidence)
        self.predictions.append(predicted_class)
        self.true_labels.append(true_class)

    def update_batch(self, confidences: np.ndarray, predictions: np.ndarray,
                     true_labels: np.ndarray):
        """Update with batch data. All arrays are 1D."""
        for c, p, t in zip(confidences.flatten(), predictions.flatten(), true_labels.flatten()):
            self.update(float(c), int(p), int(t))

    def compute_ece(self) -> float:
        if not self.confidences:
            return 0.0

        confidences = np.array(self.confidences)
        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)
        n = len(confidences)

        bin_boundaries = np.linspace(0.0, 1.0, self.num_bins + 1)
        ece = 0.0

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            count = np.sum(mask)
            if count == 0:
                continue

            bin_acc = np.mean(predictions[mask] == true_labels[mask])
            bin_conf = np.mean(confidences[mask])
            ece += (count / n) * abs(bin_acc - bin_conf)

        return float(ece)

    def compute_mce(self) -> float:
        """Maximum Calibration Error - worst-case bin miscalibration."""
        if not self.confidences:
            return 0.0

        confidences = np.array(self.confidences)
        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)

        bin_boundaries = np.linspace(0.0, 1.0, self.num_bins + 1)
        mce = 0.0

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            count = np.sum(mask)
            if count == 0:
                continue

            bin_acc = np.mean(predictions[mask] == true_labels[mask])
            bin_conf = np.mean(confidences[mask])
            mce = max(mce, abs(bin_acc - bin_conf))

        return float(mce)

    def get_reliability_diagram_data(self) -> Dict:
        """Returns data for plotting a reliability diagram."""
        if not self.confidences:
            return {'bin_centers': [], 'bin_accuracies': [], 'bin_counts': []}

        confidences = np.array(self.confidences)
        predictions = np.array(self.predictions)
        true_labels = np.array(self.true_labels)

        bin_boundaries = np.linspace(0.0, 1.0, self.num_bins + 1)
        bin_centers = []
        bin_accuracies = []
        bin_counts = []

        for i in range(self.num_bins):
            lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
            mask = (confidences > lo) & (confidences <= hi)
            count = int(np.sum(mask))
            bin_counts.append(count)
            bin_centers.append((lo + hi) / 2)

            if count > 0:
                bin_accuracies.append(float(np.mean(predictions[mask] == true_labels[mask])))
            else:
                bin_accuracies.append(0.0)

        return {
            'bin_centers': bin_centers,
            'bin_accuracies': bin_accuracies,
            'bin_counts': bin_counts,
        }

    def summary(self) -> Dict:
        return {
            'ece': self.compute_ece(),
            'mce': self.compute_mce(),
            'num_samples': len(self.confidences),
            'avg_confidence': float(np.mean(self.confidences)) if self.confidences else 0.0,
            'reliability_diagram': self.get_reliability_diagram_data(),
        }


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

class ConfusionMatrix:
    """Tracks a full confusion matrix for digit and operator classification."""

    def __init__(self, num_classes: int = 14):
        self.num_classes = num_classes
        self.matrix = np.zeros((num_classes, num_classes), dtype=int)

    def update(self, predicted: int, true: int):
        self.matrix[true, predicted] += 1

    def update_batch(self, predicted: np.ndarray, true: np.ndarray):
        for p, t in zip(predicted.flatten(), true.flatten()):
            self.update(int(p), int(t))

    def get_matrix(self) -> np.ndarray:
        return self.matrix.copy()

    def get_precision(self, class_idx: int) -> float:
        col_sum = self.matrix[:, class_idx].sum()
        if col_sum == 0:
            return 0.0
        return self.matrix[class_idx, class_idx] / col_sum

    def get_recall(self, class_idx: int) -> float:
        row_sum = self.matrix[class_idx, :].sum()
        if row_sum == 0:
            return 0.0
        return self.matrix[class_idx, class_idx] / row_sum

    def get_f1(self, class_idx: int) -> float:
        p = self.get_precision(class_idx)
        r = self.get_recall(class_idx)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)

    def summary(self) -> Dict:
        per_class = {}
        for c in range(self.num_classes):
            label = IDX_TO_SYMBOL.get(c, str(c))
            per_class[label] = {
                'precision': self.get_precision(c),
                'recall': self.get_recall(c),
                'f1': self.get_f1(c),
                'support': int(self.matrix[c, :].sum()),
            }
        return {
            'confusion_matrix': self.matrix.tolist(),
            'per_class': per_class,
        }


# ---------------------------------------------------------------------------
# Full Evaluation Runner
# ---------------------------------------------------------------------------

class EvaluationSuite:
    """
    Orchestrates the full evaluation pipeline for the neuro-symbolic system.

    Collects all metrics in a single pass over the evaluation dataset.
    """

    def __init__(self):
        self.per_class = PerClassAccuracy()
        self.expression = ExpressionAccuracy()
        self.result = ResultAccuracy()
        self.abduction = AbductionTracker()
        self.calibration = CalibrationMetrics()
        self.confusion = ConfusionMatrix()

    def evaluate(self, neural_module, symbolic_module, dataset,
                 batch_size: int = 32) -> Dict:
        """
        Run full evaluation on a dataset.

        Args:
            neural_module: the neural module (DigitRecognizer)
            symbolic_module: the symbolic module (ArithmeticSymbolicModule)
            dataset: ExpressionDataset instance
            batch_size: evaluation batch size

        Returns:
            Complete evaluation results dictionary
        """
        symbol_to_idx = {str(i): i for i in range(10)}
        symbol_to_idx.update({'+': 10, '-': 11, '×': 12, '÷': 13})
        idx_to_sym = {v: k for k, v in symbol_to_idx.items()}

        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))

            batch_images = []
            batch_true_symbols = []
            batch_true_results = []

            for idx in range(start, end):
                item = dataset[idx]
                batch_images.append(item['images'])
                batch_true_symbols.append(item.get('text', ''))
                batch_true_results.append(item['result'])

            images = np.stack(batch_images)

            # Neural deduction
            neural_out = neural_module.neural_deduction(images)
            probs = neural_out['probabilities']
            class_ids = neural_out['class_ids']
            confidence = neural_out['confidence']

            for i in range(len(batch_images)):
                # --- Per-class and confusion matrix ---
                true_text = batch_true_symbols[i]
                true_result = batch_true_results[i]

                # Parse true symbols
                if isinstance(true_text, str) and len(true_text) >= 3:
                    true_syms = [true_text[0], true_text[1], true_text[2]]
                elif isinstance(true_text, list):
                    true_syms = true_text
                else:
                    continue

                true_indices = [symbol_to_idx.get(s, -1) for s in true_syms]
                if -1 in true_indices:
                    continue

                pred_ids = class_ids[i]
                if pred_ids.ndim == 0:
                    pred_ids = np.array([pred_ids])
                pred_syms = [idx_to_sym.get(int(c), '?') for c in pred_ids]

                # Per-class accuracy
                for p, t in zip(pred_ids.flatten(), true_indices):
                    self.per_class.update(int(p), int(t))
                    self.confusion.update(int(p), int(t))

                # Calibration
                sample_conf = confidence[i]
                if sample_conf.ndim == 0:
                    sample_conf = np.array([sample_conf])
                for c, p, t in zip(sample_conf.flatten(), pred_ids.flatten(),
                                   true_indices):
                    self.calibration.update(float(c), int(p), int(t))

                # Expression-level accuracy
                self.expression.update(pred_syms, true_syms)

                # Symbolic deduction for result accuracy
                deduction = symbolic_module.symbolic_deduction({'symbols': pred_syms})
                pred_result = deduction.get('result')
                operator = true_syms[1] if len(true_syms) >= 2 else None
                self.result.update(pred_result, true_result, operator)

                # Abduction tracking
                used_abduction = False
                abduction_helped = False
                num_paths = 0

                if true_result is not None and (pred_result is None or
                        abs(pred_result - true_result) > 0.01):
                    # Would need abduction
                    abd_results = symbolic_module.symbolic_abduction(
                        desired_output=true_result,
                        current_state={'symbols': pred_syms},
                        neural_probs={}
                    )
                    if abd_results:
                        used_abduction = True
                        num_paths = len(abd_results)
                        # Check if best abduction gives correct result
                        best = abd_results[0]
                        abd_deduction = symbolic_module.symbolic_deduction(
                            {'symbols': best}
                        )
                        if (abd_deduction['valid'] and abd_deduction['result'] is not None
                                and abs(abd_deduction['result'] - true_result) < 0.01):
                            abduction_helped = True

                self.abduction.update(used_abduction, abduction_helped, num_paths)

        return self.get_results()

    def get_results(self) -> Dict:
        """Compile all metric results into a single dictionary."""
        return {
            'per_class': self.per_class.summary(),
            'expression': self.expression.summary(),
            'result': self.result.summary(),
            'abduction': self.abduction.summary(),
            'calibration': self.calibration.summary(),
            'confusion_matrix': self.confusion.summary(),
        }

    @staticmethod
    def print_report(results: Dict):
        """Pretty-print evaluation results."""
        print("\n" + "=" * 70)
        print("  EVALUATION REPORT")
        print("=" * 70)

        # Per-class accuracy
        pc = results['per_class']
        print("\n--- Per-Class Accuracy ---")
        print(f"  Overall Digit Accuracy:    {pc['overall_digit_accuracy']:.4f}")
        print(f"  Overall Operator Accuracy: {pc['overall_operator_accuracy']:.4f}")
        print("  Digits:")
        for d, acc in pc['digit_accuracy'].items():
            print(f"    {d}: {acc:.4f}")
        print("  Operators:")
        for op, acc in pc['operator_accuracy'].items():
            print(f"    {op}: {acc:.4f}")

        # Expression accuracy
        expr = results['expression']
        print(f"\n--- Expression Accuracy ---")
        print(f"  Full Expression Accuracy: {expr['expression_accuracy']:.4f}")
        print(f"  ({expr['correct_expressions']}/{expr['total_expressions']})")
        print("  Per Operator:")
        for op, acc in expr['per_operator'].items():
            print(f"    {op}: {acc:.4f}")

        # Result accuracy
        res = results['result']
        print(f"\n--- Result Accuracy ---")
        print(f"  Result Accuracy: {res['result_accuracy']:.4f}")
        print(f"  ({res['correct_results']}/{res['total_evaluated']})")

        # Abduction
        abd = results['abduction']
        print(f"\n--- Abduction Statistics ---")
        print(f"  Abduction Rate:          {abd['abduction_rate']:.4f}")
        print(f"  Abduction Effectiveness: {abd['abduction_effectiveness']:.4f}")
        print(f"  Avg Valid Paths:         {abd['avg_valid_paths']:.1f}")

        # Calibration
        cal = results['calibration']
        print(f"\n--- Calibration ---")
        print(f"  ECE: {cal['ece']:.4f}")
        print(f"  MCE: {cal['mce']:.4f}")
        print(f"  Avg Confidence: {cal['avg_confidence']:.4f}")

        # Confusion matrix summary (precision/recall/F1)
        cm = results['confusion_matrix']
        print(f"\n--- Per-Class Precision/Recall/F1 ---")
        print(f"  {'Class':<6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'Support':>8}")
        for label, stats in cm['per_class'].items():
            if stats['support'] > 0:
                print(f"  {label:<6} {stats['precision']:>6.3f} {stats['recall']:>6.3f} "
                      f"{stats['f1']:>6.3f} {stats['support']:>8}")

        print("\n" + "=" * 70)
