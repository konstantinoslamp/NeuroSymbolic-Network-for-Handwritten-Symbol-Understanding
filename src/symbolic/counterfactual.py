"""
Counterfactual Explanations via Abduction — P3.3

Leverages the symbolic abduction engine to generate counterfactual
explanations for incorrect predictions, providing a direct bridge
to the Explainable AI (XAI) literature.

Key insight: when the neural module predicts wrongly, the abductive
correction IS a counterfactual explanation:

  "The model predicted 3 + 6 = 9, but the correct answer is 8.
   The model would be correct if digit at position 2 were 5 instead of 6,
   giving 3 + 5 = 8."

This provides:
  - Human-readable explanations of model failures
  - Minimal change sets (which symbols need to change)
  - Confidence-weighted alternatives
  - Symbolic derivation proofs for each counterfactual
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


# ---------------------------------------------------------------------------
# Symbol mapping
# ---------------------------------------------------------------------------

SYMBOL_TO_IDX = {str(i): i for i in range(10)}
SYMBOL_TO_IDX.update({'+': 10, '-': 11, '×': 12, '÷': 13})
IDX_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_IDX.items()}

POSITION_NAMES = {0: 'first operand', 1: 'operator', 2: 'second operand'}


# ---------------------------------------------------------------------------
# Counterfactual Explanation
# ---------------------------------------------------------------------------

class CounterfactualExplanation:
    """
    A single counterfactual explanation for an incorrect prediction.

    Encodes: "If [changes] had been different, the prediction would be correct."
    """

    def __init__(self, original_symbols: List[str],
                 corrected_symbols: List[str],
                 original_result: Optional[float],
                 corrected_result: float,
                 target_result: float,
                 neural_confidence: Optional[Dict] = None,
                 plausibility: float = 0.0):
        self.original_symbols = original_symbols
        self.corrected_symbols = corrected_symbols
        self.original_result = original_result
        self.corrected_result = corrected_result
        self.target_result = target_result
        self.neural_confidence = neural_confidence
        self.plausibility = plausibility

        # Compute changed positions
        self.changed_positions = []
        for i, (orig, corr) in enumerate(zip(original_symbols, corrected_symbols)):
            if orig != corr:
                self.changed_positions.append(i)

        self.num_changes = len(self.changed_positions)

    @property
    def is_minimal(self) -> bool:
        """Whether this is a minimal counterfactual (fewest changes)."""
        return self.num_changes <= 1

    def to_natural_language(self) -> str:
        """
        Generate a human-readable natural language explanation.

        Example:
          "The model predicted 3 + 6 = 9, but the target was 8.
           If the second operand were 5 instead of 6, then 3 + 5 = 8."
        """
        orig_expr = ' '.join(self.original_symbols)
        corr_expr = ' '.join(self.corrected_symbols)

        orig_res_str = (f"{self.original_result}" if self.original_result is not None
                        else "invalid")

        lines = [
            f"The model predicted {orig_expr} = {orig_res_str}, "
            f"but the target was {self.target_result}."
        ]

        if not self.changed_positions:
            lines.append("No changes needed (prediction is correct).")
        elif len(self.changed_positions) == 1:
            pos = self.changed_positions[0]
            pos_name = POSITION_NAMES.get(pos, f"position {pos}")
            orig_sym = self.original_symbols[pos]
            corr_sym = self.corrected_symbols[pos]
            lines.append(
                f"If the {pos_name} were '{corr_sym}' instead of '{orig_sym}', "
                f"then {corr_expr} = {self.corrected_result}."
            )
        else:
            changes = []
            for pos in self.changed_positions:
                pos_name = POSITION_NAMES.get(pos, f"position {pos}")
                changes.append(
                    f"the {pos_name} were '{self.corrected_symbols[pos]}' "
                    f"instead of '{self.original_symbols[pos]}'"
                )
            lines.append(
                f"If {' and '.join(changes)}, then {corr_expr} = {self.corrected_result}."
            )

        if self.neural_confidence:
            for pos in self.changed_positions:
                orig_sym = self.original_symbols[pos]
                corr_sym = self.corrected_symbols[pos]
                orig_conf = self.neural_confidence.get(f'pos_{pos}_orig', 0)
                corr_conf = self.neural_confidence.get(f'pos_{pos}_corr', 0)
                lines.append(
                    f"  Neural confidence: P('{orig_sym}') = {orig_conf:.3f}, "
                    f"P('{corr_sym}') = {corr_conf:.3f}"
                )

        return "\n".join(lines)

    def to_formal(self) -> str:
        """
        Generate a formal logical representation.

        Example:
          CF: {pos_2: 6 -> 5} |- result(3, +, 5) = 8 = target
        """
        changes = []
        for pos in self.changed_positions:
            changes.append(f"pos_{pos}: {self.original_symbols[pos]} -> "
                          f"{self.corrected_symbols[pos]}")

        if not changes:
            return "CORRECT (no counterfactual needed)"

        change_str = ", ".join(changes)
        corr_expr = "".join(self.corrected_symbols)

        return (f"CF: {{{change_str}}} |- "
                f"result({corr_expr}) = {self.corrected_result} = target")

    def to_dict(self) -> Dict:
        """Structured representation for programmatic use."""
        return {
            'original_expression': self.original_symbols,
            'corrected_expression': self.corrected_symbols,
            'original_result': self.original_result,
            'corrected_result': self.corrected_result,
            'target_result': self.target_result,
            'changed_positions': self.changed_positions,
            'num_changes': self.num_changes,
            'is_minimal': self.is_minimal,
            'plausibility': self.plausibility,
            'natural_language': self.to_natural_language(),
            'formal': self.to_formal(),
        }


# ---------------------------------------------------------------------------
# Counterfactual Generator
# ---------------------------------------------------------------------------

class CounterfactualGenerator:
    """
    Generates counterfactual explanations for incorrect predictions.

    Uses the symbolic abduction engine to find ALL valid corrections,
    then ranks and formats them as counterfactual explanations.

    Prioritization:
      1. Minimal changes (fewest positions modified)
      2. High neural plausibility (the correction is close to what the CNN saw)
      3. Simple operations (prefer + over ÷)
    """

    def __init__(self, symbolic_module):
        """
        Args:
            symbolic_module: a SymbolicModule (e.g., ArithmeticSymbolicModule)
        """
        self.symbolic = symbolic_module

    def explain(self, predicted_symbols: List[str],
                target_result: float,
                neural_probs: Optional[np.ndarray] = None,
                top_k: int = 5) -> List[CounterfactualExplanation]:
        """
        Generate counterfactual explanations for a wrong prediction.

        Args:
            predicted_symbols: e.g. ['3', '+', '6'] (wrong prediction)
            target_result: desired result, e.g. 8.0
            neural_probs: (seq_len, num_classes) neural output probabilities
            top_k: maximum number of counterfactuals to return

        Returns:
            List of CounterfactualExplanation objects, sorted by quality.
        """
        # Get original prediction result
        deduction = self.symbolic.symbolic_deduction({'symbols': predicted_symbols})
        original_result = deduction.get('result')

        # Check if prediction is actually correct
        if (original_result is not None and target_result is not None and
                abs(original_result - target_result) < 0.01):
            return []  # No counterfactual needed

        # Get all valid corrections via abduction
        abductions = self.symbolic.symbolic_abduction(
            desired_output=target_result,
            current_state={'symbols': predicted_symbols},
            neural_probs={},
        )

        if not abductions:
            return []

        # Convert abductions to counterfactual explanations
        explanations = []
        for abd in abductions:
            # Handle both dict and list formats
            if isinstance(abd, dict):
                corrected = abd.get('correction', abd.get('symbols', []))
                plausibility = abd.get('plausibility', 0.0)
            elif isinstance(abd, list):
                corrected = abd
                plausibility = 0.0
            else:
                continue

            if len(corrected) != len(predicted_symbols):
                continue

            # Compute corrected result
            corr_deduction = self.symbolic.symbolic_deduction({'symbols': corrected})
            corr_result = corr_deduction.get('result')
            if corr_result is None:
                continue

            # Build neural confidence info
            confidence_info = {}
            if neural_probs is not None:
                for pos in range(len(predicted_symbols)):
                    orig_sym = predicted_symbols[pos]
                    corr_sym = corrected[pos]
                    orig_idx = SYMBOL_TO_IDX.get(orig_sym)
                    corr_idx = SYMBOL_TO_IDX.get(corr_sym)

                    if orig_idx is not None and pos < len(neural_probs):
                        confidence_info[f'pos_{pos}_orig'] = float(
                            neural_probs[pos][orig_idx])
                    if corr_idx is not None and pos < len(neural_probs):
                        confidence_info[f'pos_{pos}_corr'] = float(
                            neural_probs[pos][corr_idx])

            cf = CounterfactualExplanation(
                original_symbols=predicted_symbols,
                corrected_symbols=corrected,
                original_result=original_result,
                corrected_result=corr_result,
                target_result=target_result,
                neural_confidence=confidence_info,
                plausibility=plausibility,
            )
            explanations.append(cf)

        # Sort: minimal changes first, then by plausibility
        explanations.sort(
            key=lambda cf: (cf.num_changes, -cf.plausibility)
        )

        return explanations[:top_k]

    def explain_batch(self, predicted_batch: List[List[str]],
                      targets: List[float],
                      probs_batch: Optional[np.ndarray] = None,
                      top_k: int = 3) -> List[List[CounterfactualExplanation]]:
        """
        Generate counterfactual explanations for a batch.

        Returns:
            List of explanation lists, one per sample.
        """
        results = []
        for i, (pred, target) in enumerate(zip(predicted_batch, targets)):
            if target is None:
                results.append([])
                continue

            probs = probs_batch[i] if probs_batch is not None else None
            explanations = self.explain(pred, target, probs, top_k)
            results.append(explanations)

        return results

    def print_explanations(self, explanations: List[CounterfactualExplanation],
                           header: str = ""):
        """Pretty-print counterfactual explanations."""
        if header:
            print(f"\n{'='*60}")
            print(f"  {header}")
            print(f"{'='*60}")

        if not explanations:
            print("  No counterfactuals generated (prediction is correct or no valid correction found)")
            return

        for i, cf in enumerate(explanations):
            print(f"\n  --- Counterfactual #{i+1} (changes: {cf.num_changes}, "
                  f"plausibility: {cf.plausibility:.3f}) ---")
            print(f"  {cf.to_natural_language()}")
            print(f"  Formal: {cf.to_formal()}")


# ---------------------------------------------------------------------------
# Batch Analysis
# ---------------------------------------------------------------------------

class CounterfactualAnalyzer:
    """
    Analyzes counterfactual explanations across a dataset to identify
    systematic error patterns.

    Produces reports like:
      - "Position 2 (second operand) is the most commonly corrected position"
      - "The model confuses 3 and 8 most frequently"
      - "Division expressions have the highest correction rate"
    """

    def __init__(self):
        self.all_explanations = []
        self.position_corrections = {0: [], 1: [], 2: []}
        self.symbol_confusions = {}  # (predicted, corrected) -> count
        self.operator_errors = {}    # operator -> count

    def add(self, explanations: List[CounterfactualExplanation]):
        """Add a batch of explanations for analysis."""
        for cf in explanations:
            self.all_explanations.append(cf)

            for pos in cf.changed_positions:
                if pos < len(cf.original_symbols):
                    orig = cf.original_symbols[pos]
                    corr = cf.corrected_symbols[pos]

                    if pos < 3:
                        self.position_corrections[pos].append((orig, corr))

                    key = (orig, corr)
                    self.symbol_confusions[key] = self.symbol_confusions.get(key, 0) + 1

            # Track operator-specific errors
            if len(cf.original_symbols) >= 2:
                op = cf.original_symbols[1]
                self.operator_errors[op] = self.operator_errors.get(op, 0) + 1

    def get_most_corrected_position(self) -> Tuple[int, int]:
        """Return (position_index, correction_count) for most error-prone position."""
        counts = {pos: len(corrs) for pos, corrs in self.position_corrections.items()}
        if not counts:
            return (0, 0)
        pos = max(counts, key=counts.get)
        return pos, counts[pos]

    def get_top_confusions(self, top_k: int = 10) -> List[Tuple[str, str, int]]:
        """Return most common (predicted, corrected) symbol confusions."""
        sorted_conf = sorted(self.symbol_confusions.items(),
                            key=lambda x: x[1], reverse=True)
        return [(k[0], k[1], v) for k, v in sorted_conf[:top_k]]

    def get_error_rate_by_operator(self) -> Dict[str, float]:
        """Error rate broken down by operator type."""
        total = len(self.all_explanations)
        if total == 0:
            return {}
        return {op: count / total for op, count in self.operator_errors.items()}

    def print_report(self):
        """Print full analysis report."""
        print("\n" + "=" * 60)
        print("  COUNTERFACTUAL ANALYSIS REPORT")
        print("=" * 60)

        n = len(self.all_explanations)
        print(f"\n  Total counterfactuals analyzed: {n}")

        if n == 0:
            return

        # Minimal vs non-minimal
        minimal = sum(1 for cf in self.all_explanations if cf.is_minimal)
        print(f"  Minimal (1 change): {minimal} ({minimal/n*100:.1f}%)")
        print(f"  Multi-change: {n - minimal} ({(n-minimal)/n*100:.1f}%)")

        # Most corrected position
        pos, count = self.get_most_corrected_position()
        pos_name = POSITION_NAMES.get(pos, f"position {pos}")
        print(f"\n  Most error-prone position: {pos_name} ({count} corrections)")

        # Position breakdown
        print("\n  Corrections by position:")
        for pos in sorted(self.position_corrections.keys()):
            count = len(self.position_corrections[pos])
            if count > 0:
                pos_name = POSITION_NAMES.get(pos, f"pos {pos}")
                print(f"    {pos_name}: {count}")

        # Top confusions
        top_conf = self.get_top_confusions(8)
        if top_conf:
            print("\n  Top symbol confusions (predicted -> corrected):")
            for orig, corr, count in top_conf:
                print(f"    '{orig}' -> '{corr}': {count} times")

        # Operator errors
        op_rates = self.get_error_rate_by_operator()
        if op_rates:
            print("\n  Error rate by operator:")
            for op, rate in sorted(op_rates.items(), key=lambda x: x[1], reverse=True):
                print(f"    {op}: {rate:.3f}")

        # Average plausibility
        avg_plaus = np.mean([cf.plausibility for cf in self.all_explanations
                            if cf.plausibility > 0])
        if not np.isnan(avg_plaus):
            print(f"\n  Average counterfactual plausibility: {avg_plaus:.3f}")

        print("=" * 60)

    def summary(self) -> Dict:
        """Structured summary for programmatic use."""
        return {
            'total_counterfactuals': len(self.all_explanations),
            'minimal_ratio': (sum(1 for cf in self.all_explanations if cf.is_minimal)
                             / max(len(self.all_explanations), 1)),
            'most_corrected_position': self.get_most_corrected_position(),
            'top_confusions': self.get_top_confusions(5),
            'error_by_operator': self.get_error_rate_by_operator(),
        }
