"""
Expression Parser with Operator Precedence for MATH(n)

Handles variable-length arithmetic expressions (length 3, 5, 7, ...):
  - MATH(3): d op d           (e.g., 3 + 5)
  - MATH(5): d op d op d      (e.g., 3 + 5 × 2)
  - MATH(7): d op d op d op d (e.g., 3 + 5 × 2 - 1)

Implements proper operator precedence:
  1. × and ÷ bind tighter than + and -
  2. Left-to-right evaluation within same precedence level

Uses a recursive descent parser to build an AST, then evaluates bottom-up.
"""

from fractions import Fraction
from typing import List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
# AST Nodes
# ---------------------------------------------------------------------------

class ASTNode:
    """Base class for AST nodes."""
    pass


class NumberNode(ASTNode):
    """Leaf node: a single digit."""

    def __init__(self, value: int):
        self.value = value

    def evaluate(self) -> Fraction:
        return Fraction(self.value)

    def __repr__(self):
        return str(self.value)


class BinOpNode(ASTNode):
    """Binary operation node: left op right."""

    def __init__(self, left: ASTNode, op: str, right: ASTNode):
        self.left = left
        self.op = op
        self.right = right

    def evaluate(self) -> Optional[Fraction]:
        l = self.left.evaluate()
        r = self.right.evaluate()
        if l is None or r is None:
            return None

        if self.op in ('+',):
            return l + r
        elif self.op in ('-',):
            return l - r
        elif self.op in ('×', '*'):
            return l * r
        elif self.op in ('÷', '/'):
            if r == 0:
                return None  # Division by zero
            return Fraction(l, r)
        return None

    def __repr__(self):
        return f"({self.left} {self.op} {self.right})"


# ---------------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------------

PRECEDENCE = {
    '+': 1, '-': 1,
    '×': 2, '*': 2,
    '÷': 2, '/': 2,
}

VALID_OPERATORS = set(PRECEDENCE.keys())


# ---------------------------------------------------------------------------
# Recursive Descent Parser
# ---------------------------------------------------------------------------

class ExpressionParser:
    """
    Parses flat symbol lists into ASTs with correct operator precedence.

    Supports MATH(n) expressions of any odd length >= 3.

    Usage:
        parser = ExpressionParser()
        ast = parser.parse(['3', '+', '5', '×', '2'])
        result = ast.evaluate()  # Fraction(13, 1) because 3 + (5 × 2) = 13
    """

    def __init__(self):
        self.tokens = []
        self.pos = 0

    def parse(self, symbols: List[str]) -> Optional[ASTNode]:
        """
        Parse a flat symbol list into an AST.

        Args:
            symbols: e.g. ['3', '+', '5', '×', '2']

        Returns:
            ASTNode root, or None if parsing fails
        """
        if not self._validate_structure(symbols):
            return None

        self.tokens = symbols
        self.pos = 0

        try:
            ast = self._parse_expression(min_precedence=1)
            if self.pos != len(self.tokens):
                return None  # Unconsumed tokens
            return ast
        except (IndexError, ValueError):
            return None

    def evaluate(self, symbols: List[str]) -> Optional[float]:
        """
        Parse and evaluate a symbol list.

        Returns:
            Float result, or None if invalid
        """
        ast = self.parse(symbols)
        if ast is None:
            return None
        result = ast.evaluate()
        if result is None:
            return None
        return float(result)

    def _validate_structure(self, symbols: List[str]) -> bool:
        """Check that the symbol list has valid structure: d op d op d ..."""
        if len(symbols) < 3 or len(symbols) % 2 == 0:
            return False

        for i, sym in enumerate(symbols):
            if i % 2 == 0:
                # Should be a digit
                if not (sym.isdigit() and len(sym) == 1):
                    return False
            else:
                # Should be an operator
                if sym not in VALID_OPERATORS:
                    return False
        return True

    def _parse_expression(self, min_precedence: int) -> ASTNode:
        """
        Pratt parser / precedence climbing.

        Parses expressions respecting operator precedence.
        """
        left = self._parse_primary()

        while (self.pos < len(self.tokens) and
               self.tokens[self.pos] in VALID_OPERATORS and
               PRECEDENCE[self.tokens[self.pos]] >= min_precedence):

            op = self.tokens[self.pos]
            op_prec = PRECEDENCE[op]
            self.pos += 1

            # Right-associative would use op_prec, left-associative uses op_prec + 1
            right = self._parse_expression(op_prec + 1)
            left = BinOpNode(left, op, right)

        return left

    def _parse_primary(self) -> ASTNode:
        """Parse a primary expression (single digit)."""
        token = self.tokens[self.pos]
        self.pos += 1
        return NumberNode(int(token))


# ---------------------------------------------------------------------------
# MATH(n) Expression Generation Utilities
# ---------------------------------------------------------------------------

def generate_math_n_expression(n: int) -> Tuple[List[str], float]:
    """
    Generate a random valid MATH(n) expression.

    Args:
        n: expression length (must be odd, >= 3). Number of symbols, not digits.

    Returns:
        (symbols, result) where symbols is e.g. ['3', '+', '5'] and result is 8.0
    """
    import numpy as np

    assert n >= 3 and n % 2 == 1, f"n must be odd and >= 3, got {n}"
    num_digits = (n + 1) // 2
    num_ops = n // 2

    operators = ['+', '-', '×', '÷']
    parser = ExpressionParser()

    # Try until we get a valid expression (avoid div by zero etc.)
    for _ in range(1000):
        symbols = []
        for i in range(num_digits):
            d = np.random.randint(0, 10)
            symbols.append(str(d))
            if i < num_ops:
                op = operators[np.random.randint(len(operators))]
                symbols.append(op)

        result = parser.evaluate(symbols)
        if result is not None and not (abs(result) > 1e6):  # Skip overflow
            return symbols, result

    # Fallback: simple addition
    symbols = ['1', '+', '1'] + ['+', '1'] * (num_ops - 1)
    result = float(num_digits)
    return symbols[:n], result


# ---------------------------------------------------------------------------
# Extended Deduction for MATH(n)
# ---------------------------------------------------------------------------

class MathNDeductionEngine:
    """
    Deduction engine for variable-length expressions.

    Unlike the length-3 engine, this uses the ExpressionParser
    to handle operator precedence correctly.
    """

    def __init__(self):
        self.parser = ExpressionParser()

    def run(self, symbols: List[str]) -> dict:
        """
        Validate and evaluate a variable-length symbol sequence.

        Returns same format as DeductionEngine.run().
        """
        out = {
            'valid': False,
            'result': None,
            'derivation': [],
            'contradictions': [],
            'intermediate_states': [],
        }

        if len(symbols) < 3 or len(symbols) % 2 == 0:
            out['contradictions'].append('invalid_expression_length')
            out['derivation'].append(
                f"Expected odd length >= 3, got {len(symbols)}: {symbols}"
            )
            return out

        # Validate structure
        if not self.parser._validate_structure(symbols):
            out['contradictions'].append('invalid_structure')
            out['derivation'].append(f"Invalid structure: {symbols}")
            return out

        # Parse and evaluate
        ast = self.parser.parse(symbols)
        if ast is None:
            out['contradictions'].append('parse_error')
            out['derivation'].append(f"Failed to parse: {symbols}")
            return out

        result = ast.evaluate()
        if result is None:
            out['contradictions'].append('evaluation_error')
            out['derivation'].append(f"Evaluation failed (e.g. div by zero)")
            return out

        out['valid'] = True
        out['result'] = float(result)
        out['derivation'].append(f"AST: {ast}")
        out['derivation'].append(f"Result: {float(result)}")
        out['intermediate_states'].append({'ast': str(ast), 'result': float(result)})

        return out


# ---------------------------------------------------------------------------
# Extended Abduction for MATH(n)
# ---------------------------------------------------------------------------

class MathNAbductionEngine:
    """
    Abduction engine for variable-length expressions.

    For MATH(n) with n > 3, full enumeration is intractable.
    Instead, we use a beam search approach:
      1. Fix the operator positions (most confident)
      2. For each operator configuration, enumerate digit combinations
         using constraint propagation
      3. Score by neural plausibility

    For n=3, falls back to the original AbductionEngine.
    """

    def __init__(self, beam_width: int = 50):
        self.beam_width = beam_width
        self.parser = ExpressionParser()

    def run(self, target: float, neural_probs: Optional[dict],
            seq_len: int = 3) -> List[List[str]]:
        """
        Find valid expressions of length seq_len that evaluate to target.

        Args:
            target: desired result
            neural_probs: dict of position -> probability array
            seq_len: expression length (3, 5, 7)

        Returns:
            List of symbol lists, sorted by neural plausibility
        """
        import numpy as np

        if seq_len == 3:
            return self._abduce_length3(target, neural_probs)

        num_digits = (seq_len + 1) // 2
        num_ops = seq_len // 2
        operators = ['+', '-', '×', '÷']
        digit_range = range(10)

        # Get top-k operator assignments by neural probability
        op_candidates = self._get_top_op_assignments(
            neural_probs, num_ops, seq_len, top_k=min(16, 4 ** num_ops)
        )

        results = []
        for ops in op_candidates:
            # For each operator assignment, find digit combinations
            digit_combos = self._find_digits_for_ops(
                ops, target, num_digits
            )
            for digits in digit_combos:
                symbols = self._interleave(digits, ops)
                score = self._score(symbols, neural_probs)
                results.append((symbols, score))

        # Sort by score, return top beam_width
        results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in results[:self.beam_width]]

    def _abduce_length3(self, target: float, neural_probs) -> List[List[str]]:
        """Delegate length-3 abduction to KB-based engine."""
        from src.symbolic.knowledge_base import KnowledgeBase
        kb = KnowledgeBase()
        solutions = kb.abduce(target)
        result = [[str(d1), op, str(d2)] for d1, op, d2 in solutions]
        # Sort by neural plausibility
        result.sort(key=lambda s: self._score(s, neural_probs), reverse=True)
        return result

    def _get_top_op_assignments(self, neural_probs, num_ops, seq_len, top_k):
        """Get most probable operator assignments from neural output."""
        import numpy as np
        operators = ['+', '-', '×', '÷']
        op_indices = {'+': 10, '-': 11, '×': 12, '÷': 13}

        # Get probabilities for operator positions
        op_probs = []
        for i in range(num_ops):
            pos = 2 * i + 1  # Operator positions: 1, 3, 5, ...
            if neural_probs and f'position_{pos}' in neural_probs:
                probs = neural_probs[f'position_{pos}']
                op_p = [(op, float(probs[idx])) for op, idx in op_indices.items()]
            else:
                op_p = [(op, 0.25) for op in operators]
            op_probs.append(op_p)

        # Generate top-k combinations via beam search
        from itertools import product
        all_combos = list(product(*[[x[0] for x in pos_p] for pos_p in op_probs]))

        def combo_score(combo):
            s = 0.0
            for i, op in enumerate(combo):
                for op_name, p in op_probs[i]:
                    if op_name == op:
                        s += np.log(max(p, 1e-10))
            return s

        all_combos.sort(key=combo_score, reverse=True)
        return [list(c) for c in all_combos[:top_k]]

    def _find_digits_for_ops(self, ops: List[str], target: float,
                             num_digits: int) -> List[List[str]]:
        """
        Given fixed operators, find digit combinations that produce target.

        Uses constrained enumeration with early pruning.
        """
        results = []

        if num_digits <= 3:
            # Enumerate digit combinations (10^num_digits is manageable for small n)
            self._enumerate_digits([], ops, target, num_digits, results, max_results=20)
        else:
            # For larger expressions, use random sampling with validation
            self._sample_digits(ops, target, num_digits, results, max_attempts=500)

        return results

    def _enumerate_digits(self, current_digits, ops, target, num_digits,
                          results, max_results):
        """Recursive enumeration of digit combinations."""
        if len(results) >= max_results:
            return

        if len(current_digits) == num_digits:
            symbols = self._interleave([str(d) for d in current_digits], ops)
            val = self.parser.evaluate(symbols)
            if val is not None and abs(val - target) < 0.01:
                results.append([str(d) for d in current_digits])
            return

        for d in range(10):
            self._enumerate_digits(current_digits + [d], ops, target,
                                   num_digits, results, max_results)

    def _sample_digits(self, ops, target, num_digits, results, max_attempts):
        """Sample random digit combinations and check if they match target."""
        import numpy as np
        seen = set()

        for _ in range(max_attempts):
            if len(results) >= 20:
                break
            digits = [str(np.random.randint(0, 10)) for _ in range(num_digits)]
            key = tuple(digits)
            if key in seen:
                continue
            seen.add(key)

            symbols = self._interleave(digits, ops)
            val = self.parser.evaluate(symbols)
            if val is not None and abs(val - target) < 0.01:
                results.append(digits)

    def _interleave(self, digits: List[str], ops: List[str]) -> List[str]:
        """Interleave digits and operators: [d0, op0, d1, op1, d2]."""
        result = []
        for i, d in enumerate(digits):
            result.append(d)
            if i < len(ops):
                result.append(ops[i])
        return result

    def _score(self, symbols: List[str], neural_probs) -> float:
        """Score a symbol list by neural probability."""
        import numpy as np

        if not neural_probs:
            return 0.0

        op_to_idx = {'+': 10, '-': 11, '×': 12, '÷': 13}
        score = 0.0

        for pos, sym in enumerate(symbols):
            key = f'position_{pos}'
            probs = neural_probs.get(key) if neural_probs else None
            if probs is None:
                continue

            if sym.isdigit():
                idx = int(sym)
            else:
                idx = op_to_idx.get(sym)
                if idx is None:
                    continue

            if idx < len(probs):
                score += np.log(max(float(probs[idx]), 1e-10))

        return score
