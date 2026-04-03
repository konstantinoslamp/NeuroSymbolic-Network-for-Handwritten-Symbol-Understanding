"""
deduction.py

Forward reasoning engine for the symbolic layer.

Given a symbol sequence produced by the neural module, this engine:
1. Pre-screens structural validity (length, type) in Python.
2. Runs all registered integrity constraints via the ConstraintRegistry.
3. Delegates arithmetic evaluation to the KnowledgeBase (clingo for +/−/×,
   Python rational arithmetic for ÷).

This is a genuine symbolic reasoner: the rules that produce the result live in
the ASP KB, not in procedural Python arithmetic.
"""

from fractions import Fraction
from typing import Any, Dict, List, Optional

from src.symbolic.knowledge_base import KnowledgeBase
from src.symbolic.constraints import ConstraintRegistry

_VALID_OPS = ('+', '-', '×', '*', '÷', '/')


class DeductionEngine:
    """
    Forward reasoning over a symbol sequence.

    Parameters
    ----------
    kb : KnowledgeBase
        Shared KB instance (stateless).
    registry : ConstraintRegistry
        Active integrity constraints.
    """

    def __init__(
        self,
        kb: Optional[KnowledgeBase] = None,
        registry: Optional[ConstraintRegistry] = None,
    ) -> None:
        self.kb = kb or KnowledgeBase()
        self.registry = registry or ConstraintRegistry()

    def run(self, symbols: List[str]) -> Dict[str, Any]:
        """
        Validate and evaluate a symbol sequence.

        Returns
        -------
        dict:
            valid          – bool
            result         – float | None
            derivation     – proof-step strings
            contradictions – violated constraint names
            intermediate_states – debug snapshots
        """
        out: Dict[str, Any] = {
            'valid': False,
            'result': None,
            'derivation': [],
            'contradictions': [],
            'intermediate_states': [],
        }

        # ── 1. Structural check ───────────────────────────────────────────
        if len(symbols) != 3:
            out['contradictions'].append('invalid_expression_length')
            out['derivation'].append(
                f"Expected 3 symbols (digit op digit), got {len(symbols)}: {symbols}"
            )
            return out

        raw_d1, raw_op, raw_d2 = symbols

        # ── 2. Type check ─────────────────────────────────────────────────
        if not (raw_d1.isdigit() and raw_d2.isdigit()):
            out['contradictions'].append('non_digit_operand')
            out['derivation'].append(
                f"Operands must be single-digit characters; got ({raw_d1!r}, {raw_d2!r})"
            )
            return out

        if raw_op not in _VALID_OPS:
            out['contradictions'].append('invalid_operator')
            out['derivation'].append(f"Unknown operator: {raw_op!r}")
            return out

        d1, d2 = int(raw_d1), int(raw_d2)

        # ── 3. Integrity constraint screening (Python layer) ──────────────
        pre_ok, pre_violated = self.registry.check(d1, raw_op, d2)
        if not pre_ok:
            out['contradictions'].extend(pre_violated)
            out['derivation'].append(f"Constraint violation(s): {pre_violated}")
            out['intermediate_states'].append(
                {'stage': 'constraint_prescreen', 'violated': pre_violated}
            )
            return out

        out['derivation'].append(
            f"Constraints satisfied: {self.registry.list_constraints()}"
        )

        # ── 4. Arithmetic deduction via KB ────────────────────────────────
        if raw_op in ('÷', '/'):
            # Division: Python rational arithmetic (clingo is integer-only).
            frac = Fraction(d1, d2)
            numeric_result = float(frac)
            out['derivation'].append(
                f"Python ⊢ {d1} ÷ {d2} = {frac} (exact rational)"
            )
            out['derivation'].append(f"  = {numeric_result}")
        else:
            # Integer operators: delegate to clingo KB.
            kb_result = self.kb.deduce(d1, raw_op, d2)
            if kb_result is None:
                out['contradictions'].append('kb_produced_no_result')
                out['derivation'].append(
                    f"KB returned no result for ({d1}, {raw_op}, {d2})"
                )
                return out
            numeric_result = kb_result
            out['derivation'].append(
                f"KB ⊢ result({d1}, {raw_op}, {d2}, {int(numeric_result)})"
            )
            out['derivation'].append(f"  = {numeric_result}")

        out['intermediate_states'].append(
            {'stage': 'kb_deduction', 'result': numeric_result}
        )

        out['valid'] = True
        out['result'] = numeric_result
        return out
