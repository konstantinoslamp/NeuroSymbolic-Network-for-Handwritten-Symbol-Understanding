"""
abduction.py

Backward reasoning engine for the symbolic layer.

Implements symbolic_abduction: given a desired output (ground-truth result),
enumerate *all* valid (d1, op, d2) expressions that produce that result, then
rank them by neural plausibility.

This is the key innovation from Tsamoura et al. — using ALL abductive
explanations, not just one, weighted by neural probability to form a proper
semantic loss signal.

The engine delegates symbolic enumeration to the KnowledgeBase (clingo for
+/−/×, Python rational arithmetic for ÷), then scores each explanation using
the neural probability distributions supplied by the caller.
"""

import math
from fractions import Fraction
from typing import Any, Dict, List, Optional

import numpy as np

from src.symbolic.knowledge_base import KnowledgeBase
from src.symbolic.constraints import ConstraintRegistry

# Operator symbol → position-2 class index mapping expected by the trainer.
# Digits occupy indices 0–9; operators 10–13.
_OP_TO_IDX: Dict[str, int] = {'+': 10, '-': 11, '×': 12, '÷': 13}
_IDX_TO_OP: Dict[int, str] = {v: k for k, v in _OP_TO_IDX.items()}


class AbductionEngine:
    """
    Enumerates all valid expressions consistent with a target result and
    scores them by neural plausibility.

    Parameters
    ----------
    kb : KnowledgeBase
        Used for clingo-backed enumeration of +/−/× solutions.
    registry : ConstraintRegistry
        Fast Python constraint pre-screening.
    """

    def __init__(
        self,
        kb: Optional[KnowledgeBase] = None,
        registry: Optional[ConstraintRegistry] = None,
    ):
        self.kb = kb or KnowledgeBase()
        self.registry = registry or ConstraintRegistry()

    def run(
        self,
        target: float,
        neural_probs: Optional[Dict[str, np.ndarray]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Find all expressions (d1, op, d2) that evaluate to *target* and
        score them by neural plausibility.

        Parameters
        ----------
        target : float
            Desired arithmetic result (e.g. 8.0 for "3 + 5").
        neural_probs : dict, optional
            Neural output probabilities keyed by position:
            {
                'position_0': np.ndarray shape (10,),   # P(digit | img_0)
                'position_1': np.ndarray shape (14,),   # P(class | img_1)
                'position_2': np.ndarray shape (10,),   # P(digit | img_2)
            }
            If None, all explanations get equal plausibility (uniform prior).

        Returns
        -------
        list of dicts, sorted descending by 'log_plausibility':
            {
                'correction'      : ['3', '+', '5'],
                'changed_positions': [int, ...],      # vs. current prediction
                'log_plausibility': float,            # sum of log P(symbol)
                'plausibility'    : float,            # exp(log_plausibility)
                'explanation'     : str,
                'derivation'      : [str, ...],
            }
        """
        # ── 1. Enumerate all valid (d1, op, d2) via KB ───────────────────
        raw_solutions = self.kb.abduce(target)     # List[(int, str, int)]

        if not raw_solutions:
            return []

        # ── 2. Score each solution ────────────────────────────────────────
        scored: List[Dict[str, Any]] = []
        for d1, op, d2 in raw_solutions:
            symbols = [str(d1), op, str(d2)]
            log_p = self._log_probability(symbols, neural_probs)

            # Build derivation trace
            frac_result = Fraction(d1) if op != '÷' else Fraction(d1, d2)
            if op == '÷':
                result_str = str(frac_result)
            else:
                ops = {'+': d1 + d2, '-': d1 - d2, '×': d1 * d2}
                result_str = str(ops[op])

            derivation = [
                f"KB ⊢ valid_syntax({d1}, {op}, {d2})",
                f"KB ⊢ result({d1}, {op}, {d2}, {result_str})",
                f"   = {target}  ✓",
            ]

            scored.append({
                'correction': symbols,
                'log_plausibility': log_p,
                'plausibility': math.exp(log_p),
                'explanation': (
                    f"Expression {d1} {op} {d2} = {result_str} "
                    f"satisfies target={target}"
                ),
                'derivation': derivation,
            })

        # ── 3. Sort by descending log-plausibility ────────────────────────
        scored.sort(key=lambda x: x['log_plausibility'], reverse=True)

        # ── 4. Annotate changed_positions vs. the most-likely prediction ──
        if scored:
            best_symbols = scored[0]['correction']
            for entry in scored:
                entry['changed_positions'] = [
                    i for i, (a, b) in enumerate(zip(best_symbols, entry['correction']))
                    if a != b
                ]

        return scored

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _log_probability(
        self,
        symbols: List[str],
        neural_probs: Optional[Dict[str, np.ndarray]],
    ) -> float:
        """
        Compute log P(symbols | neural_probs).

        Falls back to a uniform-prior approximation when neural_probs is
        absent (all digits equally likely ≈ 1/10, all ops ≈ 1/4).
        """
        if neural_probs is None:
            # Uniform prior: log(1/10) for each digit, log(1/4) for operator.
            return math.log(0.1) * 2 + math.log(0.25)

        log_p = 0.0
        for pos_idx, sym in enumerate(symbols):
            key = f'position_{pos_idx}'
            probs = neural_probs.get(key)
            if probs is None:
                # Position not in probs — use uniform.
                log_p += math.log(0.1) if pos_idx != 1 else math.log(0.25)
                continue

            # Determine class index.
            if sym.isdigit():
                class_idx = int(sym)
            else:
                class_idx = _OP_TO_IDX.get(sym)
                if class_idx is None:
                    log_p += math.log(1e-10)
                    continue

            if class_idx >= len(probs):
                log_p += math.log(1e-10)
                continue

            p = float(probs[class_idx])
            log_p += math.log(max(p, 1e-10))

        return log_p
