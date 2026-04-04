"""
Symbolic Module Interface - Implements symbolic_deduction and symbolic_abduction
Based on Tsamoura & Michael paper

Architecture (post-P1.1 refactor)
----------------------------------
SymbolicModule          – unchanged abstract base class
DatalogArithmeticModule – NEW concrete implementation backed by a real clingo
                          ASP solver (knowledge_base.py + deduction.py +
                          abduction.py).  This replaces the Python-conditional
                          ArithmeticSymbolicModule.
ArithmeticSymbolicModule – alias kept for backward compatibility with existing
                           training scripts and tests.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
import numpy as np

from src.symbolic.knowledge_base import KnowledgeBase
from src.symbolic.constraints import ConstraintRegistry
from src.symbolic.deduction import DeductionEngine
from src.symbolic.abduction import AbductionEngine

class SymbolicModule(ABC):
    """
    Abstract base class for symbolic reasoning components
    
    Provides two key operations:
    1. symbolic_deduction: Apply rules to derive conclusions
    2. symbolic_abduction: Find explanations/corrections for desired outputs
    """
    
    @abstractmethod
    def symbolic_deduction(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward reasoning: apply rules to derive what is logically implied
        
        Args:
            input_state: {
                'symbols': List[str],  # e.g., ['3', '+', '5']
                'confidences': List[float],  # Neural confidence per symbol
                'constraints': List[str]  # Active constraints
            }
            
        Returns:
            {
                'valid': bool,  # Does expression follow grammar?
                'result': Optional[float],  # Evaluated result (if valid)
                'derivation': List[str],  # Proof steps taken
                'contradictions': List[str],  # Violated constraints
                'intermediate_states': List[Dict]  # For debugging
            }
        """
        pass
    
    @abstractmethod
    def symbolic_abduction(self, 
                          desired_output: float,
                          current_state: Dict[str, Any],
                          neural_probs: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
        """
        Backward reasoning: find all valid explanations for desired output
        
        This is the KEY innovation from the paper - using ALL abductions,
        not just one (unlike ABL).
        
        Args:
            desired_output: Ground truth result (e.g., 8)
            current_state: {
                'symbols': ['3', '+', '6'],  # Current (wrong) prediction
                'positions': [0, 1, 2],
                'confidences': [0.9, 0.85, 0.92]
            }
            neural_probs: {
                'position_0': (10,) array,  # P(digit | image at position 0)
                'position_1': (4,) array,   # P(operator | image at position 1)
                'position_2': (10,) array
            }
            
        Returns:
            List of abductive hypotheses, sorted by plausibility:
            [
                {
                    'correction': ['3', '+', '5'],  # Corrected symbols
                    'changed_positions': [2],  # Which positions were modified
                    'plausibility': 0.85,  # Based on neural probs
                    'explanation': "Changed symbol at position 2 from '6' to '5'",
                    'derivation': ['3 + 5', '= 8']  # Proof that this works
                },
                {
                    'correction': ['2', '+', '6'],
                    'changed_positions': [0],
                    'plausibility': 0.62,
                    'explanation': "Changed symbol at position 0 from '3' to '2'"
                },
                ...
            ]
        """
        pass
    
    @abstractmethod
    def add_constraint(self, constraint_name: str, constraint_fn):
        """
        Add integrity constraint
        
        Example:
            def no_div_zero(symbols):
                return '/' not in symbols or symbols[symbols.index('/')+1] != '0'
            
            module.add_constraint('no_division_by_zero', no_div_zero)
        """
        pass
    
    @abstractmethod
    def get_rules(self) -> List[str]:
        """Return list of active symbolic rules"""
        pass


class DatalogArithmeticModule(SymbolicModule):
    """
    Concrete implementation of SymbolicModule backed by a real ASP knowledge
    base (clingo 5.x) rather than Python conditionals.

    Deduction  – fires the KB to check constraints and derive a result.
    Abduction  – uses clingo's enumeration mode to find *all* (d1, op, d2)
                 that satisfy the target, then scores them by neural
                 probability.

    This satisfies P1.1 of the PhD roadmap.
    """

    def __init__(self):
        self._kb       = KnowledgeBase()
        self._registry = ConstraintRegistry()
        self._deduction_engine = DeductionEngine(self._kb, self._registry)
        self._abduction_engine = AbductionEngine(self._kb, self._registry)

    # ------------------------------------------------------------------ #
    # SymbolicModule interface                                             #
    # ------------------------------------------------------------------ #

    def symbolic_deduction(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Forward reasoning: validate and evaluate symbol sequence via KB.

        Accepts the same input_state dict as the original module so the
        training loop needs no changes.
        """
        symbols = input_state['symbols']
        return self._deduction_engine.run(symbols)

    def symbolic_abduction(
        self,
        desired_output: float,
        current_state: Dict[str, Any],
        neural_probs: Dict[str, np.ndarray],
    ) -> List[Any]:
        """
        Backward reasoning: enumerate all KB-consistent expressions that
        evaluate to desired_output, ranked by neural plausibility.

        Returns a list of dicts (rich form) *or* a list of symbol lists
        (legacy form) depending on whether neural_probs is supplied.

        The training_loop.py consumes the 'correction' key from each dict,
        so it is compatible with both the old and new calling conventions.
        """
        # Convert neural_probs from the trainer format if needed.
        # Trainer passes {} when it hasn't wired up per-position probs yet.
        probs = neural_probs if neural_probs else None

        results = self._abduction_engine.run(desired_output, probs)

        # Backward-compatible: if caller expects plain lists, unwrap.
        # Detect by checking whether the training loop passes empty dict.
        if not neural_probs:
            # Legacy mode: return list of symbol lists (same as before).
            return [r['correction'] for r in results]

        return results

    def add_constraint(self, constraint_name: str, constraint_fn) -> None:
        """
        Register a runtime Python constraint (legacy API).

        New constraints should be added as Constraint objects via
        self._registry.add(Constraint(...)), but this shim keeps backward
        compatibility with test code that calls add_constraint with a
        bare function.
        """
        from src.symbolic.constraints import Constraint

        def _safe_check(d1: int, op: str, d2: int) -> bool:
            # Wrap the old-style function signature (symbols: List[str]).
            symbols = [str(d1), op, str(d2)]
            try:
                return bool(constraint_fn(symbols))
            except Exception:
                return True   # Unknown constraint → non-blocking

        self._registry.add(Constraint(
            name=constraint_name,
            description=f"User-defined constraint: {constraint_name}",
            asp_fragment="",    # Python-only; not injected into ASP.
            python_check=_safe_check,
        ))

    def get_rules(self) -> List[str]:
        """Return names of all active rules and constraints."""
        return self._registry.list_constraints() + [
            'arithmetic_deduction_plus',
            'arithmetic_deduction_minus',
            'arithmetic_deduction_times',
            'arithmetic_deduction_divide',
        ]


# ---------------------------------------------------------------------------
# MATH(n) Variable-Length Symbolic Module (P2.1)
# ---------------------------------------------------------------------------

class MathNSymbolicModule(SymbolicModule):
    """
    Symbolic module for variable-length arithmetic expressions (MATH(n)).

    Handles expressions of length 3, 5, 7, ... with proper operator precedence.
    Uses ExpressionParser (precedence-climbing) instead of flat evaluation.

    For length-3 expressions, delegates to the original DatalogArithmeticModule
    for full clingo-backed reasoning.
    """

    def __init__(self, expression_length: int = 3):
        self.expression_length = expression_length
        self._length3_module = DatalogArithmeticModule()

        from src.symbolic.expression_parser import (
            MathNDeductionEngine, MathNAbductionEngine
        )
        self._mathn_deduction = MathNDeductionEngine()
        self._mathn_abduction = MathNAbductionEngine()

    def symbolic_deduction(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        symbols = input_state['symbols']
        if len(symbols) == 3:
            return self._length3_module.symbolic_deduction(input_state)
        return self._mathn_deduction.run(symbols)

    def symbolic_abduction(
        self,
        desired_output: float,
        current_state: Dict[str, Any],
        neural_probs: Dict[str, np.ndarray],
    ) -> List[Any]:
        symbols = current_state.get('symbols', [])
        seq_len = len(symbols) if symbols else self.expression_length

        if seq_len == 3 and not neural_probs:
            return self._length3_module.symbolic_abduction(
                desired_output, current_state, neural_probs
            )

        probs = neural_probs if neural_probs else None
        return self._mathn_abduction.run(desired_output, probs, seq_len)

    def add_constraint(self, constraint_name: str, constraint_fn):
        self._length3_module.add_constraint(constraint_name, constraint_fn)

    def get_rules(self) -> List[str]:
        return self._length3_module.get_rules() + [
            'operator_precedence',
            'variable_length_parsing',
        ]


# ---------------------------------------------------------------------------
# Backward-compatibility alias
# ---------------------------------------------------------------------------

# All existing training scripts and tests import ArithmeticSymbolicModule.
# Pointing it at the new DatalogArithmeticModule requires zero changes there.
ArithmeticSymbolicModule = DatalogArithmeticModule