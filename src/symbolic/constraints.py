"""
constraints.py

Constraint registry for the symbolic reasoning layer.

Each constraint is a named rule that can be added to or removed from the
active set at runtime.  Constraints are stored in two forms:

1. asp_fragment  – the ASP integrity constraint text that is injected into
                   the clingo program.  This is the authoritative, machine-
                   checkable form.
2. description   – a human-readable explanation used in proof traces and
                   error messages.
3. python_check  – a lightweight Python function for fast pre-screening
                   before invoking the clingo solver.  Must be consistent
                   with the ASP fragment (same semantics, cheaper to run).
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional


@dataclass
class Constraint:
    """A single named integrity constraint."""
    name: str
    description: str
    asp_fragment: str
    python_check: Callable[[int, str, int], bool]  # (d1, op, d2) -> True iff satisfied


# ---------------------------------------------------------------------------
# Built-in constraint definitions
# ---------------------------------------------------------------------------

def _no_div_zero(d1: int, op: str, d2: int) -> bool:
    return not (op in ('÷', '/') and d2 == 0)


def _valid_digit_range(d1: int, op: str, d2: int) -> bool:
    return 0 <= d1 <= 9 and 0 <= d2 <= 9


def _valid_operator(d1: int, op: str, d2: int) -> bool:
    return op in ('+', '-', '×', '*', '÷', '/')


NO_DIVISION_BY_ZERO = Constraint(
    name="no_division_by_zero",
    description="The divisor must not be zero.",
    asp_fragment=":- digit(D1), digit(D2), D2 = 0, valid_syntax(D1, divide, D2).",
    python_check=_no_div_zero,
)

VALID_DIGIT_RANGE = Constraint(
    name="valid_digit_range",
    description="Both operands must be single digits in [0, 9].",
    asp_fragment="",   # Enforced by digit(0..9) domain declaration in KB.
    python_check=_valid_digit_range,
)

VALID_OPERATOR = Constraint(
    name="valid_operator",
    description="The operator must be one of +, -, ×, ÷.",
    asp_fragment="",   # Enforced by operator/1 domain declaration in KB.
    python_check=_valid_operator,
)

# Default constraint set shipped with the arithmetic KB.
DEFAULT_CONSTRAINTS: List[Constraint] = [
    NO_DIVISION_BY_ZERO,
    VALID_DIGIT_RANGE,
    VALID_OPERATOR,
]


# ---------------------------------------------------------------------------
# Constraint registry
# ---------------------------------------------------------------------------

class ConstraintRegistry:
    """
    Runtime registry of active integrity constraints.

    Usage
    -----
    registry = ConstraintRegistry()
    registry.add(NO_DIVISION_BY_ZERO)

    ok, violated = registry.check(d1=7, op='÷', d2=0)
    # ok == False, violated == ['no_division_by_zero']
    """

    def __init__(self, defaults: Optional[List[Constraint]] = None):
        self._constraints: Dict[str, Constraint] = {}
        for c in (defaults or DEFAULT_CONSTRAINTS):
            self.add(c)

    def add(self, constraint: Constraint) -> None:
        self._constraints[constraint.name] = constraint

    def remove(self, name: str) -> None:
        self._constraints.pop(name, None)

    def check(self, d1: int, op: str, d2: int) -> tuple:
        """
        Run all registered Python checks.

        Returns (is_valid, violated_names).
        Fast path — does not invoke clingo.
        """
        violated = [
            c.name
            for c in self._constraints.values()
            if not c.python_check(d1, op, d2)
        ]
        return len(violated) == 0, violated

    def asp_fragments(self) -> List[str]:
        """Return all non-empty ASP constraint fragments for KB injection."""
        return [c.asp_fragment for c in self._constraints.values() if c.asp_fragment]

    def list_constraints(self) -> List[str]:
        return list(self._constraints.keys())

    def describe(self, name: str) -> str:
        c = self._constraints.get(name)
        return c.description if c else f"Unknown constraint: {name}"
