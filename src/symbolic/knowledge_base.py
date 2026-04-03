"""
knowledge_base.py

Core ASP (Answer Set Programming) Knowledge Base for arithmetic reasoning.
Uses the clingo solver (https://potassco.org/clingo/) as the inference engine.

Design principles
-----------------
* The KB encodes *what is true* (domain facts + arithmetic rules).
  Constraint enforcement (what is forbidden) is handled by the Python-layer
  ConstraintRegistry, which is faster and avoids a well-known ASP pitfall:
  globally-quantified integrity constraints make the whole program UNSAT if
  any grounding satisfies the constraint body — even for facts you are not
  testing.

* clingo uses integer arithmetic internally.  The KB handles +, -, and × in
  pure ASP.  Division requires exact rational arithmetic (e.g. 7/2 = 3.5,
  not 3), so division candidates are enumerated in Python using the
  fractions.Fraction type.

* All public methods return plain Python types; clingo is an implementation
  detail invisible to callers.
"""

import clingo
from fractions import Fraction
from typing import List, Optional, Tuple

# ---------------------------------------------------------------------------
# ASP program fragments
# ---------------------------------------------------------------------------

# Base KB: domain facts + arithmetic deduction rules.
# No integrity constraints here — those live in constraints.py.
_BASE_KB = """
digit(0..9).
operator(plus; minus; times; divide).

% Arithmetic deduction (integer-safe operators only).
result(D1, plus,  D2, R) :- digit(D1), digit(D2), R = D1 + D2.
result(D1, minus, D2, R) :- digit(D1), digit(D2), R = D1 - D2.
result(D1, times, D2, R) :- digit(D1), digit(D2), R = D1 * D2.
"""

# Deduction query: derive the result of one specific (D1, Op, D2).
# The caller substitutes {d1}, {op_atom}, {d2} as integer/atom literals.
_DEDUCTION_QUERY = """
{base_kb}
deduced(R) :- result({d1}, {op_atom}, {d2}, R).
#show deduced/1.
"""

# Abduction query: find ALL (D1, Op, D2) triples whose result equals target.
# We project result/4 down to solution/3; the single answer set contains
# every matching solution atom — no choice rules or enumeration needed.
_ABDUCTION_QUERY = """
{base_kb}
solution(D1, Op, D2) :- result(D1, Op, D2, {target}).
#show solution/3.
"""


# ---------------------------------------------------------------------------
# Operator symbol helpers
# ---------------------------------------------------------------------------

def _op_atom(symbol: str) -> str:
    """Operator character  →  ASP atom name."""
    table = {'+': 'plus', '-': 'minus', '×': 'times', '*': 'times',
             '÷': 'divide', '/': 'divide'}
    atom = table.get(symbol)
    if atom is None:
        raise ValueError(f"Unknown operator symbol: {symbol!r}")
    return atom


def _op_symbol(atom: str) -> str:
    """ASP atom name  →  operator character."""
    table = {'plus': '+', 'minus': '-', 'times': '×', 'divide': '÷'}
    sym = table.get(atom)
    if sym is None:
        raise ValueError(f"Unknown operator atom: {atom!r}")
    return sym


# ---------------------------------------------------------------------------
# KnowledgeBase
# ---------------------------------------------------------------------------

class KnowledgeBase:
    """
    Wrapper around clingo exposing three operations:

    deduce(d1, op, d2)
        Forward reasoning: return the numeric result that the KB derives for
        (d1 op d2), or None if the KB cannot derive one (e.g. unsupported op).

    abduce(target)
        Backward reasoning: return every (d1, op_symbol, d2) triple whose
        KB-derived or Python-rational result equals *target*.

    check_satisfiable(program)
        Low-level: return True iff *program* is satisfiable under clingo.
        Used by the DeductionEngine for fine-grained constraint checking.
    """

    def __init__(self) -> None:
        # Smoke-test: ensure clingo is functional.
        _ctl = clingo.Control()
        _ctl.add("base", [], "digit(0).")
        _ctl.ground([("base", [])])

    # ------------------------------------------------------------------ #
    # deduce                                                               #
    # ------------------------------------------------------------------ #

    def deduce(self, d1: int, op: str, d2: int) -> Optional[float]:
        """
        Derive the result of (d1 op d2) through the KB rules.

        Returns the numeric result, or None if the KB produces no result
        (e.g. the operator is '÷' — handled by the caller via Python).
        """
        if op in ('÷', '/'):
            # Division is handled outside ASP to preserve rational results.
            return None

        program = _DEDUCTION_QUERY.format(
            base_kb=_BASE_KB,
            d1=d1,
            op_atom=_op_atom(op),
            d2=d2,
        )

        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])

        result: Optional[float] = None
        with ctl.solve(yield_=True) as handle:
            for model in handle:
                for atom in model.symbols(shown=True):
                    if atom.name == "deduced":
                        result = float(str(atom.arguments[0]))
        return result

    # ------------------------------------------------------------------ #
    # abduce                                                               #
    # ------------------------------------------------------------------ #

    def abduce(self, target: float) -> List[Tuple[int, str, int]]:
        """
        Enumerate all (d1, op_symbol, d2) triples whose result equals *target*.

        Integer operators (+, -, ×) are enumerated via clingo.
        Division is enumerated via Python rational arithmetic.
        """
        solutions: List[Tuple[int, str, int]] = []

        # ── Integer operators via clingo ──────────────────────────────────
        # clingo only handles integers, so we only attempt this when target
        # is itself an integer value.
        if target == int(target):
            t_int = int(target)
            program = _ABDUCTION_QUERY.format(base_kb=_BASE_KB, target=t_int)

            ctl = clingo.Control(["--warn=none"])
            ctl.add("base", [], program)
            ctl.ground([("base", [])])

            with ctl.solve(yield_=True) as handle:
                for model in handle:
                    for atom in model.symbols(shown=True):
                        if atom.name == "solution":
                            args = atom.arguments
                            d1  = int(str(args[0]))
                            op  = _op_symbol(str(args[1]))
                            d2  = int(str(args[2]))
                            solutions.append((d1, op, d2))

        # ── Division via Python rational arithmetic ───────────────────────
        target_frac = Fraction(target).limit_denominator(1000)
        for d1 in range(10):
            for d2 in range(1, 10):          # d2 != 0 enforced here
                if Fraction(d1, d2) == target_frac:
                    solutions.append((d1, '÷', d2))

        return solutions

    # ------------------------------------------------------------------ #
    # check_satisfiable                                                    #
    # ------------------------------------------------------------------ #

    def check_satisfiable(self, program: str) -> bool:
        """Return True iff *program* is satisfiable under clingo."""
        ctl = clingo.Control(["--warn=none"])
        ctl.add("base", [], program)
        ctl.ground([("base", [])])
        result = ctl.solve()
        return result.satisfiable
