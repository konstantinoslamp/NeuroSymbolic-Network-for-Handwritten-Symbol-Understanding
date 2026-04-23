"""
Machine Coaching (MC) Engine — Propositional Prioritized Rule Policies

Implements the Machine Coaching semantics from Michael (2019), as used
in the Evolvable Policies framework (Thoma et al. 2026).

Key concepts:
  - Atoms: binary concepts (e.g., a1, a2, ..., a8)
  - Literals: positive (a1) or negative (-a1)
  - Rules: body (conjunction of literals) → head (single literal, 'h' or '-h')
  - Policy: prioritized sequence of rules (later = higher priority)
  - Decision: determined by highest-priority triggered rule

Open-world assumption: absence of an atom does NOT imply its negation.

Example policy:
  R1 :: -p1, -p2, -p3 implies h;
  R2 :: -p1, -p2, -p3, p4 implies -h;
  R3 :: -p5, -p6 implies -h;
  R4 :: -p5, -p6, p7 implies h;
  R5 :: -p5, -p6, p7, p8 implies -h;
"""

import copy
import random
from typing import List, Dict, Set, Tuple, Optional, Any
from itertools import combinations


# ---------------------------------------------------------------------------
# Literal
# ---------------------------------------------------------------------------

class Literal:
    """A positive or negative atom."""

    def __init__(self, atom: str, positive: bool = True):
        self.atom = atom
        self.positive = positive

    def negated(self) -> 'Literal':
        return Literal(self.atom, not self.positive)

    def holds_in(self, context: Dict[str, bool]) -> bool:
        """Check if this literal holds in a given context."""
        if self.atom not in context:
            return False  # Open-world: unknown → not triggered
        return context[self.atom] == self.positive

    def __eq__(self, other):
        return isinstance(other, Literal) and self.atom == other.atom and self.positive == other.positive

    def __hash__(self):
        return hash((self.atom, self.positive))

    def __repr__(self):
        return self.atom if self.positive else f"-{self.atom}"

    @staticmethod
    def parse(s: str) -> 'Literal':
        """Parse a literal string like 'a1' or '-a2'."""
        s = s.strip()
        if s.startswith('-'):
            return Literal(s[1:], positive=False)
        return Literal(s, positive=True)


# ---------------------------------------------------------------------------
# Rule
# ---------------------------------------------------------------------------

class Rule:
    """
    A propositional rule: body_literals implies head_literal.

    The body is a conjunction of literals. If ALL body literals hold
    in a context, the rule fires and concludes the head.
    """

    def __init__(self, body: List[Literal], head: Literal, name: str = ""):
        self.body = list(body)
        self.head = head
        self.name = name

    def is_triggered(self, context: Dict[str, bool]) -> bool:
        """Check if ALL body literals hold in the context."""
        return all(lit.holds_in(context) for lit in self.body)

    def __repr__(self):
        body_str = ", ".join(str(lit) for lit in self.body)
        return f"{self.name} :: {body_str} implies {self.head}"

    def copy(self) -> 'Rule':
        return Rule(list(self.body), Literal(self.head.atom, self.head.positive), self.name)

    @staticmethod
    def parse(s: str) -> 'Rule':
        """
        Parse a rule string like '-p1, -p2, -p3 implies h'.
        """
        s = s.strip().rstrip(';')
        if '::' in s:
            name, rest = s.split('::', 1)
            name = name.strip()
        else:
            name, rest = "", s

        parts = rest.split('implies')
        body_str = parts[0].strip()
        head_str = parts[1].strip()

        body = [Literal.parse(b) for b in body_str.split(',') if b.strip()]
        head = Literal.parse(head_str)

        return Rule(body, head, name)


# ---------------------------------------------------------------------------
# Policy (Machine Coaching)
# ---------------------------------------------------------------------------

class Policy:
    """
    A prioritized sequence of propositional rules.

    Later rules have HIGHER priority and override earlier ones.
    This is the key MC property enabling elaboration tolerance:
    new rules can be appended to correct/override previous behavior.

    Decision logic:
      1. Find all triggered rules (body holds in context)
      2. Take the LAST (highest-priority) triggered rule
      3. Return its head as the decision
      4. If no rule triggers → ABSTAIN
    """

    def __init__(self, rules: List[Rule] = None):
        self.rules = list(rules) if rules else []

    def add_rule(self, rule: Rule):
        """Append a rule (highest priority)."""
        if not rule.name:
            rule.name = f"R{len(self.rules) + 1}"
        self.rules.append(rule)

    def deduce(self, context: Dict[str, bool]) -> Optional[bool]:
        """
        Forward reasoning: apply the policy to a context.

        Args:
            context: mapping atom_name → True/False

        Returns:
            True if decision is 'h' (head)
            False if decision is '-h' (negative head)
            None if no rule triggers (ABSTAIN)
        """
        decision = None

        # Iterate in order; last triggered rule wins (highest priority)
        for rule in self.rules:
            if rule.is_triggered(context):
                decision = rule.head.positive

        return decision

    def abduce(self, target_output: bool) -> List[Dict[str, bool]]:
        """
        Backward reasoning: find all contexts that produce the target output.

        For each rule whose head matches target_output, construct the
        minimal context that triggers it (and doesn't trigger any
        higher-priority rule with the opposite conclusion).

        Returns:
            List of context dicts (abductive proofs)
        """
        proofs = []

        for i, rule in enumerate(self.rules):
            if rule.head.positive != target_output:
                continue

            # Build the context that triggers this rule
            context = {}
            for lit in rule.body:
                context[lit.atom] = lit.positive

            # Check that no HIGHER-priority rule (later in list) overrides
            overridden = False
            for j in range(i + 1, len(self.rules)):
                higher_rule = self.rules[j]
                if higher_rule.head.positive != target_output:
                    if higher_rule.is_triggered(context):
                        overridden = True
                        break

            if not overridden:
                proofs.append(context)

        return proofs

    def copy(self) -> 'Policy':
        """Deep copy of this policy."""
        return Policy([r.copy() for r in self.rules])

    def __len__(self):
        return len(self.rules)

    def __repr__(self):
        lines = ["@KnowledgeBase"]
        for rule in self.rules:
            lines.append(f"  {rule};")
        return "\n".join(lines)

    def to_string(self) -> str:
        return repr(self)


# ---------------------------------------------------------------------------
# Policy Generator (for creating random target policies)
# ---------------------------------------------------------------------------

class PolicyGenerator:
    """
    Generates random target policies for experiments.

    Follows the methodology from Markos et al. (2022):
    - Fixed set of atoms A = {a1, ..., an}
    - Random rules with random body subsets and h/-h heads
    - 3-7 rules per policy
    """

    def __init__(self, atoms: List[str] = None, num_atoms: int = 8):
        if atoms:
            self.atoms = atoms
        else:
            self.atoms = [f"a{i+1}" for i in range(num_atoms)]

    def generate(self, num_rules: int = None, seed: int = None) -> Policy:
        """
        Generate a random policy.

        Args:
            num_rules: number of rules (default: random 3-7)
            seed: random seed for reproducibility
        """
        if seed is not None:
            random.seed(seed)

        if num_rules is None:
            num_rules = random.randint(3, 7)

        policy = Policy()

        # Pre-assign heads so at least half are -h, ensuring label balance.
        # Without this, unlucky seeds generate all-h policies → 100% positive
        # dataset → any rule fires → trivial 1-generation "solution".
        n_positive = num_rules // 2
        head_signs = [True] * n_positive + [False] * (num_rules - n_positive)
        random.shuffle(head_signs)

        for i in range(num_rules):
            # Random body length (1 to len(atoms)//2 + 1)
            body_len = random.randint(1, min(len(self.atoms), 4))

            # Random atoms for body
            body_atoms = random.sample(self.atoms, body_len)

            # Random polarity for each
            body = [Literal(a, random.choice([True, False])) for a in body_atoms]

            # Head: use pre-assigned sign to guarantee balance
            head = Literal('h', head_signs[i])

            rule = Rule(body, head, name=f"R{i+1}")
            policy.add_rule(rule)

        if seed is not None:
            random.seed(None)

        return policy

    def generate_set(self, num_policies: int = 30, seed: int = 42) -> List[Policy]:
        """Generate a set of random target policies."""
        policies = []
        for i in range(num_policies):
            p = self.generate(seed=seed + i if seed else None)
            policies.append(p)
        return policies


# ---------------------------------------------------------------------------
# MC-based Symbolic Module (SymbolicModule interface)
# ---------------------------------------------------------------------------

class MCSymbolicModule:
    """
    Symbolic module using Machine Coaching semantics.

    Implements the SymbolicModule interface (deduce, abduce, induce)
    as required by the extended NEUROLOG architecture.
    """

    def __init__(self, policy: Policy = None, atoms: List[str] = None):
        self.policy = policy or Policy()
        self.atoms = atoms or [f"a{i+1}" for i in range(8)]

    def deduce(self, atom_values: Dict[str, bool]) -> Optional[bool]:
        """Forward reasoning via the MC policy."""
        return self.policy.deduce(atom_values)

    def abduce(self, target_output: bool) -> List[Dict[str, bool]]:
        """Backward reasoning: find contexts producing target_output."""
        return self.policy.abduce(target_output)

    def induce(self, new_rule: Rule):
        """Add a new rule to the policy (symbolic learning via mutation)."""
        self.policy.add_rule(new_rule)

    def symbolic_deduction(self, input_state: Dict[str, Any]) -> Dict[str, Any]:
        """SymbolicModule interface compatibility."""
        atom_values = input_state.get('atom_values', {})
        decision = self.deduce(atom_values)

        return {
            'valid': decision is not None,
            'result': decision,
            'derivation': [f"Policy decision: {decision}"],
            'contradictions': [] if decision is not None else ['abstain'],
            'intermediate_states': [],
        }

    def symbolic_abduction(self, desired_output, current_state=None,
                           neural_probs=None) -> List[Dict[str, bool]]:
        """SymbolicModule interface compatibility."""
        target = bool(desired_output) if isinstance(desired_output, (int, float)) else desired_output
        return self.abduce(target)

    def get_policy(self) -> Policy:
        return self.policy

    def copy(self) -> 'MCSymbolicModule':
        return MCSymbolicModule(self.policy.copy(), list(self.atoms))
