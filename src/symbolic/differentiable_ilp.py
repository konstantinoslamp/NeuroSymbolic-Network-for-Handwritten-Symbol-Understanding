"""
Differentiable Inductive Logic Programming (dILP) — P3.1

Instead of hardcoding arithmetic rules, this module learns them from data
using a differentiable ILP approach inspired by:
  - Evans & Grefenstette (2018) "Learning Explanatory Rules through Neural Logic"
  - Shindo et al. (2021) "αILP: Differentiable ILP for neural-symbolic integration"

Core idea:
  - Define a *rule template* space (possible Horn clause structures)
  - Assign a learnable weight (confidence) to each candidate rule
  - Forward reasoning applies all rules with soft (probabilistic) semantics
  - Backward pass updates rule weights via gradient descent
  - After training, high-weight rules are the learned symbolic knowledge

For arithmetic: the system discovers rules like
  result(D1, plus, D2, R) :- R = D1 + D2
from examples of (input, output) pairs, without being told what '+' means.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from itertools import product


# ---------------------------------------------------------------------------
# Rule Template Language
# ---------------------------------------------------------------------------

class RuleTemplate:
    """
    A candidate Horn clause with a learnable weight.

    Structure: head :- body_1, body_2, ..., body_k.

    The weight θ ∈ ℝ is passed through sigmoid to get P(rule is active).
    """

    def __init__(self, name: str, head: str, body: List[str],
                 evaluate_fn=None, description: str = ""):
        """
        Args:
            name: unique rule identifier
            head: head predicate string, e.g. "result(D1, Op, D2, R)"
            body: list of body literals
            evaluate_fn: callable(d1, op_idx, d2) -> Optional[float]
                         returns the result if this rule fires, else None
            description: human-readable description
        """
        self.name = name
        self.head = head
        self.body = body
        self.evaluate_fn = evaluate_fn
        self.description = description

        # Learnable weight (logit space)
        self.weight = 0.0  # Initialized to 0 (sigmoid = 0.5)
        self.grad_weight = 0.0

    @property
    def confidence(self) -> float:
        """Probability that this rule is active."""
        return 1.0 / (1.0 + np.exp(-np.clip(self.weight, -20, 20)))

    def __repr__(self):
        return f"Rule({self.name}, conf={self.confidence:.3f}): {self.head} :- {', '.join(self.body)}"


# ---------------------------------------------------------------------------
# Arithmetic Rule Templates
# ---------------------------------------------------------------------------

def _make_arithmetic_templates() -> List[RuleTemplate]:
    """
    Generate candidate arithmetic rule templates.

    These cover all reasonable binary arithmetic operations that the system
    might need to discover. The dILP learner will assign high weights to
    the correct ones (addition, subtraction, multiplication, division)
    and low weights to incorrect ones.
    """
    templates = []

    # === Correct rules (should be learned) ===

    # Addition
    templates.append(RuleTemplate(
        name="add",
        head="result(D1, plus, D2, R)",
        body=["digit(D1)", "digit(D2)", "R is D1 + D2"],
        evaluate_fn=lambda d1, op, d2: float(d1 + d2) if op == 0 else None,
        description="R = D1 + D2",
    ))

    # Subtraction
    templates.append(RuleTemplate(
        name="sub",
        head="result(D1, minus, D2, R)",
        body=["digit(D1)", "digit(D2)", "R is D1 - D2"],
        evaluate_fn=lambda d1, op, d2: float(d1 - d2) if op == 1 else None,
        description="R = D1 - D2",
    ))

    # Multiplication
    templates.append(RuleTemplate(
        name="mul",
        head="result(D1, times, D2, R)",
        body=["digit(D1)", "digit(D2)", "R is D1 * D2"],
        evaluate_fn=lambda d1, op, d2: float(d1 * d2) if op == 2 else None,
        description="R = D1 × D2",
    ))

    # Division
    templates.append(RuleTemplate(
        name="div",
        head="result(D1, divide, D2, R)",
        body=["digit(D1)", "digit(D2)", "D2 != 0", "R is D1 / D2"],
        evaluate_fn=lambda d1, op, d2: float(d1 / d2) if op == 3 and d2 != 0 else None,
        description="R = D1 ÷ D2 (D2 ≠ 0)",
    ))

    # === Distractor rules (should NOT be learned) ===

    # Wrong: addition for multiplication operator
    templates.append(RuleTemplate(
        name="wrong_mul_as_add",
        head="result(D1, times, D2, R)",
        body=["digit(D1)", "digit(D2)", "R is D1 + D2"],
        evaluate_fn=lambda d1, op, d2: float(d1 + d2) if op == 2 else None,
        description="WRONG: R = D1 + D2 when op = ×",
    ))

    # Wrong: subtraction for addition operator
    templates.append(RuleTemplate(
        name="wrong_add_as_sub",
        head="result(D1, plus, D2, R)",
        body=["digit(D1)", "digit(D2)", "R is D1 - D2"],
        evaluate_fn=lambda d1, op, d2: float(d1 - d2) if op == 0 else None,
        description="WRONG: R = D1 - D2 when op = +",
    ))

    # Wrong: always return D1 (ignoring D2)
    templates.append(RuleTemplate(
        name="wrong_identity_d1",
        head="result(D1, Op, D2, R)",
        body=["digit(D1)", "R is D1"],
        evaluate_fn=lambda d1, op, d2: float(d1),
        description="WRONG: R = D1 (ignore operation)",
    ))

    # Wrong: always return 0
    templates.append(RuleTemplate(
        name="wrong_zero",
        head="result(D1, Op, D2, R)",
        body=["R is 0"],
        evaluate_fn=lambda d1, op, d2: 0.0,
        description="WRONG: R = 0 always",
    ))

    # Modular arithmetic (plausible but wrong for standard arithmetic)
    templates.append(RuleTemplate(
        name="wrong_mod",
        head="result(D1, plus, D2, R)",
        body=["digit(D1)", "digit(D2)", "R is (D1 + D2) mod 10"],
        evaluate_fn=lambda d1, op, d2: float((d1 + d2) % 10) if op == 0 else None,
        description="WRONG: R = (D1 + D2) mod 10",
    ))

    # Max operation
    templates.append(RuleTemplate(
        name="wrong_max",
        head="result(D1, Op, D2, R)",
        body=["digit(D1)", "digit(D2)", "R is max(D1, D2)"],
        evaluate_fn=lambda d1, op, d2: float(max(d1, d2)),
        description="WRONG: R = max(D1, D2)",
    ))

    return templates


# Operator index mapping: 0=+, 1=-, 2=×, 3=÷
OP_IDX = {'+': 0, '-': 1, '×': 2, '÷': 3}
OP_SYM = {0: '+', 1: '-', 2: '×', 3: '÷'}


# ---------------------------------------------------------------------------
# Differentiable Forward Chaining
# ---------------------------------------------------------------------------

class DifferentiableForwardChainer:
    """
    Soft forward reasoning over weighted rule templates.

    Given an expression (d1, op, d2), computes a weighted result by
    mixing the outputs of all applicable rules weighted by their confidence:

      result = Σ_r  conf(r) × eval_r(d1, op, d2)  /  Σ_r conf(r)

    where conf(r) = σ(θ_r) is the sigmoid of the learnable rule weight.

    This is differentiable w.r.t. rule weights, enabling gradient-based learning.
    """

    def __init__(self, templates: List[RuleTemplate] = None):
        self.templates = templates or _make_arithmetic_templates()

    def forward(self, d1: int, op_idx: int, d2: int) -> Tuple[Optional[float], Dict]:
        """
        Soft forward reasoning.

        Args:
            d1: first digit (0-9)
            op_idx: operator index (0=+, 1=-, 2=×, 3=÷)
            d2: second digit (0-9)

        Returns:
            (weighted_result, info_dict)
        """
        total_weight = 0.0
        weighted_sum = 0.0
        rule_contributions = []

        for rule in self.templates:
            conf = rule.confidence
            result = rule.evaluate_fn(d1, op_idx, d2)

            if result is not None:
                total_weight += conf
                weighted_sum += conf * result
                rule_contributions.append({
                    'rule': rule.name,
                    'confidence': conf,
                    'result': result,
                    'contribution': conf * result,
                })

        if total_weight < 1e-10:
            return None, {'error': 'no_applicable_rules'}

        soft_result = weighted_sum / total_weight

        return soft_result, {
            'soft_result': soft_result,
            'total_weight': total_weight,
            'contributions': rule_contributions,
        }

    def compute_loss(self, d1: int, op_idx: int, d2: int,
                     target: float) -> Tuple[float, Dict]:
        """
        Compute loss for a single training example.

        Loss = (soft_result - target)² weighted by rule confidence.
        """
        soft_result, info = self.forward(d1, op_idx, d2)

        if soft_result is None:
            return 0.0, info

        loss = (soft_result - target) ** 2
        info['loss'] = loss
        info['target'] = target
        return loss, info

    def backward(self, d1: int, op_idx: int, d2: int, target: float):
        """
        Compute gradients for rule weights.

        d(Loss)/d(θ_r) = d(Loss)/d(result) × d(result)/d(conf_r) × d(conf_r)/d(θ_r)
        """
        soft_result, info = self.forward(d1, op_idx, d2)
        if soft_result is None:
            return

        total_weight = info['total_weight']
        dloss_dresult = 2.0 * (soft_result - target)

        for rule in self.templates:
            conf = rule.confidence
            result = rule.evaluate_fn(d1, op_idx, d2)

            if result is None:
                continue

            # d(weighted_avg)/d(conf_r)
            # = (result × total_weight - weighted_sum) / total_weight²
            # = (result - soft_result) / total_weight
            dresult_dconf = (result - soft_result) / (total_weight + 1e-10)

            # d(conf)/d(θ) = conf × (1 - conf)  (sigmoid derivative)
            dconf_dtheta = conf * (1 - conf)

            rule.grad_weight += dloss_dresult * dresult_dconf * dconf_dtheta


# ---------------------------------------------------------------------------
# dILP Rule Learner
# ---------------------------------------------------------------------------

class DifferentiableILP:
    """
    Differentiable Inductive Logic Programming system.

    Learns which symbolic rules are correct from (input, output) examples.

    Training loop:
      1. Present (d1, op, d2, result) examples
      2. Forward: soft evaluation using weighted rules
      3. Backward: update rule weights via gradient descent
      4. After convergence: extract high-confidence rules as learned KB

    This closes the neuro-symbolic loop completely:
      - Neural module learns perception (images → symbols)
      - Symbolic module learns rules (symbols → results)
      - Both trained jointly via semantic loss
    """

    def __init__(self, learning_rate: float = 0.1):
        self.forward_chainer = DifferentiableForwardChainer()
        self.learning_rate = learning_rate
        self.training_history = []

    @property
    def templates(self):
        return self.forward_chainer.templates

    def train_step(self, examples: List[Tuple[int, int, int, float]]) -> Dict:
        """
        One training step over a batch of examples.

        Args:
            examples: list of (d1, op_idx, d2, target_result) tuples

        Returns:
            Metrics dict with loss and rule confidences.
        """
        # Reset gradients
        for rule in self.templates:
            rule.grad_weight = 0.0

        total_loss = 0.0
        for d1, op_idx, d2, target in examples:
            loss, _ = self.forward_chainer.compute_loss(d1, op_idx, d2, target)
            self.forward_chainer.backward(d1, op_idx, d2, target)
            total_loss += loss

        avg_loss = total_loss / max(len(examples), 1)

        # Update rule weights
        for rule in self.templates:
            rule.weight -= self.learning_rate * rule.grad_weight / max(len(examples), 1)

        # Record history
        confidences = {r.name: r.confidence for r in self.templates}
        self.training_history.append({
            'loss': avg_loss,
            'confidences': confidences,
        })

        return {'loss': avg_loss, 'confidences': confidences}

    def train(self, examples: List[Tuple[int, int, int, float]],
              epochs: int = 100, batch_size: int = 32,
              verbose: bool = True) -> List[Dict]:
        """
        Full training loop.

        Args:
            examples: list of (d1, op_idx, d2, target_result)
            epochs: number of training epochs
            batch_size: mini-batch size
            verbose: print progress

        Returns:
            Training history.
        """
        if verbose:
            print(f"dILP Training: {len(examples)} examples, {epochs} epochs")
            print(f"Rule templates: {len(self.templates)}")

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(len(examples))

            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, len(examples), batch_size):
                batch_idx = indices[start:start + batch_size]
                batch = [examples[i] for i in batch_idx]
                metrics = self.train_step(batch)
                epoch_loss += metrics['loss']
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            if verbose and (epoch + 1) % max(1, epochs // 10) == 0:
                print(f"  Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
                self._print_top_rules(top_k=4)

        return self.training_history

    def extract_learned_rules(self, threshold: float = 0.7) -> List[RuleTemplate]:
        """
        Extract rules with confidence above threshold.

        These are the rules the system has learned from data.
        """
        return [r for r in self.templates if r.confidence >= threshold]

    def extract_asp_program(self, threshold: float = 0.7) -> str:
        """
        Convert learned rules to an ASP program for injection into clingo KB.
        """
        learned = self.extract_learned_rules(threshold)
        lines = [f"% Learned rules (dILP, threshold={threshold})"]
        for rule in learned:
            lines.append(f"% {rule.name} (conf={rule.confidence:.3f}): {rule.description}")
            lines.append(f"{rule.head} :- {', '.join(rule.body)}.")
        return "\n".join(lines)

    def generate_training_data(self, num_samples: int = 500) -> List[Tuple]:
        """
        Generate ground-truth training examples for rule learning.

        Returns (d1, op_idx, d2, result) tuples covering all 4 operators.
        """
        examples = []
        ops = {
            0: lambda a, b: float(a + b),        # +
            1: lambda a, b: float(a - b),        # -
            2: lambda a, b: float(a * b),        # ×
            3: lambda a, b: float(a / b) if b != 0 else None,  # ÷
        }

        for _ in range(num_samples):
            d1 = np.random.randint(0, 10)
            op_idx = np.random.randint(0, 4)
            d2 = np.random.randint(0, 10)

            if op_idx == 3 and d2 == 0:
                d2 = np.random.randint(1, 10)

            result = ops[op_idx](d1, d2)
            if result is not None:
                examples.append((d1, op_idx, d2, result))

        return examples

    def evaluate(self, examples: List[Tuple]) -> Dict:
        """Evaluate learned rules on test examples."""
        correct = 0
        total = 0
        per_op = {i: {'correct': 0, 'total': 0} for i in range(4)}

        for d1, op_idx, d2, target in examples:
            result, _ = self.forward_chainer.forward(d1, op_idx, d2)
            if result is not None and abs(result - target) < 0.1:
                correct += 1
                per_op[op_idx]['correct'] += 1
            per_op[op_idx]['total'] += 1
            total += 1

        return {
            'accuracy': correct / max(total, 1),
            'total': total,
            'correct': correct,
            'per_operator': {
                OP_SYM[k]: v['correct'] / max(v['total'], 1)
                for k, v in per_op.items()
            },
        }

    def _print_top_rules(self, top_k=5):
        """Print rules sorted by confidence."""
        sorted_rules = sorted(self.templates, key=lambda r: r.confidence, reverse=True)
        for r in sorted_rules[:top_k]:
            marker = "OK" if r.confidence > 0.7 else "  "
            print(f"    [{marker}] {r.confidence:.3f}  {r.name}: {r.description}")

    def summary(self) -> Dict:
        """Return structured summary of learned rules."""
        return {
            'all_rules': [
                {
                    'name': r.name,
                    'confidence': r.confidence,
                    'weight': r.weight,
                    'description': r.description,
                    'head': r.head,
                    'body': r.body,
                }
                for r in sorted(self.templates, key=lambda r: r.confidence, reverse=True)
            ],
            'learned_rules': [
                r.name for r in self.extract_learned_rules()
            ],
            'training_epochs': len(self.training_history),
            'final_loss': self.training_history[-1]['loss'] if self.training_history else None,
        }


# ---------------------------------------------------------------------------
# Integration with existing KB
# ---------------------------------------------------------------------------

class LearnableKnowledgeBase:
    """
    Knowledge base that combines hardcoded domain facts with learned rules.

    The dILP learner discovers which arithmetic rules are correct,
    and this KB integrates them with the existing clingo-based reasoning.
    """

    def __init__(self):
        self.ilp = DifferentiableILP()
        self._trained = False

        # Import the original KB for fallback
        from src.symbolic.knowledge_base import KnowledgeBase
        self._fallback_kb = KnowledgeBase()

    def learn_rules(self, examples: List[Tuple] = None, epochs: int = 200,
                    verbose: bool = True):
        """
        Learn arithmetic rules from examples.

        If no examples provided, generates synthetic training data.
        """
        if examples is None:
            examples = self.ilp.generate_training_data(num_samples=1000)

        self.ilp.train(examples, epochs=epochs, verbose=verbose)
        self._trained = True

        if verbose:
            learned = self.ilp.extract_learned_rules()
            print(f"\nLearned {len(learned)} rules:")
            for r in learned:
                print(f"  {r.name} (conf={r.confidence:.3f}): {r.description}")

    def deduce(self, d1: int, op: str, d2: int) -> Optional[float]:
        """
        Deduce result using learned rules (with fallback to hardcoded KB).
        """
        if self._trained:
            op_idx = OP_IDX.get(op)
            if op_idx is not None:
                result, info = self.ilp.forward_chainer.forward(d1, op_idx, d2)
                if result is not None:
                    return result

        # Fallback to original KB
        return self._fallback_kb.deduce(d1, op, d2)

    def abduce(self, target: float) -> List[Tuple[int, str, int]]:
        """Abduce using original KB (rule learning doesn't change abduction structure)."""
        return self._fallback_kb.abduce(target)

    def get_learned_asp(self) -> str:
        """Get learned rules as ASP program."""
        if self._trained:
            return self.ilp.extract_asp_program()
        return "% No rules learned yet"
