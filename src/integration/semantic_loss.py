"""
Semantic Loss via Weighted Model Counting (WMC)

Implements the semantic loss function from:
  - Xu et al. (2018) "A Semantic Loss Function for Deep Learning with Symbolic Knowledge"
  - Tsamoura, Hospedales & Michael (AAAI 2021) compositional neuro-symbolic framework

The key idea: compile logical constraints into an arithmetic circuit (d-DNNF)
and compute P(constraint satisfied | neural outputs) exactly via WMC.

Semantic Loss = -log WMC(alpha, w)
where alpha is the logical theory and w are the neural probability weights.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import product


# ---------------------------------------------------------------------------
# Arithmetic Circuit Node (d-DNNF representation)
# ---------------------------------------------------------------------------

class CircuitNode:
    """Node in a d-DNNF arithmetic circuit for WMC computation."""

    def __init__(self, node_type: str, value: float = None, children: list = None,
                 var_idx: int = None, var_val: int = None):
        """
        Args:
            node_type: 'AND', 'OR', 'LITERAL', 'TRUE', 'FALSE'
            value: cached WMC value after evaluation
            children: child nodes
            var_idx: for LITERAL nodes, which position (0=d1, 1=op, 2=d2)
            var_val: for LITERAL nodes, which class index the literal asserts
        """
        self.node_type = node_type
        self.value = value
        self.children = children or []
        self.var_idx = var_idx
        self.var_val = var_val
        self.grad = 0.0  # gradient accumulator for backprop through circuit


class ArithmeticCircuit:
    """
    Compiles a set of valid assignments (models of a logical theory)
    into a d-DNNF arithmetic circuit and computes WMC + gradients.

    For arithmetic expressions d1 op d2 = target:
      - The theory alpha encodes: "there exists a valid (d1, op, d2) producing target"
      - Each model is one valid assignment
      - WMC(alpha, w) = sum over models of product of neural probs
    """

    def __init__(self, num_positions: int = 3, num_classes: int = 14):
        self.num_positions = num_positions
        self.num_classes = num_classes

    def compile(self, valid_models: List[List[int]]) -> CircuitNode:
        """
        Compile a set of valid models into a d-DNNF circuit.

        Each model is a list of class indices [d1_idx, op_idx, d2_idx].
        The circuit is an OR of ANDs (DNF compiled to d-DNNF):
          OR( AND(lit(0,d1), lit(1,op), lit(2,d2)) for each valid model )

        This is already decomposable (AND children share no variables)
        and deterministic (OR children are mutually exclusive assignments).

        Args:
            valid_models: list of [d1_class, op_class, d2_class] index lists

        Returns:
            Root CircuitNode of the compiled d-DNNF
        """
        if not valid_models:
            return CircuitNode('FALSE', value=0.0)

        # Build OR node over all valid models
        or_children = []
        for model in valid_models:
            # Each model is an AND of literal assignments
            and_children = []
            for pos, cls_idx in enumerate(model):
                lit = CircuitNode('LITERAL', var_idx=pos, var_val=cls_idx)
                and_children.append(lit)
            and_node = CircuitNode('AND', children=and_children)
            or_children.append(and_node)

        if len(or_children) == 1:
            return or_children[0]

        return CircuitNode('OR', children=or_children)

    def evaluate_wmc(self, root: CircuitNode, probs: np.ndarray) -> float:
        """
        Evaluate WMC bottom-up through the circuit.

        Args:
            root: compiled d-DNNF circuit root
            probs: (num_positions, num_classes) neural probability matrix

        Returns:
            WMC value = P(theory satisfied | neural outputs)
        """
        return self._eval_node(root, probs)

    def _eval_node(self, node: CircuitNode, probs: np.ndarray) -> float:
        if node.node_type == 'TRUE':
            node.value = 1.0
        elif node.node_type == 'FALSE':
            node.value = 0.0
        elif node.node_type == 'LITERAL':
            node.value = float(probs[node.var_idx, node.var_val])
        elif node.node_type == 'AND':
            # Product of children (decomposable)
            node.value = 1.0
            for child in node.children:
                self._eval_node(child, probs)
                node.value *= child.value
        elif node.node_type == 'OR':
            # Sum of children (deterministic)
            node.value = 0.0
            for child in node.children:
                self._eval_node(child, probs)
                node.value += child.value
        return node.value

    def compute_gradient(self, root: CircuitNode, probs: np.ndarray) -> np.ndarray:
        """
        Compute d(WMC)/d(prob[pos, cls]) via backpropagation through the circuit.

        This is the exact gradient of the WMC w.r.t. each neural probability,
        enabling proper semantic loss gradient computation.

        Args:
            root: compiled and evaluated d-DNNF circuit
            probs: (num_positions, num_classes)

        Returns:
            grad_probs: (num_positions, num_classes) gradient of WMC w.r.t. probs
        """
        grad_probs = np.zeros_like(probs)

        # Reset gradients
        self._reset_grads(root)

        # Seed: d(loss)/d(WMC) = -1/WMC for semantic loss = -log(WMC)
        # But we return d(WMC)/d(prob) here; the caller applies the -1/WMC chain rule
        root.grad = 1.0

        # Top-down backprop through the circuit
        self._backprop_node(root, probs, grad_probs)

        return grad_probs

    def _reset_grads(self, node: CircuitNode):
        node.grad = 0.0
        for child in node.children:
            self._reset_grads(child)

    def _backprop_node(self, node: CircuitNode, probs: np.ndarray, grad_probs: np.ndarray):
        if node.node_type == 'LITERAL':
            # Accumulate gradient for this probability
            grad_probs[node.var_idx, node.var_val] += node.grad

        elif node.node_type == 'AND':
            # d(prod)/d(child_i) = prod / child_i = prod_of_others
            for i, child in enumerate(node.children):
                if child.value != 0:
                    child.grad += node.grad * (node.value / (child.value + 1e-30))
                else:
                    # If child value is 0, compute product of all other children
                    other_prod = 1.0
                    for j, other in enumerate(node.children):
                        if j != i:
                            other_prod *= other.value
                    child.grad += node.grad * other_prod
                self._backprop_node(child, probs, grad_probs)

        elif node.node_type == 'OR':
            # d(sum)/d(child_i) = 1
            for child in node.children:
                child.grad += node.grad * 1.0
                self._backprop_node(child, probs, grad_probs)


# ---------------------------------------------------------------------------
# Symbol ↔ Index mapping
# ---------------------------------------------------------------------------

SYMBOL_TO_IDX = {str(i): i for i in range(10)}
SYMBOL_TO_IDX.update({'+': 10, '-': 11, '×': 12, '÷': 13})
IDX_TO_SYMBOL = {v: k for k, v in SYMBOL_TO_IDX.items()}

OP_MAP = {'+': 10, '-': 11, '×': 12, '÷': 13}


def symbols_to_indices(symbols: list) -> list:
    """Convert symbol list like ['3', '+', '5'] to class index list [3, 10, 5]."""
    return [SYMBOL_TO_IDX[s] for s in symbols]


# ---------------------------------------------------------------------------
# WMC-based Semantic Loss
# ---------------------------------------------------------------------------

class SemanticLossWMC:
    """
    Computes semantic loss = -log WMC(alpha, w) and its gradient.

    The logical theory alpha is: "the expression evaluates to the target result".
    Valid models are enumerated via the symbolic abduction engine.
    The WMC is computed exactly via d-DNNF circuit evaluation.

    Supports two strategies:
      - 'wmc': Full Weighted Model Counting (sum over ALL valid paths)
      - 'nga': Neural-Guided Abduction (hard max, single best path)
    """

    def __init__(self, num_classes: int = 14):
        self.num_classes = num_classes
        self.circuit = ArithmeticCircuit(num_positions=3, num_classes=num_classes)

    def compute_loss_and_gradient(
        self,
        probs: np.ndarray,
        valid_paths: List[List[str]],
        strategy: str = 'wmc'
    ) -> Tuple[float, np.ndarray]:
        """
        Compute semantic loss and gradient for a single sample.

        Args:
            probs: (seq_len, num_classes) neural output probabilities
            valid_paths: list of symbol lists, e.g. [['3','+','5'], ['8','-','0'], ...]
            strategy: 'wmc' for full WMC, 'nga' for neural-guided (top-1)

        Returns:
            loss: scalar semantic loss value
            gradient: (seq_len, num_classes) gradient w.r.t. probs
        """
        if not valid_paths:
            return 0.0, np.zeros_like(probs)

        # Convert symbol paths to class index paths
        index_paths = [symbols_to_indices(p) for p in valid_paths]

        if strategy == 'nga':
            return self._nga_loss(probs, index_paths)
        else:
            return self._wmc_loss(probs, index_paths)

    def _wmc_loss(self, probs: np.ndarray, index_paths: List[List[int]]) -> Tuple[float, np.ndarray]:
        """
        Full WMC semantic loss via d-DNNF circuit.

        Loss = -log( sum_m prod_t P(m_t | x_t) )

        where m ranges over valid models and t over positions.
        """
        # Step 1: Compile valid models into d-DNNF circuit
        root = self.circuit.compile(index_paths)

        # Step 2: Evaluate WMC bottom-up
        wmc_value = self.circuit.evaluate_wmc(root, probs)

        # Clamp to avoid log(0)
        wmc_value = max(wmc_value, 1e-30)

        # Step 3: Semantic loss = -log(WMC)
        loss = -np.log(wmc_value)

        # Step 4: Gradient via circuit backprop
        # d(-log WMC)/d(prob) = -1/WMC * d(WMC)/d(prob)
        dwmc_dprob = self.circuit.compute_gradient(root, probs)
        gradient = -dwmc_dprob / wmc_value

        return float(loss), gradient

    def _nga_loss(self, probs: np.ndarray, index_paths: List[List[int]]) -> Tuple[float, np.ndarray]:
        """
        Neural-Guided Abduction: pick single most probable path.

        Loss = -log( prod_t P(m*_t | x_t) )
        where m* = argmax_m prod_t P(m_t | x_t)
        """
        # Score each path by log probability
        best_score = -np.inf
        best_path = None

        for path in index_paths:
            score = 0.0
            for t, idx in enumerate(path):
                score += np.log(probs[t, idx] + 1e-30)
            if score > best_score:
                best_score = score
                best_path = path

        loss = -best_score

        # Gradient: P - T where T is one-hot for best path
        gradient = probs.copy()
        target = np.zeros_like(probs)
        for t, idx in enumerate(best_path):
            target[t, idx] = 1.0
        gradient = probs - target

        return float(loss), gradient

    def compute_batch_loss(
        self,
        batch_probs: np.ndarray,
        batch_valid_paths: List[List[List[str]]],
        strategy: str = 'wmc'
    ) -> Tuple[float, np.ndarray]:
        """
        Compute semantic loss over a batch.

        Args:
            batch_probs: (batch_size, seq_len, num_classes)
            batch_valid_paths: list of valid_paths per sample
            strategy: 'wmc' or 'nga'

        Returns:
            avg_loss: average semantic loss over batch
            gradients: (batch_size, seq_len, num_classes)
        """
        batch_size = len(batch_probs)
        gradients = np.zeros_like(batch_probs)
        total_loss = 0.0
        active_count = 0

        for i in range(batch_size):
            paths = batch_valid_paths[i]
            if paths:
                loss, grad = self.compute_loss_and_gradient(
                    batch_probs[i], paths, strategy
                )
                gradients[i] = grad
                total_loss += loss
                active_count += 1

        avg_loss = total_loss / max(active_count, 1)
        return avg_loss, gradients


# ---------------------------------------------------------------------------
# Legacy-compatible wrapper (drop-in replacement for old compute_semantic_loss)
# ---------------------------------------------------------------------------

_global_wmc = SemanticLossWMC()


def compute_semantic_loss(
    predictions: Dict[str, np.ndarray],
    abductions: List[Dict],
    original_targets: np.ndarray,
    strategy: str = 'wmc'
) -> Tuple[float, Optional[Dict]]:
    """
    Legacy-compatible interface.

    Now delegates to proper WMC computation instead of top-1 heuristic.

    Args:
        predictions: dict with 'probabilities' key → (seq_len, num_classes)
        abductions: list of abduction dicts with 'correction' key
        original_targets: ground truth class indices
        strategy: 'wmc' or 'nga'

    Returns:
        loss: semantic loss value
        info: dict with gradient and diagnostic info
    """
    if not abductions:
        return 0.0, None

    probs = predictions.get('probabilities')
    if probs is None:
        return 0.0, None

    # Extract valid paths from abduction results
    valid_paths = [abd['correction'] for abd in abductions if 'correction' in abd]

    if not valid_paths:
        return 0.0, None

    loss, gradient = _global_wmc.compute_loss_and_gradient(probs, valid_paths, strategy)

    info = {
        'gradient': gradient,
        'wmc_loss': loss,
        'num_valid_paths': len(valid_paths),
        'strategy': strategy,
        'best_correction': abductions[0] if abductions else None
    }

    return loss, info
