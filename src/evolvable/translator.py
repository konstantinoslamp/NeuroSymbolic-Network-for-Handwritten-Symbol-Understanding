"""
Translator — Bridges Neural and Symbolic Modules

Implements the Translator from the NEUROLOG architecture (Tsamoura et al. 2021),
formalized for the Evolvable Policies framework (Thoma et al. 2026).

The Translator converts between:
  - Neural space: CNN output activations (probabilities per neuron)
  - Symbolic space: atom truth values for the MC policy

Convention (from the paper):
  - Each atom a_i corresponds to one output neuron of the CNN
  - The CNN processes MNIST images: digit 1 = positive atom, digit 2 = negative atom
  - Neural output > threshold → atom is positive; otherwise negative
  - For training: abductive proofs (atom assignments) → soft targets for the CNN

The Translator also handles the WMC computation by converting abductive proofs
into the format needed for semantic loss calculation.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any


class Translator:
    """
    Bidirectional translator between neural activations and symbolic atoms.

    neural_to_symbolic: CNN outputs → atom values for policy deduction
    symbolic_to_neural: atom assignments (from abduction) → training targets
    """

    def __init__(self, atoms: List[str], threshold: float = 0.5):
        """
        Args:
            atoms: list of atom names, e.g. ['a1', ..., 'a8']
            threshold: decision threshold for neural → symbolic conversion
        """
        self.atoms = atoms
        self.num_atoms = len(atoms)
        self.threshold = threshold

        # Each atom has 2 output neurons: P(positive), P(negative)
        # Neuron index mapping: atom i → neurons [2*i, 2*i+1]
        # Neuron 2*i   = P(atom_i is positive)  → MNIST digit 1
        # Neuron 2*i+1 = P(atom_i is negative)  → MNIST digit 2
        self.num_neurons = 2 * self.num_atoms

    def neural_to_symbolic(self, neural_output: np.ndarray) -> Dict[str, bool]:
        """
        Convert neural activations to symbolic atom values.

        Args:
            neural_output: (num_neurons,) or (num_atoms, 2) probabilities
                          For each atom: [P(positive), P(negative)]

        Returns:
            Dict mapping atom names to True/False
        """
        if neural_output.ndim == 1:
            # Reshape from flat to (num_atoms, 2)
            output = neural_output[:self.num_neurons].reshape(self.num_atoms, 2)
        else:
            output = neural_output

        atom_values = {}
        for i, atom in enumerate(self.atoms):
            p_positive = output[i, 0]
            p_negative = output[i, 1]

            # Atom is positive if P(positive) > P(negative)
            atom_values[atom] = bool(p_positive > p_negative)

        return atom_values

    def neural_to_symbolic_batch(self, neural_outputs: np.ndarray) -> List[Dict[str, bool]]:
        """
        Batch conversion.

        Args:
            neural_outputs: (batch_size, num_neurons) or (batch_size, num_atoms, 2)
        """
        batch_size = neural_outputs.shape[0]
        return [self.neural_to_symbolic(neural_outputs[i]) for i in range(batch_size)]

    def symbolic_to_neural(self, atom_values: Dict[str, bool]) -> np.ndarray:
        """
        Convert symbolic atom values to neural target vector.

        Args:
            atom_values: dict mapping atom names to True/False

        Returns:
            (num_neurons,) target vector. For each atom:
                positive → [1, 0] (neuron for digit 1 active)
                negative → [0, 1] (neuron for digit 2 active)
        """
        targets = np.zeros(self.num_neurons)

        for i, atom in enumerate(self.atoms):
            if atom in atom_values:
                if atom_values[atom]:
                    targets[2 * i] = 1.0      # positive → digit 1
                    targets[2 * i + 1] = 0.0
                else:
                    targets[2 * i] = 0.0
                    targets[2 * i + 1] = 1.0  # negative → digit 2

        return targets

    def get_atom_probabilities(self, neural_output: np.ndarray) -> Dict[str, float]:
        """
        Get per-atom probability of being positive.

        Args:
            neural_output: (num_neurons,) or (num_atoms, 2)

        Returns:
            Dict mapping atom name to P(positive)
        """
        if neural_output.ndim == 1:
            output = neural_output[:self.num_neurons].reshape(self.num_atoms, 2)
        else:
            output = neural_output

        probs = {}
        for i, atom in enumerate(self.atoms):
            total = output[i, 0] + output[i, 1]
            if total > 0:
                probs[atom] = float(output[i, 0] / total)
            else:
                probs[atom] = 0.5
        return probs

    def compute_wmc(self, neural_output: np.ndarray,
                    abductive_proofs: List[Dict[str, bool]]) -> float:
        """
        Compute Weighted Model Count for semantic loss.

        WMC = Σ_proof Π_atom w(atom, proof[atom])

        where w(atom, True)  = P(atom positive | neural_output)
              w(atom, False) = P(atom negative | neural_output)

        Args:
            neural_output: (num_neurons,) or (num_atoms, 2)
            abductive_proofs: list of atom→bool dicts from policy.abduce()

        Returns:
            WMC value (probability that the symbolic constraint is satisfied)
        """
        if not abductive_proofs:
            return 0.0

        if neural_output.ndim == 1:
            output = neural_output[:self.num_neurons].reshape(self.num_atoms, 2)
        else:
            output = neural_output

        # Normalize to probabilities per atom
        atom_probs = np.zeros((self.num_atoms, 2))
        for i in range(self.num_atoms):
            total = output[i, 0] + output[i, 1] + 1e-10
            atom_probs[i, 0] = output[i, 0] / total  # P(positive)
            atom_probs[i, 1] = output[i, 1] / total  # P(negative)

        wmc = 0.0
        for proof in abductive_proofs:
            proof_prob = 1.0
            for i, atom in enumerate(self.atoms):
                if atom in proof:
                    if proof[atom]:
                        proof_prob *= atom_probs[i, 0]
                    else:
                        proof_prob *= atom_probs[i, 1]
                # If atom not in proof, it's unconstrained → contributes factor 1
            wmc += proof_prob

        return wmc

    def compute_semantic_loss(self, neural_output: np.ndarray,
                              abductive_proofs: List[Dict[str, bool]]) -> Tuple[float, np.ndarray]:
        """
        Compute semantic loss = -log(WMC) and its gradient.

        Args:
            neural_output: (num_neurons,) or (num_atoms, 2) raw activations
            abductive_proofs: from policy.abduce()

        Returns:
            (loss, gradient) where gradient has same shape as neural_output
        """
        if not abductive_proofs:
            return 0.0, np.zeros_like(neural_output)

        flat = neural_output.ndim == 1
        if flat:
            output = neural_output[:self.num_neurons].reshape(self.num_atoms, 2)
        else:
            output = neural_output

        # Softmax per atom pair to get probabilities
        atom_probs = np.zeros((self.num_atoms, 2))
        for i in range(self.num_atoms):
            exp_vals = np.exp(output[i] - np.max(output[i]))
            atom_probs[i] = exp_vals / (np.sum(exp_vals) + 1e-10)

        # Forward: compute WMC
        wmc = 0.0
        proof_probs = []
        for proof in abductive_proofs:
            proof_prob = 1.0
            for i, atom in enumerate(self.atoms):
                if atom in proof:
                    idx = 0 if proof[atom] else 1
                    proof_prob *= atom_probs[i, idx]
            proof_probs.append(proof_prob)
            wmc += proof_prob

        wmc = max(wmc, 1e-30)
        loss = -np.log(wmc)

        # Backward: gradient of -log(WMC) w.r.t. atom_probs
        grad_probs = np.zeros((self.num_atoms, 2))
        for proof_idx, proof in enumerate(abductive_proofs):
            for i, atom in enumerate(self.atoms):
                if atom in proof:
                    idx = 0 if proof[atom] else 1
                    # d(loss)/d(prob) = -1/WMC * d(WMC)/d(prob)
                    # d(WMC)/d(prob_i) = proof_prob / prob_i
                    if atom_probs[i, idx] > 1e-10:
                        grad_probs[i, idx] += -proof_probs[proof_idx] / (wmc * atom_probs[i, idx])

        # Chain rule through softmax: d(loss)/d(logit) = prob - target_prob
        grad_logits = np.zeros((self.num_atoms, 2))
        for i in range(self.num_atoms):
            # grad_logits = atom_probs * (grad_probs · 1) - grad_probs
            # Simplified softmax backward
            s = np.sum(grad_probs[i] * atom_probs[i])
            grad_logits[i] = atom_probs[i] * s - grad_probs[i] * atom_probs[i]
            # Actually, for softmax: d(loss)/d(z_j) = Σ_k d(loss)/d(p_k) * p_k * (δ_jk - p_j)
            for j in range(2):
                grad_logits[i, j] = 0
                for k in range(2):
                    if j == k:
                        grad_logits[i, j] += grad_probs[i, k] * atom_probs[i, k] * (1 - atom_probs[i, j])
                    else:
                        grad_logits[i, j] += grad_probs[i, k] * atom_probs[i, k] * (-atom_probs[i, j])

        if flat:
            return loss, grad_logits.flatten()
        return loss, grad_logits
