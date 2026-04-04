"""
NeSy Organism — Extended NEUROLOG with induce()

A NeSy organism is a single instance of the extended NEUROLOG architecture,
consisting of:
  - NeuralModule: CNN processing MNIST image sequences
  - SymbolicModule: MC policy with prioritized rules
  - Translator: bridges neural ↔ symbolic representations

Each organism exposes three methods on both modules:
  - deduce(): forward reasoning
  - abduce(): backward reasoning
  - induce(): learning (backprop for neural, rule addition for symbolic)

Organisms are the unit of evolution in the Evolvable Policies framework.
"""

import numpy as np
import copy
from typing import Dict, List, Tuple, Optional, Any

from src.evolvable.machine_coaching import Policy, Rule, Literal, MCSymbolicModule
from src.evolvable.translator import Translator
from src.neural.model import CNN


class NeSyOrganism:
    """
    A complete NeSy system: CNN + MC Policy + Translator.

    This is an "organism" in the evolutionary framework.
    It can be trained, evaluated, and mutated.
    """

    def __init__(self, atoms: List[str] = None, policy: Policy = None,
                 cnn: CNN = None):
        self.atoms = atoms or [f"a{i+1}" for i in range(8)]
        self.num_atoms = len(self.atoms)

        # Symbolic module (Machine Coaching)
        self.symbolic = MCSymbolicModule(
            policy=policy or Policy(),
            atoms=self.atoms,
        )

        # Neural module (CNN)
        # The CNN outputs num_atoms * 2 neurons (per-atom binary classification)
        # We reuse existing CNN but the output layer maps to 2*num_atoms
        if cnn is not None:
            self.cnn = cnn
        else:
            self.cnn = CNN()
            # Reinitialize fc2 for correct output size (2 * num_atoms = 16)
            from src.neural.cnn import Dense
            self.cnn.fc2 = Dense(in_features=128, out_features=2 * self.num_atoms)
            self.cnn.trainable_layers = [self.cnn.conv1, self.cnn.fc1, self.cnn.fc2]

        # Translator
        self.translator = Translator(self.atoms)

    # ------------------------------------------------------------------
    # Forward Reasoning (Deduction)
    # ------------------------------------------------------------------

    def deduce(self, images: np.ndarray) -> Dict[str, Any]:
        """
        Full forward pass: images → CNN → Translator → Policy → decision.

        Args:
            images: (num_atoms, 1, 28, 28) sequence of MNIST images,
                    one per atom

        Returns:
            {
                'decision': True/False/None (h/-h/abstain),
                'atom_values': {atom: bool},
                'neural_output': raw CNN output,
                'probabilities': per-atom probabilities,
            }
        """
        # Step 1: Neural deduction — process each image through CNN
        logits = self.cnn.forward(images)  # (num_atoms, 2*num_atoms)

        # Each image i should primarily activate neurons [2*i, 2*i+1]
        # But the CNN sees all images → aggregate
        # Approach: process images sequentially, concatenate relevant outputs
        neural_output = np.zeros(2 * self.num_atoms)
        for i in range(min(len(images), self.num_atoms)):
            img = images[i:i+1]  # (1, 1, 28, 28)
            out = self.cnn.forward(img)  # (1, 2*num_atoms)
            # Extract this atom's neurons
            neural_output[2*i] = out[0, 2*i]
            neural_output[2*i + 1] = out[0, 2*i + 1]

        # Apply softmax per atom pair
        probs = np.zeros(2 * self.num_atoms)
        for i in range(self.num_atoms):
            pair = neural_output[2*i:2*i+2]
            exp_pair = np.exp(pair - np.max(pair))
            probs[2*i:2*i+2] = exp_pair / (np.sum(exp_pair) + 1e-10)

        # Step 2: Translate to symbolic atoms
        atom_values = self.translator.neural_to_symbolic(probs)

        # Step 3: Symbolic deduction
        decision = self.symbolic.deduce(atom_values)

        return {
            'decision': decision,
            'atom_values': atom_values,
            'neural_output': neural_output,
            'probabilities': probs,
        }

    # ------------------------------------------------------------------
    # Training (Neural Induction via Abductive Feedback)
    # ------------------------------------------------------------------

    def train_step(self, images: np.ndarray, label: bool,
                   learning_rate: float = 0.001) -> Dict[str, float]:
        """
        Train the neural module using abductive feedback from the symbolic module.

        Steps:
          1. Get abductive proofs for the label from the MC policy
          2. If proofs exist, compute WMC-based semantic loss
          3. Backpropagate through CNN

        Args:
            images: (num_atoms, 1, 28, 28) MNIST image sequence
            label: True (h) or False (-h)
            learning_rate: SGD step size

        Returns:
            {'semantic_loss': float, 'has_proofs': bool}
        """
        # Get abductive proofs
        proofs = self.symbolic.abduce(label)

        if not proofs:
            return {'semantic_loss': 0.0, 'has_proofs': False}

        # Forward pass through CNN for each image
        all_logits = []
        for i in range(min(len(images), self.num_atoms)):
            img = images[i:i+1]
            logits = self.cnn.forward(img)
            all_logits.append(logits[0])

        # Build the full neural output vector
        neural_output = np.zeros(2 * self.num_atoms)
        for i, logits in enumerate(all_logits):
            neural_output[2*i] = logits[2*i]
            neural_output[2*i + 1] = logits[2*i + 1]

        # Compute semantic loss and gradient
        loss, grad = self.translator.compute_semantic_loss(neural_output, proofs)

        # Backpropagate gradient to each image's CNN pass
        for i in range(min(len(images), self.num_atoms)):
            img = images[i:i+1]
            # Forward pass (needed for cached values)
            self.cnn.forward(img)

            # This image's gradient contribution
            img_grad = np.zeros((1, 2 * self.num_atoms))
            img_grad[0, 2*i] = grad[2*i]
            img_grad[0, 2*i + 1] = grad[2*i + 1]

            self.cnn.backward(grad=img_grad)
            self.cnn.update_weights(learning_rate)

        return {'semantic_loss': float(loss), 'has_proofs': True}

    def train_epoch(self, dataset, learning_rate: float = 0.001,
                    batch_size: int = 32) -> Dict[str, float]:
        """
        Train for one full epoch over the dataset.

        Args:
            dataset: list of {'images': ndarray, 'label': bool} dicts
            learning_rate: SGD step size
            batch_size: not used for per-sample training, kept for API compat

        Returns:
            {'avg_loss': float, 'proof_rate': float}
        """
        total_loss = 0.0
        proof_count = 0
        n = len(dataset)

        indices = np.random.permutation(n)

        for idx in indices:
            sample = dataset[idx]
            images = sample['images']
            label = sample['label']

            if images.ndim == 3:
                images = images[:, np.newaxis, :, :]

            result = self.train_step(images, label, learning_rate)
            total_loss += result['semantic_loss']
            if result['has_proofs']:
                proof_count += 1

        return {
            'avg_loss': total_loss / max(n, 1),
            'proof_rate': proof_count / max(n, 1),
        }

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, dataset) -> Dict[str, float]:
        """
        Evaluate the organism on a dataset.

        Returns:
            {
                'correct': fraction of correct predictions,
                'wrong': fraction of wrong predictions,
                'abstain': fraction of abstentions,
                'total': number of samples,
            }
        """
        correct = 0
        wrong = 0
        abstain = 0
        total = len(dataset)

        for sample in dataset:
            images = sample['images']
            label = sample['label']

            if images.ndim == 3:
                images = images[:, np.newaxis, :, :]

            result = self.deduce(images)
            decision = result['decision']

            if decision is None:
                abstain += 1
            elif decision == label:
                correct += 1
            else:
                wrong += 1

        return {
            'correct': correct / max(total, 1),
            'wrong': wrong / max(total, 1),
            'abstain': abstain / max(total, 1),
            'total': total,
        }

    # ------------------------------------------------------------------
    # Relative Fitness (vs parent)
    # ------------------------------------------------------------------

    def compute_relative_fitness(self, parent_results: List[Optional[bool]],
                                 own_results: List[Optional[bool]],
                                 labels: List[bool]) -> float:
        """
        Compute relative fitness vs parent using the score matrix from the paper.

        Score matrix (parent_outcome → offspring_outcome):
          correct → correct:   0
          correct → wrong:    -2
          correct → abstain:  -1
          wrong   → correct:  +2
          wrong   → wrong:     0
          wrong   → abstain:  +1
          abstain → correct:  +1
          abstain → wrong:    -1
          abstain → abstain:   0

        Args:
            parent_results: parent's decisions per sample (True/False/None)
            own_results: this organism's decisions
            labels: ground truth labels

        Returns:
            Relative fitness score (higher = better than parent)
        """
        score = 0.0

        for parent_dec, own_dec, label in zip(parent_results, own_results, labels):
            parent_cat = self._categorize(parent_dec, label)
            own_cat = self._categorize(own_dec, label)

            score += _SCORE_MATRIX.get((parent_cat, own_cat), 0)

        return score / max(len(labels), 1)

    @staticmethod
    def _categorize(decision: Optional[bool], label: bool) -> str:
        if decision is None:
            return 'abstain'
        elif decision == label:
            return 'correct'
        else:
            return 'wrong'

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def get_weights(self) -> Dict[str, np.ndarray]:
        """Get CNN weights for inheritance."""
        return {
            'conv1_W': self.cnn.conv1.W.copy(),
            'conv1_b': self.cnn.conv1.b.copy(),
            'fc1_W': self.cnn.fc1.W.copy(),
            'fc1_b': self.cnn.fc1.b.copy(),
            'fc2_W': self.cnn.fc2.W.copy(),
            'fc2_b': self.cnn.fc2.b.copy(),
        }

    def set_weights(self, weights: Dict[str, np.ndarray]):
        """Set CNN weights (for inheritance from parent)."""
        self.cnn.conv1.W = weights['conv1_W'].copy()
        self.cnn.conv1.b = weights['conv1_b'].copy()
        self.cnn.fc1.W = weights['fc1_W'].copy()
        self.cnn.fc1.b = weights['fc1_b'].copy()
        self.cnn.fc2.W = weights['fc2_W'].copy()
        self.cnn.fc2.b = weights['fc2_b'].copy()

    def copy_organism(self) -> 'NeSyOrganism':
        """Deep copy this organism."""
        new = NeSyOrganism(atoms=list(self.atoms))
        new.symbolic = self.symbolic.copy()
        new.set_weights(self.get_weights())
        return new


# Score matrix from the paper (Table 2)
_SCORE_MATRIX = {
    ('correct', 'correct'): 0,
    ('correct', 'wrong'): -2,
    ('correct', 'abstain'): -1,
    ('wrong', 'correct'): +2,
    ('wrong', 'wrong'): 0,
    ('wrong', 'abstain'): +1,
    ('abstain', 'correct'): +1,
    ('abstain', 'wrong'): -1,
    ('abstain', 'abstain'): 0,
}
