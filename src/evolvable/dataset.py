"""
Dataset Generation for Evolvable Policies Experiments

Generates training/validation/test data from a hidden target policy.
Each sample consists of:
  - A sequence of MNIST images (one per atom)
  - A label (True/False) from the target policy's decision

Convention (from the paper):
  - MNIST digit 1 → atom is positive (True)
  - MNIST digit 2 → atom is negative (False)
  - Random context: each atom independently set to True/False
  - Images selected from MNIST pool matching the atom's truth value
"""

import numpy as np
import os
from typing import Any, Dict, List, Tuple, Optional

from src.evolvable.machine_coaching import Policy, PolicyGenerator


def load_mnist_subset(data_path: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load MNIST digits 1 and 2 for generating atom images.

    Returns:
        (images_dict, labels_dict) where:
          images_dict[1] = array of digit-1 images (N1, 28, 28)
          images_dict[2] = array of digit-2 images (N2, 28, 28)
    """
    if data_path is None:
        data_path = os.path.join(
            os.path.dirname(__file__), '..', 'data', 'mnist.npz'
        )

    if os.path.exists(data_path):
        data = np.load(data_path, allow_pickle=True)
        # Handle different MNIST file formats
        if 'x_train' in data:
            images = data['x_train']
            labels = data['y_train']
        elif 'images' in data:
            images = data['images']
            labels = data['labels']
        else:
            # Try first two arrays
            keys = list(data.keys())
            images = data[keys[0]]
            labels = data[keys[1]]

        # Filter digits 1 and 2
        mask_1 = labels == 1
        mask_2 = labels == 2

        imgs_1 = images[mask_1]
        imgs_2 = images[mask_2]

        # Normalize to [0, 1]
        if imgs_1.max() > 1.0:
            imgs_1 = imgs_1.astype(np.float32) / 255.0
            imgs_2 = imgs_2.astype(np.float32) / 255.0

        return imgs_1, imgs_2

    else:
        # Generate synthetic 28x28 images if MNIST not available
        print(f"Warning: MNIST not found at {data_path}, using synthetic images")
        return _generate_synthetic_images()


def _generate_synthetic_images(n_per_class: int = 500):
    """Generate simple synthetic 28x28 images for digits 1 and 2."""
    imgs_1 = np.zeros((n_per_class, 28, 28), dtype=np.float32)
    imgs_2 = np.zeros((n_per_class, 28, 28), dtype=np.float32)

    for i in range(n_per_class):
        # Digit 1: vertical line in center with noise
        img = np.random.rand(28, 28) * 0.1
        img[5:23, 13:16] = 0.8 + np.random.rand(18, 3) * 0.2
        imgs_1[i] = img

        # Digit 2: rough '2' shape with noise
        img = np.random.rand(28, 28) * 0.1
        img[5:8, 10:20] = 0.8 + np.random.rand(3, 10) * 0.2   # top bar
        img[8:14, 17:20] = 0.8 + np.random.rand(6, 3) * 0.2   # right side
        img[13:16, 10:20] = 0.8 + np.random.rand(3, 10) * 0.2  # middle bar
        img[15:21, 8:11] = 0.8 + np.random.rand(6, 3) * 0.2   # left side
        img[20:23, 8:20] = 0.8 + np.random.rand(3, 12) * 0.2  # bottom bar
        imgs_2[i] = img

    return imgs_1, imgs_2


class EvolvableDataset:
    """
    Generates labeled data from a hidden target policy using MNIST images.

    Each sample:
      - Random context: each atom is True or False (uniform)
      - For each atom: select an MNIST image (digit 1 if True, digit 2 if False)
      - Label: target_policy.deduce(context)
      - Samples where the policy abstains are discarded
    """

    def __init__(self, target_policy: Policy, atoms: List[str],
                 mnist_path: str = None):
        self.target_policy = target_policy
        self.atoms = atoms
        self.num_atoms = len(atoms)

        # Load MNIST digits
        self.imgs_1, self.imgs_2 = load_mnist_subset(mnist_path)

    def generate(self, num_samples: int = 500,
                 seed: int = None) -> List[Dict]:
        """
        Generate labeled samples.

        Args:
            num_samples: target number of samples (may produce fewer
                        if many contexts cause abstention)
            seed: random seed

        Returns:
            List of {'images': (num_atoms, 1, 28, 28), 'label': bool,
                     'context': {atom: bool}}
        """
        if seed is not None:
            np.random.seed(seed)

        samples = []
        attempts = 0
        max_attempts = num_samples * 5  # Allow some abstentions

        while len(samples) < num_samples and attempts < max_attempts:
            attempts += 1

            # Random context
            context = {}
            for atom in self.atoms:
                context[atom] = bool(np.random.random() > 0.5)

            # Get target decision
            decision = self.target_policy.deduce(context)

            if decision is None:
                continue  # Skip abstentions

            # Generate images
            images = np.zeros((self.num_atoms, 1, 28, 28), dtype=np.float32)
            for i, atom in enumerate(self.atoms):
                if context[atom]:
                    # Positive atom → digit 1
                    idx = np.random.randint(len(self.imgs_1))
                    images[i, 0] = self.imgs_1[idx]
                else:
                    # Negative atom → digit 2
                    idx = np.random.randint(len(self.imgs_2))
                    images[i, 0] = self.imgs_2[idx]

            samples.append({
                'images': images,
                'label': decision,
                'context': context,
            })

        if seed is not None:
            np.random.seed(None)

        return samples

    def generate_splits(self, train_size: int = 300,
                        val_size: int = 100,
                        test_size: int = 100,
                        seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Generate train/val/test splits.

        Returns:
            (train_data, val_data, test_data)
        """
        total = train_size + val_size + test_size
        all_data = self.generate(total, seed=seed)

        # Shuffle
        np.random.seed(seed)
        indices = np.random.permutation(len(all_data))

        train_end = min(train_size, len(all_data))
        val_end = min(train_size + val_size, len(all_data))

        train_data = [all_data[i] for i in indices[:train_end]]
        val_data = [all_data[i] for i in indices[train_end:val_end]]
        test_data = [all_data[i] for i in indices[val_end:]]

        np.random.seed(None)

        return train_data, val_data, test_data


def create_experiment_data(num_atoms: int = 8,
                           num_rules: int = 5,
                           policy_seed: int = 42,
                           data_seed: int = 123,
                           train_size: int = 300,
                           val_size: int = 100,
                           test_size: int = 100,
                           mnist_path: str = None) -> Dict[str, Any]:
    """
    Create a complete experiment dataset from a random target policy.

    Returns:
        {
            'target_policy': Policy,
            'atoms': list of atom names,
            'train': list of samples,
            'val': list of samples,
            'test': list of samples,
        }
    """
    atoms = [f"a{i+1}" for i in range(num_atoms)]

    # Generate target policy
    gen = PolicyGenerator(atoms=atoms)
    target_policy = gen.generate(num_rules=num_rules, seed=policy_seed)

    # Generate data
    dataset = EvolvableDataset(target_policy, atoms, mnist_path)
    train, val, test = dataset.generate_splits(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        seed=data_seed,
    )

    return {
        'target_policy': target_policy,
        'atoms': atoms,
        'train': train,
        'val': val,
        'test': test,
    }
