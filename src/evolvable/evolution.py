"""
Evolutionary Algorithm for NeSy Organisms

Implements the Evolvable Policies evolutionary loop from
Thoma, Vassiliades & Michael (2026).

The algorithm maintains a population of NeSy organisms that evolve
through generations. Each generation:
  1. Take the current best organism as parent
  2. Generate offspring via all combinations of symbolic and neural mutations
  3. Train each offspring for a fixed number of epochs
  4. Evaluate relative fitness vs parent
  5. Select the next parent from beneficial/neutral/detrimental groups

Symbolic mutations:
  - S0: clone (no change to policy)
  - S+: add a random rule (elaboration)
  - S-: simplify (remove a rule or literal)

Neural mutations:
  - Npw: inherit parent weights (preserve learned features)
  - Nrw: random weight initialization (fresh start)
"""

import numpy as np
import random
import copy
import time
from typing import Dict, List, Tuple, Optional, Any, Callable

from src.evolvable.machine_coaching import Policy, Rule, Literal, MCSymbolicModule
from src.evolvable.organism import NeSyOrganism
from src.neural.cnn import Dense


# ---------------------------------------------------------------------------
# Symbolic Mutations
# ---------------------------------------------------------------------------

def mutate_S0(organism: NeSyOrganism) -> NeSyOrganism:
    """S0: Clone — copy policy unchanged."""
    return organism.copy_organism()


def mutate_Splus(organism: NeSyOrganism, atoms: List[str] = None) -> List[NeSyOrganism]:
    """
    S+: Rule addition — 10 offspring: 5 random bodies x 2 head polarities.

    Per the paper (Thoma et al. 2026): generate 5 candidate rules with
    random bodies, each tried with both h and -h heads, giving 10 offspring.
    Each appended rule has highest priority (MC elaboration tolerance).
    """
    atoms = atoms or organism.atoms
    offspring = []

    for _ in range(5):
        body_len = random.randint(1, min(4, len(atoms)))
        body_atoms = random.sample(atoms, body_len)
        body = [Literal(a, random.choice([True, False])) for a in body_atoms]

        for polarity in [True, False]:
            child = organism.copy_organism()
            head = Literal('h', polarity)
            rule_name = f"R{len(child.symbolic.policy.rules) + 1}"
            child.symbolic.induce(Rule(body, head, name=rule_name))
            offspring.append(child)

    return offspring


def mutate_Sminus(organism: NeSyOrganism) -> List[NeSyOrganism]:
    """
    S-: Simplification — one offspring per literal j in the LATEST rule's body.

    Per the paper (Thoma et al. 2026): for each literal j in the body of the
    highest-priority (last) rule, produce a new offspring whose policy has that
    rule replaced by a version with literal j dropped from its body.
    If the latest rule has only one literal, return a single clone.

    Returns:
        List of offspring, one per removable literal in the latest rule.
    """
    rules = organism.symbolic.policy.rules
    if not rules:
        return [organism.copy_organism()]

    latest_rule = rules[-1]

    if len(latest_rule.body) <= 1:
        return [organism.copy_organism()]

    offspring = []
    for j in range(len(latest_rule.body)):
        child = organism.copy_organism()
        child_rules = child.symbolic.policy.rules
        new_body = [lit for k, lit in enumerate(latest_rule.body) if k != j]
        new_rule = Rule(new_body, child_rules[-1].head, name=child_rules[-1].name)
        child_rules[-1] = new_rule
        offspring.append(child)

    return offspring


# ---------------------------------------------------------------------------
# Neural Mutations
# ---------------------------------------------------------------------------

def mutate_Npw(child: NeSyOrganism, parent: NeSyOrganism):
    """Npw: Inherit parent weights (preserve learned features)."""
    child.set_weights(parent.get_weights())


def mutate_Nrw(child: NeSyOrganism):
    """Nrw: Random weight initialization (fresh start)."""
    # Reinitialize all layers with fresh random weights
    from src.neural.cnn import Conv2D
    new_conv = Conv2D(
        in_channels=child.cnn.conv1.in_channels,
        out_channels=child.cnn.conv1.out_channels,
        kernel_size=child.cnn.conv1.kernel_size,
    )
    child.cnn.conv1.W = new_conv.W
    child.cnn.conv1.b = new_conv.b

    new_fc1 = Dense(
        in_features=child.cnn.fc1.W.shape[0],
        out_features=child.cnn.fc1.W.shape[1],
    )
    child.cnn.fc1.W = new_fc1.W
    child.cnn.fc1.b = new_fc1.b

    new_fc2 = Dense(
        in_features=child.cnn.fc2.W.shape[0],
        out_features=child.cnn.fc2.W.shape[1],
    )
    child.cnn.fc2.W = new_fc2.W
    child.cnn.fc2.b = new_fc2.b


# ---------------------------------------------------------------------------
# Offspring Generation
# ---------------------------------------------------------------------------

SYMBOLIC_MUTATIONS = {
    'S0': mutate_S0,
    'S+': mutate_Splus,
    'S-': mutate_Sminus,
}

NEURAL_MUTATIONS = {
    'Npw': mutate_Npw,
    'Nrw': mutate_Nrw,
}


def generate_offspring(parent: NeSyOrganism,
                       atoms: List[str] = None) -> List[Tuple[str, NeSyOrganism]]:
    """
    Generate offspring from all symbolic x neural mutation combinations.

    S0 produces 2 offspring (Npw, Nrw).
    S+ produces 20 offspring (10 symbolic variants x 2 neural mutations).
    S- produces 2*|body_of_latest_rule| offspring (one per literal x 2 neural).

    Returns:
        List of (mutation_label, offspring) tuples.
    """
    offspring = []
    _atoms = atoms or parent.atoms

    # S0: clone, both neural mutations
    for n_name, n_fn in NEURAL_MUTATIONS.items():
        child = mutate_S0(parent)
        if n_name == 'Npw':
            n_fn(child, parent)
        else:
            n_fn(child)
        offspring.append((f"S0_{n_name}", child))

    # S+: 10 symbolic variants, each with both neural mutations
    splus_children = mutate_Splus(parent, _atoms)
    for i, child in enumerate(splus_children):
        for n_name, n_fn in NEURAL_MUTATIONS.items():
            c = child.copy_organism()
            if n_name == 'Npw':
                n_fn(c, parent)
            else:
                n_fn(c)
            offspring.append((f"S+{i}_{n_name}", c))

    # S-: one child per removable literal, each with both neural mutations
    sminus_children = mutate_Sminus(parent)
    for i, child in enumerate(sminus_children):
        for n_name, n_fn in NEURAL_MUTATIONS.items():
            c = child.copy_organism()
            if n_name == 'Npw':
                n_fn(c, parent)
            else:
                n_fn(c)
            offspring.append((f"S-{i}_{n_name}", c))

    return offspring


# ---------------------------------------------------------------------------
# Selection Mechanism
# ---------------------------------------------------------------------------

def select_next_parent(parent: NeSyOrganism,
                       offspring_results: List[Dict[str, Any]],
                       labels: List[bool]) -> Tuple[NeSyOrganism, Dict[str, Any]]:
    """
    Select next generation's parent using the 3-group selection mechanism.

    Groups:
      1. Beneficial: relative fitness > 0 (better than parent)
      2. Neutral: relative fitness == 0 (same as parent)
      3. Detrimental: relative fitness < 0 (worse than parent)

    Selection priority:
      - If beneficial group non-empty: k=2 fitness-proportionate selection from it
      - Elif neutral group non-empty: uniform random from it
      - Else: select best (least harmful) detrimental offspring (never keep parent)

    Args:
        parent: current parent organism
        offspring_results: list of dicts with keys:
            'organism', 'label', 'fitness', 'eval_results', 'decisions'
        labels: ground truth labels for the evaluation set

    Returns:
        (selected_organism, selection_info)
    """
    beneficial = [r for r in offspring_results if r['fitness'] > 0]
    neutral = [r for r in offspring_results if r['fitness'] == 0]
    detrimental = [r for r in offspring_results if r['fitness'] < 0]

    info = {
        'num_beneficial': len(beneficial),
        'num_neutral': len(neutral),
        'num_detrimental': len(detrimental),
    }

    if beneficial:
        # Fitness-proportionate selection with k=2 exponent (paper spec)
        fitnesses = np.array([r['fitness'] for r in beneficial], dtype=float)
        weights = fitnesses ** 2
        probs = weights / weights.sum()
        idx = np.random.choice(len(beneficial), p=probs)
        selected = beneficial[idx]
        info['selection'] = 'beneficial'
        info['selected_label'] = selected['label']
        info['selected_fitness'] = selected['fitness']
        return selected['organism'], info

    elif neutral:
        # Uniform random selection from neutral
        idx = random.randint(0, len(neutral) - 1)
        selected = neutral[idx]
        info['selection'] = 'neutral'
        info['selected_label'] = selected['label']
        info['selected_fitness'] = 0.0
        return selected['organism'], info

    else:
        # Select best detrimental offspring (least harmful) — paper spec
        # Do NOT keep parent; always move to best available offspring
        best = max(detrimental, key=lambda r: r['fitness'])
        info['selection'] = 'detrimental'
        info['selected_label'] = best['label']
        info['selected_fitness'] = best['fitness']
        return best['organism'], info


# ---------------------------------------------------------------------------
# Evolutionary Loop
# ---------------------------------------------------------------------------

class EvolutionaryEngine:
    """
    Runs the full evolutionary process for learning NeSy policies.

    The engine manages the generational loop:
      1. Generate offspring (6 per generation)
      2. Train each offspring
      3. Evaluate and compute relative fitness
      4. Select next parent
      5. Repeat until convergence or max generations
    """

    def __init__(self, atoms: List[str] = None,
                 num_atoms: int = 8,
                 train_epochs: int = 3,
                 learning_rate: float = 0.001,
                 max_generations: int = 50,
                 early_stop_accuracy: float = 0.99,
                 verbose: bool = True):
        """
        Args:
            atoms: atom names
            num_atoms: number of atoms (if atoms not provided)
            train_epochs: epochs to train each offspring per generation
            learning_rate: SGD learning rate
            max_generations: maximum number of generations
            early_stop_accuracy: stop if val accuracy reaches this
            verbose: print progress
        """
        self.atoms = atoms or [f"a{i+1}" for i in range(num_atoms)]
        self.num_atoms = len(self.atoms)
        self.train_epochs = train_epochs
        self.learning_rate = learning_rate
        self.max_generations = max_generations
        self.early_stop_accuracy = early_stop_accuracy
        self.verbose = verbose

        # Tracking
        self.history = []
        self.lineage = []  # (generation, mutation_label, fitness)

    def run(self, train_data: List[Dict],
            val_data: List[Dict],
            test_data: List[Dict] = None,
            initial_organism: NeSyOrganism = None) -> Dict[str, Any]:
        """
        Run the full evolutionary process.

        Args:
            train_data: training samples [{'images': ndarray, 'label': bool}, ...]
            val_data: validation samples
            test_data: optional test samples for final evaluation
            initial_organism: starting organism (default: random)

        Returns:
            {
                'best_organism': NeSyOrganism,
                'history': list of per-generation stats,
                'lineage': list of (gen, mutation, fitness),
                'final_test': test results if test_data provided,
            }
        """
        # Initialize parent
        if initial_organism is not None:
            parent = initial_organism
        else:
            parent = NeSyOrganism(atoms=self.atoms)

        # Initial evaluation
        parent_eval = parent.evaluate(val_data)
        parent_decisions = self._get_decisions(parent, val_data)
        val_labels = [s['label'] for s in val_data]

        if self.verbose:
            print(f"Generation 0 (initial): "
                  f"correct={parent_eval['correct']:.3f} "
                  f"wrong={parent_eval['wrong']:.3f} "
                  f"abstain={parent_eval['abstain']:.3f}")

        self.history.append({
            'generation': 0,
            'parent_eval': parent_eval,
            'selection': 'initial',
        })

        best_accuracy = parent_eval['correct']

        for gen in range(1, self.max_generations + 1):
            gen_start = time.time()

            # Early stopping
            if best_accuracy >= self.early_stop_accuracy:
                if self.verbose:
                    print(f"Early stop: {best_accuracy:.3f} >= {self.early_stop_accuracy}")
                break

            # Step 1: Generate offspring
            offspring_list = generate_offspring(parent, self.atoms)

            # Step 2: Train and evaluate each offspring
            offspring_results = []
            for label, child in offspring_list:
                # Train
                for epoch in range(self.train_epochs):
                    child.train_epoch(train_data, self.learning_rate)

                # Evaluate
                child_eval = child.evaluate(val_data)
                child_decisions = self._get_decisions(child, val_data)

                # Relative fitness
                fitness = child.compute_relative_fitness(
                    parent_decisions, child_decisions, val_labels
                )

                offspring_results.append({
                    'organism': child,
                    'label': label,
                    'fitness': fitness,
                    'eval_results': child_eval,
                    'decisions': child_decisions,
                })

            # Step 3: Select next parent
            parent, selection_info = select_next_parent(
                parent, offspring_results, val_labels
            )

            # Update parent state
            parent_eval = parent.evaluate(val_data)
            parent_decisions = self._get_decisions(parent, val_data)
            best_accuracy = max(best_accuracy, parent_eval['correct'])

            # Record
            gen_time = time.time() - gen_start
            gen_record = {
                'generation': gen,
                'parent_eval': parent_eval,
                'selection': selection_info,
                'offspring_fitnesses': {
                    r['label']: r['fitness'] for r in offspring_results
                },
                'offspring_accuracies': {
                    r['label']: r['eval_results']['correct'] for r in offspring_results
                },
                'time': gen_time,
            }
            self.history.append(gen_record)
            self.lineage.append((
                gen,
                selection_info.get('selected_label', 'parent'),
                selection_info.get('selected_fitness', 0.0),
            ))

            if self.verbose:
                print(f"Gen {gen}: "
                      f"correct={parent_eval['correct']:.3f} "
                      f"wrong={parent_eval['wrong']:.3f} "
                      f"abstain={parent_eval['abstain']:.3f} | "
                      f"selected={selection_info.get('selected_label', 'parent')} "
                      f"({selection_info['selection']}) "
                      f"B={selection_info['num_beneficial']} "
                      f"N={selection_info['num_neutral']} "
                      f"D={selection_info['num_detrimental']} "
                      f"[{gen_time:.1f}s]")

        # Final test evaluation
        result = {
            'best_organism': parent,
            'history': self.history,
            'lineage': self.lineage,
            'best_val_accuracy': best_accuracy,
        }

        if test_data is not None:
            test_eval = parent.evaluate(test_data)
            result['final_test'] = test_eval
            if self.verbose:
                print(f"\nFinal test: correct={test_eval['correct']:.3f} "
                      f"wrong={test_eval['wrong']:.3f} "
                      f"abstain={test_eval['abstain']:.3f}")

        return result

    def _get_decisions(self, organism: NeSyOrganism,
                       dataset: List[Dict]) -> List[Optional[bool]]:
        """Get organism's decisions for each sample in the dataset."""
        decisions = []
        for sample in dataset:
            images = sample['images']
            if images.ndim == 3:
                images = images[:, np.newaxis, :, :]
            result = organism.deduce(images)
            decisions.append(result['decision'])
        return decisions


# ---------------------------------------------------------------------------
# Experiment Summary
# ---------------------------------------------------------------------------

def summarize_evolution(history: List[Dict]) -> Dict[str, Any]:
    """Produce a summary of the evolutionary run."""
    if not history:
        return {}

    accuracies = [h['parent_eval']['correct'] for h in history]
    wrong_rates = [h['parent_eval']['wrong'] for h in history]
    abstain_rates = [h['parent_eval']['abstain'] for h in history]

    # Count selection types
    selections = {}
    for h in history[1:]:  # skip generation 0
        sel = h['selection']
        if isinstance(sel, dict):
            sel_type = sel.get('selection', 'unknown')
        else:
            sel_type = sel
        selections[sel_type] = selections.get(sel_type, 0) + 1

    return {
        'num_generations': len(history) - 1,
        'initial_accuracy': accuracies[0],
        'final_accuracy': accuracies[-1],
        'best_accuracy': max(accuracies),
        'accuracy_trajectory': accuracies,
        'final_wrong_rate': wrong_rates[-1],
        'final_abstain_rate': abstain_rates[-1],
        'selection_counts': selections,
    }
