"""
Evolvable Policies Framework

Implements the framework from:
  Thoma, Vassiliades & Michael (2026)
  "Neural-Symbolic Integration with Evolvable Policies"

Combines:
  - Extended NEUROLOG architecture (Tsamoura et al. 2021)
  - Valiant's Evolvability framework
  - Machine Coaching semantics (Michael 2019)
"""

from src.evolvable.machine_coaching import (
    Literal, Rule, Policy, PolicyGenerator, MCSymbolicModule,
)
from src.evolvable.translator import Translator
from src.evolvable.organism import NeSyOrganism
from src.evolvable.evolution import (
    EvolutionaryEngine, generate_offspring, select_next_parent,
    summarize_evolution,
)
from src.evolvable.dataset import EvolvableDataset, create_experiment_data
from src.evolvable.run_experiment import run_single_experiment
