# Neurosymbolic Arithmetic Solver - MVP

Complete implementation of a neurosymbolic AI system combining neural perception with symbolic reasoning for handwritten arithmetic expression solving.

## Overview

This educational project demonstrates how neural networks and symbolic AI work together:
- **Neural Network (CNN)**: Recognizes handwritten digits (0-9) and operators (+, -, ×, ÷)
- **Symbolic Reasoner**: Validates expressions and computes results
- **Bridge Layer**: Connects perception with reasoning

## Quick Start

### 1. Setup Data
```bash
cd src/neural

# Download MNIST
python -c "from urllib.request import urlretrieve; urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', 'mnist_data.npz'); print('Downloaded!')"

# Generate operators
python -c "import numpy as np; exec(open('generate_operators.py').read() if os.path.exists('generate_operators.py') else 'print(\"Create operators.npz manually\")')"
```

### 2. Train Model
```bash
python train.py  # Takes ~5-10 minutes
```

### 3. Run Application
```bash
cd ../..
python src/ui_app.py
```

Draw "3+7" and click "Recognize & Solve"!

```
neurosymbolic_mvp
├── src
│   ├── data
│   │   ├── dataset.py        # Class for generating and managing synthetic datasets
│   │   └── loader.py         # Class for loading datasets in batches
│   ├── neural
│   │   └── model.py          # Class implementing a multi-layer perceptron (MLP)
│   ├── symbolic
│   │   └── engine.py         # Class for symbolic reasoning engine
│   ├── utils
│   │   └── helpers.py        # Utility functions and helper methods
│   └── main.py               # Entry point for the application
├── tests
│   ├── test_data.py          # Unit tests for dataset and data loader
│   ├── test_neural.py        # Unit tests for the neural model
│   ├── test_symbolic.py      # Unit tests for the symbolic reasoning engine
│   └── test_utils.py         # Unit tests for utility functions
├── requirements.txt           # Project dependencies
├── pyproject.toml            # Project configuration
└── README.md                 # Project documentation
```

## Getting Started

To get started with the project, follow these steps:

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd neurosymbolic_mvp
   ```

2. **Install dependencies:**
   Use the following command to install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. **Run the application:**
   Execute the main script to start the application:
   ```
   python src/main.py
   ```

## Testing

To run the tests, use the following command:
```
pytest tests/
```

## Architecture Overview

The neurosymbolic system implements the compositional interface pattern from **Tsamoura et al. (AAAI 2021)**, with four core modules:

### Neural Layer
- **`src/neural/cnn.py`** & **`digit_recognizer.py`**: CNN-based perception, outputs softmax probabilities P(symbol | image)
- **`neural_interface.py`**: Exposes `neural_deduction()` (forward pass) and `neural_induction()` (gradient updates)

### Symbolic Reasoning Layer
- **`src/symbolic/knowledge_base.py`**: ASP-style Datalog knowledge base with arithmetic rules
  - Digits, operators, valid expressions, arithmetic results, constraints
  - `deduce()`: forward evaluation → proof traces
  - `abduce()`: backward enumeration → all KB-consistent explanations
  
- **`src/symbolic/constraints.py`**: Structured constraint registry with Python pre-checks
  - `no_division_by_zero`, `valid_digit_range`, `valid_operator`
  
- **`src/symbolic/deduction.py`**: 4-stage pipeline
  - Structural check → Type check → Python constraints → ASP query
  - Returns full derivation trace
  
- **`src/symbolic/abduction.py`**: Explains invalid predictions
  - Scores explanations by `log P(symbols | neural_probs)`
  - Returns ranked list with plausibility scores
  
- **`src/symbolic/symbolic_interface.py`**: Clean module contract
  - `symbolic_deduction()` and `symbolic_abduction()` API

### Integration Layer
- **`src/integration/semantic_loss.py`**: Weighted Model Counting (WMC) via d-DNNF
  - Compiles valid KB models to arithmetic circuits
  - Returns semantic loss = `-log(WMC)` and backprop gradients
  - Two strategies: exact WMC or NGA (top-1)

- **`src/integration/training_loop.py`**: Closed-loop training
  - Neural deduction → symbolic reasoning → semantic loss → gradient flow

### Evaluation & Monitoring
- **`src/evaluation/metrics.py`**: Full evaluation suite
  - Per-class accuracy (digits 0-9 + operators separately)
  - Expression-level accuracy (full `d op d` correct?)
  - Result accuracy (arithmetic output correct?)
  - Calibration metrics (ECE, reliability diagrams)
  - Confusion matrices with precision/recall/F1

- **`src/evaluation/ablation_studies.py`**: Ablation framework
  - Pure Neural vs. NGA vs. WMC comparison
  - Same initialization, side-by-side metrics

- **`src/utils/gradient_monitor.py`**: Gradient flow verification
  - Weights snapshots before/after training
  - Null gradient and exploding/vanishing detection
  - Full gradient report with safety checks

## Metrics and Evaluation

The system is evaluated on:

- **Per-Class Accuracy**: Separate metrics for digits and operators
- **Expression Accuracy**: Full expression recognition (3-token: `digit op digit`)
- **Result Accuracy**: Arithmetic correctness (e.g., 3+7=10)
- **Abduction Rate**: Percentage of examples requiring symbolic correction
- **Calibration (ECE)**: Neural confidence alignment with actual accuracy
- **Ablation**: NGA vs. WMC vs. pure neural baselines

## Recent Changes (Phase 1)

✅ **Symbolic Engine**: Replaced Python conditionals with real Datalog/ASP-style knowledge base
✅ **Semantic Loss**: Implemented exact WMC via d-DNNF arithmetic circuits
✅ **Evaluation Suite**: Full metrics, ablations, confusion matrices, calibration
✅ **Gradient Monitoring**: Weight change tracking, gradient norm logging, sanity checks
✅ **Training Integration**: Closed-loop neurosymbolic trainer with semantic loss

## Next Roadmap

- [ ] **MATH(n)**: Scale to variable-length expressions (length 5, 7, ...)
- [ ] **Abduction Efficiency**: Replace brute-force with constraint propagation (AC-3/SAT)
- [ ] **Separate Operator CNN**: Dedicated recognizer for operators vs. digits
- [ ] **PATH(n) Task**: Implement graph pathfinding to demonstrate compositionality
- [ ] **Rule Learning**: Inductive Logic Programming to learn symbolic rules from data
- [ ] **Counterfactual Explanations**: Leverage abduction for XAI

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.