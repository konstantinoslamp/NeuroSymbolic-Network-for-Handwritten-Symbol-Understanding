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

The neurosymbolic system implements the compositional interface pattern from **Tsamoura et al. (AAAI 2021)**:

```
+----------------------------------------------------------------------+
|                       PERCEPTION LAYER                              |
|                                                                      |
|   +------------------+         +------------------+                 |
|   |  DigitCNN        |         |  OperatorCNN     |                 |
|   |  (ResNet-9)      |         |  (ResNet-9)      |                 |
|   |  10-class        |         |  4-class         |                 |
|   |  Softmax + T     |         |  Softmax + T     |  T = temp scale |
|   |  P(d | img)      |         |  P(op | img)     |                 |
|   +--------+---------+         +---------+--------+                 |
|            +------------------+----------+                          |
|                               |                                     |
|            Probabilistic Symbol Stream                              |
|            { P(d1), P(op), P(d2) }  per image triplet              |
+-------------------------------+--------------------------------------+
                                |
                                v
+----------------------------------------------------------------------+
|                  NEURAL-SYMBOLIC BRIDGE                             |
|                                                                      |
|  NeuralModule Interface                                              |
|  +------------------------------------------------------------------+|
|  | neural_deduction(images)  ->  { probs, logits, symbols }        ||
|  | neural_induction(images, semantic_loss_grad)  ->  delta_theta   ||
|  +------------------------------------------------------------------+|
|                    <->  clean interface                             |
|  SymbolicModule Interface                                            |
|  +------------------------------------------------------------------+|
|  | symbolic_deduction(symbols)  ->  { valid, result, proof }       ||
|  | symbolic_abduction(target, probs)  ->  [{ e*, P(e*) }]         ||
|  +------------------------------------------------------------------+|
+-------------------------------+--------------------------------------+
                                |
                                v
+----------------------------------------------------------------------+
|                  SYMBOLIC REASONING LAYER                           |
|                                                                      |
|  Knowledge Base (Datalog / ASP)                                     |
|    digit(0..9).   operator(+,-,x,/).                                |
|    valid_expr(D,Op,D2) :- digit(D), operator(Op), digit(D2).       |
|    result(D1,+,D2,R)   :- R is D1 + D2.                            |
|    no_div_zero(_,/,D2) :- D2 != 0.                                 |
|                                                                      |
|  +--------------------+   +--------------------------------+         |
|  | Deduction Engine   |   | Abduction Engine               |         |
|  | (forward eval)     |   | AC-3 / SAT / ILP               |         |
|  | KB |- result(e)    |   | Returns: all e* s.t. result=R  |         |
|  +--------------------+   | + P(e* | neural_probs)         |         |
|                           +--------------------------------+         |
+-------------------------------+--------------------------------------+
                                |
                                v
+----------------------------------------------------------------------+
|                   SEMANTIC LOSS LAYER                               |
|                                                                      |
|   L  = -log SUM_{e* in E*}  P(e* | neural_probs)                   |
|   dL/d_logit_k  =  P(k | img) - T(k | abduction)                  |
|   T(k) = SUM_{e*: k in e*} w(e*) * 1[e* contains k]               |
+-------------------------------+--------------------------------------+
                                |
                                v
+----------------------------------------------------------------------+
|                    EVALUATION LAYER                                 |
|   - Per-class accuracy (digits + operators separately)               |
|   - Expression accuracy (full d op d correct?)                      |
|   - Result accuracy (arithmetic output correct?)                    |
|   - Abduction rate & explanation quality                            |
|   - NGA vs. WMC ablation                                            |
|   - Calibration: ECE + reliability diagrams                         |
+----------------------------------------------------------------------+
```

### Component Details

**Perception Layer** (`src/neural/`)
- `cnn.py`: Base CNN architecture for digit recognition
- `digit_recognizer.py`: 10-class softmax with temperature scaling
- `neural_interface.py`: Exposes `neural_deduction()` and `neural_induction()`

**Symbolic Reasoning Layer** (`src/symbolic/`)
- `knowledge_base.py`: ASP/Datalog KB with arithmetic rules, `deduce()` and `abduce()`
- `constraints.py`: Constraint registry with Python pre-checks
- `deduction.py`: 4-stage pipeline (structural → type → Python → ASP query)
- `abduction.py`: Backward inference with probability scoring
- `symbolic_interface.py`: Clean module interface contract

**Integration Layer** (`src/integration/`)
- `semantic_loss.py`: Weighted Model Counting via d-DNNF circuits, exact WMC/NGA modes
- `training_loop.py`: Closed-loop training with neural-symbolic coupling

**Evaluation & Monitoring** (`src/evaluation/`, `src/utils/`)
- `metrics.py`: Per-class, expression, result accuracy; calibration; confusion matrices
- `ablation_studies.py`: NGA vs. WMC vs. pure neural comparison
- `gradient_monitor.py`: Weight snapshots, gradient norm logging, sanity checks

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