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

The architecture of the neuro-symbolic system consists of two main components:

- **Neural Component:** Implemented in `src/neural/model.py`, this component uses a multi-layer perceptron (MLP) to perform tasks such as classification and regression.

- **Symbolic Component:** Implemented in `src/symbolic/engine.py`, this component handles symbolic reasoning, allowing the system to parse rules and perform logical reasoning based on the data.

## Metrics and Roadmap

The project aims to achieve the following metrics:

- Accuracy of the neural model
- Efficiency of the symbolic reasoning engine
- Integration performance between neural and symbolic components

Future enhancements may include:

- Expanding the dataset generation capabilities
- Improving the neural model architecture
- Enhancing the symbolic reasoning algorithms

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.