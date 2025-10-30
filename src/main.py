# main.py

import numpy as np
from data.dataset import Dataset
from data.loader import DataLoader
from neural.model import NeuralModel
from symbolic.engine import SymbolicEngine

def main():
    # Generate synthetic dataset
    dataset = Dataset()
    data = dataset.generate_data()
    
    # Load data in batches
    data_loader = DataLoader(data, batch_size=32)
    
    # Initialize neural model
    neural_model = NeuralModel(input_size=data.shape[1], hidden_size=64, output_size=10)
    
    # Train the neural model
    for epoch in range(10):
        for batch in data_loader:
            neural_model.train(batch)
    
    # Initialize symbolic engine
    symbolic_engine = SymbolicEngine()
    
    # Example of parsing rules and reasoning
    rules = ["if A then B", "if B then C"]
    symbolic_engine.parse_rules(rules)
    results = symbolic_engine.reason(facts=["A"])
    
    print("Results from symbolic reasoning:", results)

if __name__ == "__main__":
    main()