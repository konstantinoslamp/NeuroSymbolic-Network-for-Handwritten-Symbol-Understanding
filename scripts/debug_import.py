
import sys
import os
sys.path.append(os.getcwd())

try:
    import src.neural.neural_interface as ni
    print(f"Module file: {ni.__file__}")
    print(f"Has NeuralModule: {hasattr(ni, 'NeuralModule')}")
    print(f"Dir: {dir(ni)}")
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
