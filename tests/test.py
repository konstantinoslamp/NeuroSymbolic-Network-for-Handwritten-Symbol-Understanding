"""
Test the complete Neurosymbolic pipeline
"""
import sys
sys.path.append('src/bridge')
from neurosymbolic_connector import NeurosymbolicSolver
import numpy as np

# Initialize solver with trained model
solver = NeurosymbolicSolver('src/neural/trained_cnn_model.pkl')

# Create mock images (you can replace these with real MNIST images later)
def create_mock_image(value, size=28):
    """Create a simple mock image for testing"""
    img = np.random.rand(size, size) * 0.1  # Mostly zeros
    # Add some pattern in the center
    center = size // 2
    img[center-3:center+3, center-3:center+3] = 0.8
    return img

# Test expression: "3 + 7"
print("\nTesting with mock images for '3 + 7':")
print("-" * 70)

test_images = [
    create_mock_image(3),  # digit 3
    create_mock_image('+'),  # operator +
    create_mock_image(7)   # digit 7
]

result = solver.solve_expression(test_images)

if result['success']:
    print(f"\n✓ SUCCESS: {result['explanation']}")
    print(f"   Result: {result['result']}")
else:
    print(f"\n✗ FAILED: {result['explanation']}")