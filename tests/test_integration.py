"""
End-to-end integration test for the neurosymbolic system.
Run this AFTER training completes and trained_cnn_model.pkl exists.
"""
import sys
import os
sys.path.append('src')

import numpy as np
from src.bridge.neurosymbolic_connector import NeurosymbolicSolver

def create_test_images():
    """Create dummy test images (28x28) for testing"""
    # In real usage, these come from the drawing UI segmentation
    
    # Create 3 blank images with some noise
    images = []
    for i in range(3):
        img = np.random.rand(28, 28) * 0.1  # Mostly black with slight noise
        images.append(img)
    
    return images

def main():
    print("="*70)
    print("INTEGRATION TEST: Neural CNN → Symbolic Reasoner")
    print("="*70)
    
    # Check if model exists
    model_path = 'src/neural/trained_cnn_model.pkl'
    if not os.path.exists(model_path):
        print(f"\n❌ ERROR: Model file '{model_path}' not found!")
        print("Please run training first: python src/neural/train.py")
        return
    
    print(f"\n✓ Found model: {model_path}")
    
    # Initialize solver
    print("\n" + "-"*70)
    print("Initializing NeurosymbolicSolver...")
    print("-"*70)
    solver = NeurosymbolicSolver(model_path)
    
    # Test with dummy images
    print("\n" + "-"*70)
    print("Testing with synthetic images (random noise)...")
    print("-"*70)
    test_images = create_test_images()
    
    result = solver.solve_expression(test_images)
    
    # Display results
    print("\n" + "="*70)
    print("INTEGRATION TEST RESULTS")
    print("="*70)
    print(f"Success: {result['success']}")
    print(f"Expression: {result['expression']}")
    print(f"Result: {result['result']}")
    print(f"Explanation: {result['explanation']}")
    
    if result.get('validations'):
        print("\nValidation messages:")
        for msg in result['validations']:
            print(f"  {msg}")
    
    print("\n" + "="*70)
    
    if result['success']:
        print("✓ Integration test PASSED - All components connected!")
    else:
        print("⚠ Integration test completed with validation issues")
        print("  (This is expected with random images)")
    
    print("\nNext steps:")
    print("1. Run the UI: python src/ui_app.py")
    print("2. Draw an expression like '3+7'")
    print("3. Click 'Solve' to see the full pipeline in action!")

if __name__ == '__main__':
    main()
