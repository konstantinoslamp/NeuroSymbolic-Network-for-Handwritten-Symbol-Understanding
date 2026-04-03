"""
Test Expression Dataset Generator
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from src.data.expression_dataset import ExpressionDataset

def test_dataset_generation():
    print("📊 Testing Dataset Generator...")
    
    # Generate small dataset
    ds = ExpressionDataset(num_samples=10, split='train')
    
    print(f"   Generated {len(ds)} samples")
    
    # Check first sample
    sample = ds[0]
    images = sample['images']
    result = sample['result']
    text = sample['text']
    
    print(f"   Sample 0: {text} = {result} (Type: {type(result)})")
    print(f"   Image shape: {images.shape}")
    
    assert images.shape == (3, 28, 28), "Should be 3 images of 28x28"
    
    # FIX: Check for both Python numbers and Numpy numbers
    assert isinstance(result, (int, float, np.number)), f"Result should be a number, got {type(result)}"
    
    print("✅ Dataset generation works!")

if __name__ == "__main__":
    test_dataset_generation()