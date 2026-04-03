"""
Script to visualize generated expressions
"""
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.expression_dataset import ExpressionDataset

def visualize_samples():
    print("🎨 Generating samples to visualize...")
    
    # 1. Create the dataset (with 20% invalid ratio to ensure we see some weird ones)
    ds = ExpressionDataset(num_samples=10, split='train', invalid_ratio=0.2)
    
    # 2. Setup the plot
    fig, axes = plt.subplots(5, 1, figsize=(6, 12))
    fig.suptitle('Generated Neuro-Symbolic Expressions', fontsize=16)
    
    # 3. Loop through first 5 samples
    for i in range(5):
        sample = ds[i]
        images = sample['images']  # Shape is (3, 28, 28)
        text = sample['text']
        result = sample['result']
        
        # Concatenate the 3 images horizontally
        combined_image = np.concatenate([images[0], images[1], images[2]], axis=1)
        
        # Plot
        axes[i].imshow(combined_image, cmap='gray')
        
        # Handle display for Valid vs Invalid
        if result is not None:
            title = f"Valid: {text} = {result}"
            color = 'black'
        else:
            title = f"INVALID: {text} (Result: None)"
            color = 'red'
            
        axes[i].set_title(title, color=color, fontweight='bold')
        axes[i].axis('off')
    
    # 4. Save the result
    output_path = 'dataset_preview.png'
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"\n✅ Saved visualization to: {os.path.abspath(output_path)}")
    print("   Open this file to see your data!")

if __name__ == "__main__":
    visualize_samples()