import numpy as np
import sys
sys.path.append('.')
from generate_operators import generate_operator_dataset

# Generate dataset
images, labels = generate_operator_dataset(num_samples=1000)

# Save
np.savez('operators.npz', images=images, labels=labels)
print(f"Saved {len(images)} operator images to operators.npz")
print(f"Labels: 10=+, 11=-, 12=×, 13=÷")
