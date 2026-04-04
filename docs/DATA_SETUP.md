# Data Setup Guide

This folder should contain the datasets needed for training. These files are **NOT** included in git due to size.

## Required Files

### 1. MNIST Dataset (`mnist_data.npz`)
**Size**: ~11 MB  
**Source**: http://yann.lecun.com/exdb/mnist/

#### Download Method 1 (Python):
```bash
python -c "from urllib.request import urlretrieve; urlretrieve('https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz', 'mnist_data.npz'); print('✓ Downloaded MNIST!')"
```

#### Download Method 2 (Manual):
1. Go to: https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
2. Save as `mnist_data.npz` in `src/neural/` folder

**Contents**:
- Training images: 60,000 samples (28×28 grayscale)
- Test images: 10,000 samples (28×28 grayscale)
- Labels: 0-9 (digits)

### 2. Operators Dataset (`operators.npz`)
**Size**: ~500 KB  
**Source**: Generated programmatically

#### Generate:
```bash
cd src/neural
python -c "
import numpy as np

def gen_operators():
    imgs, lbls = [], []
    # + operator (horizontal + vertical lines)
    for _ in range(1000):
        img = np.zeros((28, 28))
        img[12:16, :] = 1  # Horizontal line
        img[:, 12:16] = 1  # Vertical line
        img += np.random.randn(28, 28) * 0.1
        imgs.append(img); lbls.append(10)
    
    # - operator (horizontal line)
    for _ in range(1000):
        img = np.zeros((28, 28))
        img[12:16, :] = 1
        img += np.random.randn(28, 28) * 0.1
        imgs.append(img); lbls.append(11)
    
    # × operator (two diagonals)
    for _ in range(1000):
        img = np.eye(28) * 0.5 + np.fliplr(np.eye(28)) * 0.5
        img += np.random.randn(28, 28) * 0.1
        imgs.append(img); lbls.append(12)
    
    # ÷ operator (horizontal line + dots)
    for _ in range(1000):
        img = np.zeros((28, 28))
        img[12:16, :] = 1
        img[6:10, 12:16] = 1  # Top dot
        img[18:22, 12:16] = 1  # Bottom dot
        img += np.random.randn(28, 28) * 0.1
        imgs.append(img); lbls.append(13)
    
    return np.array(imgs)[:, np.newaxis, :, :], np.array(lbls)

imgs, lbls = gen_operators()
np.savez('operators.npz', images=imgs, labels=lbls)
print(f'✓ Generated {len(imgs)} operator images')
"
```

**Contents**:
- Training images: 4,000 samples (1,000 per operator)
- Labels: 10 (+), 11 (-), 12 (×), 13 (÷)

## Folder Structure After Setup

```
neurosymbolic_mvp/
└── src/
    └── neural/
        ├── mnist_data.npz          # 11 MB (download)
        ├── operators.npz           # 500 KB (generate)
        ├── trained_cnn_model.pkl   # 1.4 MB (created by training)
        ├── cnn.py
        ├── model.py
        ├── train.py
        └── mnist_loader.py
```

## Verification

After downloading/generating, verify files exist:

```bash
# Windows PowerShell
cd neurosymbolic_mvp/src/neural
Test-Path mnist_data.npz    # Should return True
Test-Path operators.npz      # Should return True

# Linux/Mac
cd neurosymbolic_mvp/src/neural
ls -lh mnist_data.npz        # Should show ~11M
ls -lh operators.npz         # Should show ~500K
```

## Class Mapping

| Class ID | Symbol | Type | Source |
|----------|--------|------|--------|
| 0-9 | 0-9 | Digit | MNIST |
| 10 | + | Operator | Generated |
| 11 | - | Operator | Generated |
| 12 | × | Operator | Generated |
| 13 | ÷ | Operator | Generated |

Total: **14 classes**

## Troubleshooting

**Issue**: "File not found: mnist_data.npz"  
**Solution**: Run the download command above from `src/neural/` folder

**Issue**: "File not found: operators.npz"  
**Solution**: Run the generation script above

**Issue**: Download fails  
**Solution**: 
1. Check internet connection
2. Try manual download from browser
3. Use alternative mirror if available

**Issue**: Generated operators look wrong  
**Solution**: Operators are simple geometric shapes + noise. This is intentional for training.

## Next Steps

After data setup:
1. Train the model: `python train.py`
2. Test the system: `python ../../test_integration.py`
3. Launch UI: `python ../ui_app.py`

## Data License

- **MNIST**: Available under Yann LeCun's license for research and education
- **Operators**: Generated data, free to use

## Notes

- Data files are excluded from git via `.gitignore`
- Each user must download/generate their own data
- Total data size: ~12 MB (reasonable for local storage)
- Data is cached after first download
