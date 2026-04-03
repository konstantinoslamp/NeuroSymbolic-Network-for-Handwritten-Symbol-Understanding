"""
Expression Dataset Generator
Combines MNIST digits and YOUR handwritten operators into arithmetic expressions.
Now includes INVALID SYNTAX examples to train the symbolic logic!
"""

import numpy as np
import os
import urllib.request
import glob
from PIL import Image

class ExpressionDataset:
    def __init__(self, num_samples: int = 1000, split: str = 'train',
                 invalid_ratio: float = 0.05, expression_length: int = 3):
        """
        Args:
            num_samples: number of expressions to generate
            split: 'train' or 'test'
            invalid_ratio: fraction of invalid syntax examples
            expression_length: number of symbols per expression (3, 5, or 7 for MATH(n))
        """
        self.num_samples = num_samples
        self.split = split
        self.invalid_ratio = invalid_ratio
        self.expression_length = expression_length
        self.data_dir = 'src/data'
        self.symbols_dir = os.path.join(self.data_dir, 'symbols')
        
        # Ensure data directory exists
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Load data
        print(f"🔄 Loading datasets for {split}...")
        self.digits_data = self._load_mnist()
        self.operators_data = self._load_handwritten_operators()
        
        # Cache generated data
        self.data = []
        self.labels = []
        self.expressions = []
        
        print(f"✨ Generating {num_samples} expressions ({int(invalid_ratio*100)}% invalid)...")
        self._generate_dataset()

    def _load_mnist(self):
        """Downloads and loads MNIST dataset"""
        path = os.path.join(self.data_dir, 'mnist.npz')
        
        if not os.path.exists(path):
            print("⬇️ Downloading MNIST dataset...")
            url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz"
            urllib.request.urlretrieve(url, path)
            
        with np.load(path) as f:
            if self.split == 'train':
                x, y = f['x_train'], f['y_train']
            else:
                x, y = f['x_test'], f['y_test']
        
        # Normalize to 0-1 range
        x = x.astype(np.float32) / 255.0
        return {'x': x, 'y': y}

    def _load_handwritten_operators(self):
        """Loads ONLY your images from folders"""
        ops_map = {
            '+': 'plus',
            '-': 'minus',
            '×': 'times',
            '÷': 'divide'
        }
        
        data = {}
        
        for op_char, folder_name in ops_map.items():
            folder_path = os.path.join(self.symbols_dir, folder_name)
            
            # Find all images
            files = glob.glob(os.path.join(folder_path, "*.*"))
            valid_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            
            if not valid_files:
                # Fallback to synthetic if empty (prevents crash)
                print(f"⚠️ No images in {folder_name}, using synthetic fallback.")
                data[op_char] = np.zeros((10, 28, 28), dtype=np.float32) 
            else:
                loaded_imgs = []
                for f in valid_files:
                    img = Image.open(f).convert('L')
                    img = img.resize((28, 28))
                    img_arr = np.array(img).astype(np.float32) / 255.0
                    loaded_imgs.append(img_arr)
                
                print(f"✅ Loaded {len(loaded_imgs)} images for '{op_char}'")
                data[op_char] = np.array(loaded_imgs)
                
        return data

    def _get_random_digit(self):
        idx = np.random.randint(len(self.digits_data['x']))
        return self.digits_data['x'][idx], str(self.digits_data['y'][idx])

    def _get_random_op(self):
        op_sym = np.random.choice(['+', '-', '×', '÷'])
        op_imgs = self.operators_data[op_sym]
        op_img = op_imgs[np.random.randint(len(op_imgs))]
        return op_img, op_sym

    def _generate_dataset(self):
        """Generates expressions, including some invalid ones.
        Supports variable-length expressions for MATH(n)."""

        if self.expression_length == 3:
            self._generate_length3()
        else:
            self._generate_variable_length()

    def _generate_length3(self):
        """Original length-3 generation (MATH(3))."""
        for _ in range(self.num_samples):
            is_invalid = np.random.random() < self.invalid_ratio

            if not is_invalid:
                img1, val1 = self._get_random_digit()
                img2, val2 = self._get_random_op()
                img3, val3 = self._get_random_digit()

                if val2 == '÷' and val3 == '0':
                    val3 = str(np.random.randint(1, 10))
                    indices = np.where(self.digits_data['y'] == int(val3))[0]
                    img3 = self.digits_data['x'][indices[np.random.randint(len(indices))]]

                expression_imgs = np.stack([img1, img2, img3])
                res = self._evaluate(float(val1), val2, float(val3))
                text = f"{val1}{val2}{val3}"
            else:
                scenario = np.random.choice(['DDO', 'ODD', 'OOO', 'DDD'])
                comps = []
                vals = []
                for char_type in scenario:
                    if char_type == 'D':
                        img, val = self._get_random_digit()
                    else:
                        img, val = self._get_random_op()
                    comps.append(img)
                    vals.append(val)

                expression_imgs = np.stack(comps)
                res = None
                text = "".join(vals)

            self.data.append(expression_imgs)
            self.labels.append(res)
            self.expressions.append(text)

    def _generate_variable_length(self):
        """Generate variable-length expressions for MATH(5), MATH(7), etc."""
        from src.symbolic.expression_parser import ExpressionParser
        parser = ExpressionParser()

        n = self.expression_length
        num_digits = (n + 1) // 2
        num_ops = n // 2

        for _ in range(self.num_samples):
            is_invalid = np.random.random() < self.invalid_ratio

            if not is_invalid:
                # Generate valid expression with proper precedence evaluation
                for attempt in range(100):
                    imgs = []
                    vals = []

                    for i in range(num_digits):
                        img, val = self._get_random_digit()
                        imgs.append(img)
                        vals.append(val)
                        if i < num_ops:
                            img_op, val_op = self._get_random_op()
                            imgs.append(img_op)
                            vals.append(val_op)

                    # Check for division by zero and evaluate with precedence
                    symbols = vals
                    result = parser.evaluate(symbols)

                    if result is not None and abs(result) < 1e6:
                        expression_imgs = np.stack(imgs)
                        text = symbols  # Store as list for variable-length
                        self.data.append(expression_imgs)
                        self.labels.append(result)
                        self.expressions.append(text)
                        break
                else:
                    # Fallback: simple addition chain
                    imgs = []
                    vals = []
                    total = 0
                    for i in range(num_digits):
                        d = np.random.randint(1, 5)
                        total += d
                        idx = np.where(self.digits_data['y'] == d)[0]
                        img = self.digits_data['x'][idx[np.random.randint(len(idx))]]
                        imgs.append(img)
                        vals.append(str(d))
                        if i < num_ops:
                            img_op = self.operators_data['+'][
                                np.random.randint(len(self.operators_data['+']))]
                            imgs.append(img_op)
                            vals.append('+')

                    expression_imgs = np.stack(imgs)
                    self.data.append(expression_imgs)
                    self.labels.append(float(total))
                    self.expressions.append(vals)
            else:
                # Invalid: random mix of digits/operators in wrong structure
                comps = []
                vals = []
                pattern = list('D' * n)  # Start with all digits
                # Randomly place operators in wrong positions
                wrong_positions = np.random.choice(range(0, n, 2),
                    size=min(2, num_digits), replace=False)
                for pos in wrong_positions:
                    pattern[pos] = 'O'

                for char_type in pattern:
                    if char_type == 'D':
                        img, val = self._get_random_digit()
                    else:
                        img, val = self._get_random_op()
                    comps.append(img)
                    vals.append(val)

                expression_imgs = np.stack(comps)
                self.data.append(expression_imgs)
                self.labels.append(None)
                self.expressions.append(vals)
            
    def _evaluate(self, d1, op, d2):
        if op == '+': return float(d1 + d2)
        if op == '-': return float(d1 - d2)
        if op == '×': return float(d1 * d2)
        if op == '÷': return float(d1 / d2)
        return 0.0

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            'images': self.data[idx],
            'result': self.labels[idx], # Can be None now!
            'text': self.expressions[idx]
        }