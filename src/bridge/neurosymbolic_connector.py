"""
Bridge between Neural CNN and Symbolic Reasoner
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'neural'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'symbolic'))

import numpy as np
from typing import List, Tuple
from model import CNN

from dataclasses import dataclass
from typing import Optional, Dict

# ============= Symbolic Reasoning Components =============

@dataclass
class Expression:
    """Represents a binary arithmetic expression"""
    left: int
    operator: str
    right: int
    confidence: float = 1.0
    
    def __str__(self):
        return f"{self.left} {self.operator} {self.right}"

def parse_symbols(symbols: List[Tuple[str, float]]) -> Optional[Expression]:
    if len(symbols) != 3:
        print(f"Error: Expected 3 symbols, got {len(symbols)}")
        return None
    
    left_sym, left_conf = symbols[0]
    op_sym, op_conf = symbols[1]
    right_sym, right_conf = symbols[2]
    
    if not (left_sym.isdigit() and right_sym.isdigit()):
        print(f"Error: Operands must be digits. Got: '{left_sym}' and '{right_sym}'")
        return None
    
    valid_operators = ['+', '-', '×', '÷', '*', '/']
    if op_sym not in valid_operators:
        print(f"Error: Invalid operator '{op_sym}'")
        return None
    
    if op_sym == '*':
        op_sym = '×'
    if op_sym == '/':
        op_sym = '÷'
    
    overall_conf = (left_conf * op_conf * right_conf) ** (1/3)
    
    return Expression(
        left=int(left_sym),
        operator=op_sym,
        right=int(right_sym),
        confidence=overall_conf
    )

def evaluate_expression(expr: Expression) -> Optional[float]:
    left, right, op = expr.left, expr.right, expr.operator
    
    try:
        if op == '+':
            return left + right
        elif op == '-':
            return left - right
        elif op == '×':
            return left * right
        elif op == '÷':
            if right == 0:
                print("Error: Division by zero")
                return None
            return left / right
        else:
            print(f"Error: Unknown operator '{op}'")
            return None
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

def validate_expression(expr: Expression) -> Tuple[bool, List[str]]:
    messages = []
    is_valid = True
    
    if expr.operator not in ['+', '-', '×', '÷']:
        messages.append(f"✗ [ERROR] Invalid operator '{expr.operator}'")
        is_valid = False
    else:
        messages.append(f"✓ [INFO] Operator is valid")
    
    if not (0 <= expr.left <= 9 and 0 <= expr.right <= 9):
        messages.append(f"✗ [ERROR] Operands must be 0-9. Got: {expr.left}, {expr.right}")
        is_valid = False
    else:
        messages.append(f"✓ [INFO] Operands in valid range")
    
    if expr.operator == '÷' and expr.right == 0:
        messages.append(f"✗ [ERROR] Division by zero: {expr.left} ÷ 0 is undefined")
        is_valid = False
    else:
        messages.append(f"✓ [INFO] Division is valid (or not division)")
    
    if expr.confidence < 0.5:
        messages.append(f"⚠ [WARNING] Low confidence: {expr.confidence:.2%}")
    else:
        messages.append(f"✓ [INFO] Confidence acceptable: {expr.confidence:.2%}")
    
    return is_valid, messages

def solve_arithmetic(symbols: List[Tuple[str, float]]) -> Dict:
    result = {
        'success': False,
        'expression': None,
        'result': None,
        'explanation': '',
        'validations': []
    }
    
    print("\n=== SYMBOLIC REASONING ===")
    print(f"Input symbols: {symbols}")
    
    expr = parse_symbols(symbols)
    if expr is None:
        result['explanation'] = "Failed to parse symbols"
        return result
    
    result['expression'] = str(expr)
    print(f"Parsed expression: {expr}")
    print(f"Confidence: {expr.confidence:.2%}")
    
    print("\n--- Validation ---")
    is_valid, messages = validate_expression(expr)
    result['validations'] = messages
    
    for msg in messages:
        print(msg)
    
    if not is_valid:
        result['explanation'] = "Expression failed validation"
        return result
    
    print("\n--- Computation ---")
    ans = evaluate_expression(expr)
    
    if ans is None:
        result['explanation'] = "Failed to compute result"
        return result
    
    result['result'] = ans
    result['success'] = True
    result['explanation'] = f"Computed {expr} = {ans}"
    
    print(f"\n=== FINAL RESULT: {ans} ===")
    
    return result

class NeurosymbolicSolver:
    """Connects CNN predictions with symbolic reasoning"""
    
    def __init__(self, model_path: str):
        """
        Initialize with trained CNN model.
        
        Args:
            model_path: Path to saved model weights (.pkl file)
        """
        print("Loading CNN model...")
        self.cnn = CNN()
        self.cnn.load_weights(model_path)
        print("✓ Model loaded successfully")
        
        # Mapping from class index to symbol
        self.class_to_symbol = {
            0: '0', 1: '1', 2: '2', 3: '3', 4: '4',
            5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
            10: '+', 11: '-', 12: '×', 13: '÷'
        }
    
    def predict_symbol(self, image: np.ndarray) -> Tuple[str, float]:
        """
        Predict symbol from image using CNN.
        
        Args:
            image: numpy array of shape (28, 28) or (1, 1, 28, 28)
            
        Returns:
            (predicted_symbol, confidence)
        """
        # Ensure correct shape: (1, 1, 28, 28)
        if image.shape == (28, 28):
            image = image[np.newaxis, np.newaxis, :, :]
        elif image.shape == (1, 28, 28):
            image = image[:, np.newaxis, :, :]
        
        # CNN forward pass
        logits = self.cnn.forward(image)
        
        # Convert to probabilities (softmax)
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / np.sum(probs, axis=1, keepdims=True)
        
        # Get prediction
        predicted_class = np.argmax(probs[0])
        confidence = float(probs[0][predicted_class])
        
        # Map to symbol
        symbol = self.class_to_symbol.get(predicted_class, '?')
        
        return symbol, confidence
    
    def solve_expression(self, images: List[np.ndarray]) -> Dict:
        """
        Solve arithmetic expression from list of images.
        
        Args:
            images: List of 3 images (digit, operator, digit)
            
        Returns:
            Dictionary with result and explanation
        """
        if len(images) != 3:
            return {
                'success': False,
                'explanation': f'Expected 3 images, got {len(images)}',
                'result': None
            }
        
        print("\n" + "="*70)
        print("NEUROSYMBOLIC PIPELINE")
        print("="*70)
        
        # Step 1: CNN predictions
        print("\n--- Neural Network Predictions ---")
        symbols_with_confidence = []
        
        for i, img in enumerate(images):
            symbol, confidence = self.predict_symbol(img)
            symbols_with_confidence.append((symbol, confidence))
            print(f"Image {i+1}: '{symbol}' (confidence: {confidence:.2%})")
        
        # Step 2: Symbolic reasoning
        print("\n--- Symbolic Reasoning ---")
        result = solve_arithmetic(symbols_with_confidence)
        
        return result