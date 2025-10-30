from typing import List, Tuple, Optional, Dict
import sys
import os
# Add symbolic module to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src', 'symbolic'))

from reasoner import solve_arithmetic

# Test cases
test_cases = [
    # (description, symbols)
    ("Valid addition", [("3", 0.95), ("+", 0.87), ("7", 0.92)]),
    ("Valid multiplication", [("4", 0.88), ("×", 0.91), ("5", 0.93)]),
    ("Division by zero", [("8", 0.90), ("÷", 0.85), ("0", 0.95)]),
    ("Low confidence", [("2", 0.45), ("+", 0.40), ("3", 0.48)]),
    ("Invalid operator", [("5", 0.90), ("&", 0.80), ("2", 0.92)]),
]

print("=" * 60)
print("SYMBOLIC REASONING TESTS")
print("=" * 60)

for description, symbols in test_cases:
    print(f"\n\nTest: {description}")
    print("-" * 60)
    result = solve_arithmetic(symbols)
    
    if result['success']:
        print(f"✓ SUCCESS: Result = {result['result']}")
    else:
        print(f"✗ FAILED: {result['explanation']}")

class SymbolicReasoner:
    """
    Main symbolic reasoning engine.
    Integrates parsing, validation, and computation.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        self.confidence_threshold = confidence_threshold
        self.engine = ArithmeticEngine()
    
    def reason(self, symbols: List[Tuple[str, float]]) -> Dict:
        """
        Main reasoning pipeline.
        
        Args:
            symbols: List of (symbol, confidence) from neural network
            
        Returns:
            Dictionary with result, explanation, and validation info
        """
        result = {
            'success': False,
            'expression': None,
            'result': None,
            'explanation': '',
            'validations': [],
            'steps': []
        }
        
        # Step 1: Parse symbols into expression
        print("\n=== SYMBOLIC REASONING ===")
        print(f"Input symbols: {symbols}")
        
        expr = parse_symbols(symbols)
        if expr is None:
            result['explanation'] = "Failed to parse symbols into valid expression"
            return result
        
        result['expression'] = str(expr)
        print(f"Parsed expression: {expr}")
        print(f"Confidence: {expr.confidence:.2%}")
        
        # Step 2: Validate expression
        print("\n--- Validation ---")
        is_valid, validation_messages = validate_expression(expr)
        result['validations'] = validation_messages
        
        for msg in validation_messages:
            print(msg)
        
        if not is_valid:
            result['explanation'] = "Expression failed validation"
            return result
        
        # Step 3: Compute result
        print("\n--- Computation ---")
        computation = self.engine.evaluate_with_steps(expr)
        
        if not computation['success']:
            result['explanation'] = computation['explanation']
            return result
        
        result['result'] = computation['result']
        result['steps'] = computation['steps']
        result['success'] = True
        
        for step in computation['steps']:
            print(step)
        
        # Step 4: Generate final explanation
        confidence_note = ""
        if expr.confidence < 0.7:
            confidence_note = f" (Low confidence: {expr.confidence:.2%})"
        
        result['explanation'] = (
            f"Recognized: {expr}\n"
            f"Validation: Passed{confidence_note}\n"
            f"Computation: {computation['explanation']}"
        )
        
        print(f"\n=== FINAL RESULT: {result['result']} ===")
        
        return result

# Convenience function
def solve_arithmetic(symbols: List[Tuple[str, float]], confidence_threshold: float = 0.5) -> Dict:
    """
    Solve arithmetic expression from neural network predictions.
    
    Args:
        symbols: List of (symbol_str, confidence) tuples
        confidence_threshold: Minimum confidence to accept predictions
        
    Returns:
        Dictionary with computation result and explanations
    """
    reasoner = SymbolicReasoner(confidence_threshold)
    return reasoner.reason(symbols)