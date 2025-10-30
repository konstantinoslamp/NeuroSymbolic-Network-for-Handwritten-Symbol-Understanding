"""
Simple standalone test for symbolic reasoning - no file imports needed
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

# ============= Expression Parser =============
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

# ============= Arithmetic Engine =============
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

# ============= Rules =============
def validate_expression(expr: Expression) -> Tuple[bool, List[str]]:
    messages = []
    is_valid = True
    
    # Rule 1: Valid operator
    if expr.operator not in ['+', '-', '×', '÷']:
        messages.append(f"✗ [ERROR] Invalid operator '{expr.operator}'")
        is_valid = False
    else:
        messages.append(f"✓ [INFO] Operator is valid")
    
    # Rule 2: Operands in range
    if not (0 <= expr.left <= 9 and 0 <= expr.right <= 9):
        messages.append(f"✗ [ERROR] Operands must be 0-9. Got: {expr.left}, {expr.right}")
        is_valid = False
    else:
        messages.append(f"✓ [INFO] Operands in valid range")
    
    # Rule 3: No division by zero
    if expr.operator == '÷' and expr.right == 0:
        messages.append(f"✗ [ERROR] Division by zero: {expr.left} ÷ 0 is undefined")
        is_valid = False
    else:
        messages.append(f"✓ [INFO] Division is valid (or not division)")
    
    # Rule 4: Confidence check
    if expr.confidence < 0.5:
        messages.append(f"⚠ [WARNING] Low confidence: {expr.confidence:.2%}")
    else:
        messages.append(f"✓ [INFO] Confidence acceptable: {expr.confidence:.2%}")
    
    return is_valid, messages

# ============= Main Reasoner =============
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
    
    # Parse
    expr = parse_symbols(symbols)
    if expr is None:
        result['explanation'] = "Failed to parse symbols"
        return result
    
    result['expression'] = str(expr)
    print(f"Parsed expression: {expr}")
    print(f"Confidence: {expr.confidence:.2%}")
    
    # Validate
    print("\n--- Validation ---")
    is_valid, messages = validate_expression(expr)
    result['validations'] = messages
    
    for msg in messages:
        print(msg)
    
    if not is_valid:
        result['explanation'] = "Expression failed validation"
        return result
    
    # Compute
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

# ============= Tests =============
if __name__ == "__main__":
    test_cases = [
        ("Valid addition", [("3", 0.95), ("+", 0.87), ("7", 0.92)]),
        ("Valid multiplication", [("4", 0.88), ("×", 0.91), ("5", 0.93)]),
        ("Division by zero", [("8", 0.90), ("÷", 0.85), ("0", 0.95)]),
        ("Low confidence", [("2", 0.45), ("+", 0.40), ("3", 0.48)]),
        ("Invalid operator", [("5", 0.90), ("&", 0.80), ("2", 0.92)]),
    ]
    
    print("=" * 70)
    print("SYMBOLIC REASONING TESTS")
    print("=" * 70)
    
    for description, symbols in test_cases:
        print(f"\n\n{'='*70}")
        print(f"Test: {description}")
        print('='*70)
        result = solve_arithmetic(symbols)
        
        if result['success']:
            print(f"\n✓ SUCCESS: Result = {result['result']}")
        else:
            print(f"\n✗ FAILED: {result['explanation']}")
