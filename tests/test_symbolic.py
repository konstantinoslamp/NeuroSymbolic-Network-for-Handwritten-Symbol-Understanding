"""
Test symbolic abduction implementation
"""
import numpy as np  # <--- ADDED THIS
from src.symbolic.symbolic_interface import ArithmeticSymbolicModule

def test_symbolic_deduction():
    """Test basic deduction"""
    symbolic = ArithmeticSymbolicModule()
    
    # Valid expression
    result = symbolic.symbolic_deduction({
        'symbols': ['3', '+', '5'],
        'confidences': [0.9, 0.85, 0.92]
    })
    
    print("✅ Test 1: Valid expression")
    print(f"   Valid: {result['valid']}")
    print(f"   Result: {result['result']}")
    print(f"   Derivation: {result['derivation']}")
    assert result['valid'] == True
    assert result['result'] == 8.0
    
    # Invalid expression (division by zero)
    result = symbolic.symbolic_deduction({
        'symbols': ['5', '÷', '0'],
        'confidences': [0.9, 0.85, 0.92]
    })
    
    print("\n✅ Test 2: Division by zero")
    print(f"   Valid: {result['valid']}")
    print(f"   Contradictions: {result['contradictions']}")
    assert 'no_division_by_zero' in result['contradictions']


def test_symbolic_abduction():
    """Test abduction - KEY FEATURE from paper"""
    symbolic = ArithmeticSymbolicModule()
    
    # Current prediction: 3+6=9 (WRONG!)
    # Desired output: 8
    
    neural_probs = {
        'position_0': np.array([0.05, 0.03, 0.07, 0.85, 0.05, 0.02, 0.01, 0.01, 0.01, 0.01]),  # High confidence on '3'
        'position_1': np.array([0.80, 0.10, 0.05, 0.05]),  # '+' likely
        'position_2': np.array([0.02, 0.02, 0.03, 0.04, 0.05, 0.75, 0.05, 0.02, 0.01, 0.01])   # High confidence on '6'
    }
    
    abductions = symbolic.symbolic_abduction(
        desired_output=8.0,
        current_state={
            'symbols': ['3', '+', '6'],
            'positions': [0, 1, 2],
            'confidences': [0.85, 0.80, 0.75]
        },
        neural_probs=neural_probs
    )
    
    print("\n✅ Test 3: Abduction (find corrections)")
    print(f"   Current prediction: 3 + 6 = 9")
    print(f"   Target: 8")
    print(f"   Found {len(abductions)} possible corrections:\n")
    
    # --- ADDED MISSING PART BELOW ---
    for i, abd in enumerate(abductions[:3]):  # Show top 3
        print(f"   {i+1}. {abd['correction']} (plausibility: {abd['plausibility']:.3f})")
        print(f"      {abd['explanation']}")
        print(f"      Proof: {' '.join(abd['derivation'])}\n")
    
    # Check that we found valid corrections
    assert len(abductions) > 0
    assert any(abd['correction'] == ['3', '+', '5'] for abd in abductions)
    assert any(abd['correction'] == ['2', '+', '6'] for abd in abductions)


if __name__ == "__main__":
    print("🧪 Testing Symbolic Module\n")
    print("="*60)
    test_symbolic_deduction()
    test_symbolic_abduction()
    print("="*60)
    print("\n🎉 All tests passed!")