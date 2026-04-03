"""
Task definition for neuro-symbolic arithmetic solver
"""

from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class TaskSpecification:
    """Complete specification of the learning task"""
    
    # Input specification
    input_type: str = "image"
    input_shape: Tuple[int, ...] = (28, 28)
    input_channels: int = 1
    
    # Symbol vocabulary
    digits: List[str] = None
    operators: List[str] = None
    
    # Expression constraints
    min_expression_length: int = 3
    max_expression_length: int = 3  # Will increase in Phase 8
    
    # Output specification
    output_type: str = "float"  # Result of arithmetic expression
    
    # Neural architecture
    neural_components: List[str] = None
    
    # Symbolic rules
    symbolic_rules: List[str] = None
    
    # Training configuration
    use_abduction: bool = True
    use_all_abductions: bool = True  # Paper's key contribution
    abduction_weight: float = 0.5
    
    def __post_init__(self):
        if self.digits is None:
            self.digits = [str(i) for i in range(10)]
        if self.operators is None:
            self.operators = ['+', '-', '×', '÷']
        if self.neural_components is None:
            self.neural_components = ['DigitRecognizer', 'OperatorRecognizer']
        if self.symbolic_rules is None:
            self.symbolic_rules = [
                'valid_syntax',      # digit op digit
                'arithmetic_eval',   # compute result
                'no_division_by_zero'
            ]
    
    def validate(self) -> bool:
        """Check if specification is consistent"""
        assert self.min_expression_length <= self.max_expression_length
        assert len(self.digits) > 0
        assert len(self.operators) > 0
        assert 0.0 <= self.abduction_weight <= 1.0
        return True

# Global task instance
TASK = TaskSpecification()
TASK.validate()