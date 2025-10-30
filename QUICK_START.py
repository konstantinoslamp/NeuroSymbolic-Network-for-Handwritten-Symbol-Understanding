"""
Quick Start Guide - Neurosymbolic MVP
======================================

This file shows you exactly what to do once training completes.
"""

# ============================================
# STEP 1: Verify Training Completed
# ============================================
# Look for this file in the project root:
#   trained_cnn_model.pkl

# ============================================
# STEP 2: Test the Integration
# ============================================
# Run this command:
#   python test_integration.py
# 
# You should see:
#   ✓ Model loaded
#   Neural predictions for 3 images
#   Symbolic reasoning output
#   Final result

# ============================================
# STEP 3: Launch the UI
# ============================================
# Run this command:
#   python src/ui_app.py
#
# Then:
#   1. Draw "3+5" on the canvas
#   2. Click "Recognize & Solve"
#   3. See result: "✓ 3 + 5 = 8"

# ============================================
# STEP 4: Test Edge Cases
# ============================================
# Try these expressions to test the symbolic reasoning:
# 
# Test Case 1: Normal addition
#   Draw: 2+3
#   Expected: ✓ 2 + 3 = 5
#
# Test Case 2: Multiplication
#   Draw: 4×6  (use × or x)
#   Expected: ✓ 4 × 6 = 24
#
# Test Case 3: Division by zero (should be rejected!)
#   Draw: 8÷0
#   Expected: ✗ Division by zero error
#
# Test Case 4: Subtraction
#   Draw: 9-4
#   Expected: ✓ 9 - 4 = 5

# ============================================
# Understanding the Output
# ============================================
"""
When you click "Recognize & Solve", you'll see in the terminal:

======================================================================
NEUROSYMBOLIC PIPELINE
======================================================================

--- Neural Network Predictions ---
Image 1: '3' (confidence: 98.5%)
Image 2: '+' (confidence: 95.2%)
Image 3: '5' (confidence: 97.8%)

--- Symbolic Reasoning ---

=== SYMBOLIC REASONING ===
Input symbols: [('3', 0.985), ('+', 0.952), ('5', 0.978)]
Parsed expression: 3 + 5
Confidence: 97.16%

--- Validation ---
✓ [INFO] Operator is valid
✓ [INFO] Operands in valid range
✓ [INFO] Division is valid (or not division)
✓ [INFO] Confidence acceptable: 97.16%

--- Computation ---

=== FINAL RESULT: 8.0 ===
"""

# ============================================
# Troubleshooting
# ============================================

# Problem: "Model not found"
# Solution: Training hasn't finished yet. Wait for trained_cnn_model.pkl

# Problem: "Found 2 symbols" or "Found 4 symbols"
# Solution: Space out your characters more when drawing

# Problem: Wrong symbol recognized
# Solution: 
#   - Draw larger and clearer
#   - Use standard forms (× not *, ÷ not /)
#   - Check if more training is needed

# Problem: "Low confidence" warning
# Solution: Redraw more clearly, or this is normal for difficult handwriting

# ============================================
# System Architecture Recap
# ============================================
"""
Drawing → Segmentation → CNN (Neural) → Symbolic Reasoning → Result

1. You draw "3+5" on canvas
2. Segmentation splits into 3 images
3. CNN predicts each symbol with confidence
4. Symbolic parser creates Expression(3, '+', 5)
5. Validator checks for errors
6. Arithmetic engine computes: 8
7. Result displayed in UI
"""

# ============================================
# Next Steps for Extension
# ============================================
"""
Want to extend this system? Try:

1. Multi-digit numbers (e.g., "12+34")
   - Need better segmentation
   - Need to group digits together

2. More operators (^, %, etc.)
   - Add to dataset generation
   - Add to symbolic engine

3. Parentheses and order of operations
   - Parse full expression trees
   - Implement precedence rules

4. Better UI
   - Show confidence per symbol
   - Highlight which symbol was wrong
   - Undo/redo functionality

5. More robust training
   - Data augmentation
   - More training epochs
   - Validation set monitoring
"""

if __name__ == "__main__":
    print(__doc__)
    print("\nReady to test! Follow the steps above. 🚀")
