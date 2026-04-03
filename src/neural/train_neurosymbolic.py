"""
Main Training Script for Neuro-symbolic Integration
"""

import sys
import os
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.expression_dataset import ExpressionDataset
from src.neural.digit_recognizer import DigitRecognizer
from src.symbolic.symbolic_interface import ArithmeticSymbolicModule
from src.integration.training_loop import NeuroSymbolicTrainer

def main():
    # 1. Configuration
    NUM_SAMPLES = 2000
    BATCH_SIZE = 32
    EPOCHS = 5
    INVALID_RATIO = 0.05 # 5% invalid data as requested
    
    print("🚀 Starting Neuro-symbolic Training...")
    print(f"Configuration: {NUM_SAMPLES} samples, {EPOCHS} epochs, {INVALID_RATIO} invalid ratio")
    
    # 2. Initialize Components
    print("\n[1/4] Initializing Dataset...")
    dataset = ExpressionDataset(num_samples=NUM_SAMPLES, split='train', invalid_ratio=INVALID_RATIO)
    
    print("\n[2/4] Initializing Neural Module (CNN)...")
    neural_module = DigitRecognizer()
    
    print("\n[3/4] Initializing Symbolic Module (Arithmetic Engine)...")
    symbolic_module = ArithmeticSymbolicModule()
    
    print("\n[4/4] Initializing Trainer...")
    # Config object for trainer
    class Config:
        use_abduction = True
        abduction_strategy = 'wmc' # Options: 'nga' (Neural-Guided) or 'wmc' (Weighted Model Counting)
    
    trainer = NeuroSymbolicTrainer(neural_module, symbolic_module, Config())
    
    # 3. Training Loop
    print(f"Starting Training Loop with strategy: {Config.abduction_strategy.upper()}...")
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        
        # Shuffle data
        indices = np.arange(len(dataset))
        np.random.shuffle(indices)
        
        epoch_loss = 0
        epoch_correct = 0
        epoch_abductions = 0
        
        # Batch processing
        num_batches = len(dataset) // BATCH_SIZE
        
        with tqdm(total=num_batches) as pbar:
            for i in range(0, len(dataset), BATCH_SIZE):
                batch_indices = indices[i:i+BATCH_SIZE]
                if len(batch_indices) == 0: break
                
                # Prepare batch
                batch_images = []
                batch_results = []
                
                for idx in batch_indices:
                    item = dataset[idx]
                    batch_images.append(item['images'])
                    batch_results.append(item['result'])
                
                batch_images = np.stack(batch_images) # (B, 3, 28, 28)
                
                # Train step
                metrics = trainer.train_step(batch_images, batch_results)
                
                # Update stats
                epoch_loss += metrics['loss']
                epoch_correct += metrics['correct']
                epoch_abductions += metrics['abductions']
                
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}", 
                    'acc': f"{metrics['correct']/len(batch_indices):.2f}"
                })
                pbar.update(1)
        
        # Epoch Summary
        avg_loss = epoch_loss / num_batches
        accuracy = epoch_correct / len(dataset)
        print(f"Epoch {epoch+1} Summary:")
        print(f"  Loss: {avg_loss:.4f}")
        print(f"  Accuracy: {accuracy*100:.2f}%")
        print(f"  Abductions used: {epoch_abductions}")
        
        # Save checkpoint
        neural_module.model.save_weights(f"src/neural/trained_cnn_epoch_{epoch+1}.pkl")

    # --- Gradient Flow Report (P1.4) ---
    trainer.gradient_monitor.snapshot_weights(neural_module.model, 'final')
    sanity = trainer.gradient_monitor.run_sanity_checks(neural_module.model)

    print("\n--- Gradient Flow Sanity Checks ---")
    for check_name, result in sanity.items():
        status = "PASS" if result['passed'] else "FAIL"
        print(f"  [{status}] {check_name}: {result['message']}")

    trainer.gradient_monitor.print_gradient_report()

    # --- Evaluation (P1.3) ---
    print("\nRunning Full Evaluation Suite...")
    test_dataset = ExpressionDataset(num_samples=500, split='test', invalid_ratio=INVALID_RATIO)

    from src.evaluation.metrics import EvaluationSuite
    eval_suite = EvaluationSuite()
    results = eval_suite.evaluate(neural_module, symbolic_module, test_dataset)
    EvaluationSuite.print_report(results)

    print("\n✅ Training Complete!")

if __name__ == "__main__":
    main()
