import os
import pandas as pd
import pickle

# Import progressive wear functions
from progressive_wear_dataset import (
    collect_combined_training_dataset,
    create_test_failure_map
)
from ml_models import train_final_model, svr_pipe, svr_param_grid, gb_pipe, gb_param_grid

def main():
    print("=== Progressive Drill Wear Prediction Model Training ===")
    print("Using multiple time windows per sample with actual failure times")
    print("Training on samples 1-17, testing on samples 18-23")
    
    # Get data directory (contains all samples 1-23)
    data_dir = "C:\\Users\\User\\Documents\\AAU 8. semester\\Projekt\\Data"
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return
    
    # Create output directory
    output_dir = "models/drill_progressive"
    os.makedirs(output_dir, exist_ok=True)
    
    # Number of windows per recording
    n_windows = int(input("Enter number of time windows per recording (default 4): ") or "4")
    
    try:
        # Collect combined training data (samples 1-17)
        print(f"\nCollecting progressive training data with {n_windows} windows per recording...")
        X_train, y_train, info_train = collect_combined_training_dataset(data_dir, n_windows)
        
        print(f"\nTraining on {len(X_train)} windows from {len(info_train['sample_number'].unique())} samples")
        
        # Train both models
        print("\n=== Training SVR Model ===")
        svr_files = train_final_model(
            X_train, y_train, svr_pipe, svr_param_grid, 
            f"SVR_Drill_Progressive_{n_windows}windows", 
            output_dir=output_dir
        )
        print(f"SVR model saved to: {svr_files[0]}")
        
        print("\n=== Training Gradient Boosting Model ===")
        gb_files = train_final_model(
            X_train, y_train, gb_pipe, gb_param_grid, 
            f"GB_Drill_Progressive_{n_windows}windows", 
            output_dir=output_dir
        )
        print(f"GB model saved to: {gb_files[0]}")
        
        print(f"\n=== Training Complete! ===")
        print(f"All models saved to: {output_dir}")
        
        # Save training info and test failure map for evaluation
        info_train.to_csv(os.path.join(output_dir, "training_info.csv"), index=False)
        
        # Save test failure map for evaluation
        test_failure_map = create_test_failure_map()
        with open(os.path.join(output_dir, "test_failure_map.pkl"), "wb") as f:
            pickle.dump(test_failure_map, f)
        print("Test failure map saved for evaluation.")
        
        # Save summary
        with open(os.path.join(output_dir, "training_summary.txt"), "w") as f:
            f.write("Progressive Model Training Summary\n")
            f.write("=" * 40 + "\n")
            f.write(f"Training approach: Progressive wear with {n_windows} windows\n")
            f.write(f"Training samples: 1-17 ({len(X_train)} windows)\n")
            f.write(f"Test samples: 18-23 (held out)\n")
            f.write(f"Total features per window: {len(X_train.columns)}\n")
            f.write(f"Training samples used: {sorted(info_train['sample_number'].unique())}\n")
            f.write("\nTraining data distribution:\n")
            wear_bins = [(0, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)]
            for low, high in wear_bins:
                mask = (y_train >= low) & (y_train < high)
                count = mask.sum()
                f.write(f"  {low:.1f}-{high:.1f}: {count} windows ({count/len(y_train)*100:.1f}%)\n")
        
        print("\nNext step: Run evaluation script to test on samples 18-23")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()