import os
import sys
import subprocess
import pandas as pd  
import numpy as np

# Import necessary functions
from progressive_wear_dataset import (
    collect_combined_training_dataset,
    create_test_failure_map
)
from ml_models import train_final_model, svr_pipe, svr_param_grid, gb_pipe, gb_param_grid
from tool_wear_predictor import evaluate_progressive_model
import pickle

def run_analysis_script(data_dir):
    """
    Optional: Run failure point analysis
    """
    print("=== Step 1: Analyzing Failure Points (Optional) ===")
    run_analysis = input("Do you want to analyze/review failure points? (y/n): ")
    
    if run_analysis.lower() == 'y':
        print("Creating analysis plots...")
        from analyze_failure_points import create_failure_analysis_plots
        create_failure_analysis_plots(data_dir, output_dir="failure_analysis_plots")
        print("Plots created in 'failure_analysis_plots' directory")
        print("Review the plots and update failure times in progressive_wear_dataset.py if needed")
        input("Press Enter when ready to continue...")
    else:
        print("Skipping failure point analysis")

def train_progressive_model(data_dir, n_windows):
    """
    Step 2: Train the progressive model
    """
    print("=== Step 2: Training Progressive Model ===")
    print(f"Training with {n_windows} windows per recording")
    print("Training on samples 1-17, testing on samples 18-23")
    
    # Create output directory
    output_dir = "models/drill_progressive"
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Collect combined training data (samples 1-17)
        print(f"\nCollecting progressive training data...")
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
        
        print("Training info and test failure map saved for evaluation.")
        return True
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False

def evaluate_model(data_dir, n_windows):
    """
    Step 3: Evaluate the progressive model
    """
    print("=== Step 3: Evaluating Progressive Model ===")
    print(f"Evaluating with {n_windows} windows (matching training)")
    
    # Create a custom evaluate function that doesn't ask for input
    from progressive_wear_dataset import collect_test_dataset
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import matplotlib.pyplot as plt
    import numpy as np
    import joblib
    
    model_dir = "models/drill_progressive"
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        return False
    
    # Collect test data (automatically filters for samples 18-23)
    print("\nCollecting test data...")
    X_test, y_test, info_test = collect_test_dataset(data_dir, n_windows)
    
    print(f"Test dataset: {len(X_test)} windows from samples 18-23")
    
    # Find trained models
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
    
    if not model_files:
        print("No trained models found")
        return False
    
    print("\nEvaluating models...")
    results = {}
    
    for model_file in model_files:
        base_name = model_file.replace('_model.pkl', '')
        print(f"\nEvaluating {base_name}...")
        
        # Load model components
        model_path = os.path.join(model_dir, f"{base_name}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{base_name}_scaler.pkl")
        features_path = os.path.join(model_dir, f"{base_name}_features.pkl")
        
        try:
            model = joblib.load(model_path)
            scaler = joblib.load(scaler_path)
            feature_cols = joblib.load(features_path)
            
            # Handle missing features
            X_test_model = X_test.copy()
            missing_features = [f for f in feature_cols if f not in X_test_model.columns]
            for feature in missing_features:
                X_test_model[feature] = 0
                if missing_features:  # Only print once if there are missing features
                    print(f"  Warning: Missing feature {feature}, filled with 0")
            
            # Scale and predict
            X_test_scaled = scaler.transform(X_test_model[feature_cols])
            y_pred = model.predict(X_test_scaled)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[base_name] = {
                'predictions': y_pred,
                'MSE': mse,
                'MAE': mae,
                'R2': r2,
                'RMSE': np.sqrt(mse)
            }
            
            print(f"  MSE: {mse:.4f}")
            print(f"  MAE: {mae:.4f}")
            print(f"  R²: {r2:.4f}")
            
        except Exception as e:
            print(f"  Error loading {base_name}: {e}")
            continue
    
    if not results:
        print("No models could be evaluated")
        return False
    
    # Display comparison
    print(f"\n{'='*70}")
    print("EVALUATION RESULTS ON TEST DATA (Samples 18-23)")
    print(f"{'='*70}")
    print(f"{'Model':<35} {'MSE':<10} {'MAE':<10} {'R²':<10} {'RMSE':<10}")
    print(f"{'-'*70}")
    
    best_model = None
    best_r2 = -float('inf')
    
    for model_name, metrics in results.items():
        print(f"{model_name:<35} {metrics['MSE']:<10.4f} {metrics['MAE']:<10.4f} {metrics['R2']:<10.4f} {metrics['RMSE']:<10.4f}")
        if metrics['R2'] > best_r2:
            best_r2 = metrics['R2']
            best_model = model_name
    
    print(f"\nBest model: {best_model} (R² = {best_r2:.4f})")
    
    # Plot results for best model
    if best_model:
        print(f"\nCreating evaluation plots for {best_model}...")
        y_pred_best = results[best_model]['predictions']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Evaluation Results: {best_model}', fontsize=16)
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred_best, alpha=0.7)
        axes[0, 0].plot([0, 1], [0, 1], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Wear')
        axes[0, 0].set_ylabel('Predicted Wear')
        axes[0, 0].set_title('Actual vs Predicted Wear')
        axes[0, 0].grid(True)
        
        # 2. Residuals
        residuals = y_test - y_pred_best
        axes[0, 1].scatter(y_pred_best, residuals, alpha=0.7)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted Wear')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residuals Plot')
        axes[0, 1].grid(True)
        
        # 3. Wear progression by sample
        sample_numbers = info_test['sample_number'].values
        unique_samples = sorted(info_test['sample_number'].unique())
        
        axes[1, 0].set_title('Wear Progression by Sample')
        axes[1, 0].set_xlabel('Window Number')
        axes[1, 0].set_ylabel('Wear Level')
        
        for sample in unique_samples:
            mask = sample_numbers == sample
            sample_actual = y_test.values[mask]
            sample_pred = y_pred_best[mask]
            windows = range(1, len(sample_actual) + 1)
            
            axes[1, 0].plot(windows, sample_actual, 'o-', label=f'Sample {sample} (Actual)', alpha=0.8)
            axes[1, 0].plot(windows, sample_pred, 's--', label=f'Sample {sample} (Pred)', alpha=0.8)
        
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 0].grid(True)
        
        # 4. Error distribution
        axes[1, 1].hist(residuals, bins=15, alpha=0.7, edgecolor='black')
        axes[1, 1].set_xlabel('Prediction Error')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Error Distribution')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # Save detailed results
        results_df = pd.DataFrame({
            'Sample': info_test['sample_number'],
            'Window': info_test['window_number'],
            'Actual_Wear': y_test,
            'Predicted_Wear': y_pred_best,
            'Error': residuals,
            'Abs_Error': np.abs(residuals)
        })
        
        results_file = os.path.join(model_dir, 'test_evaluation_results.csv')
        results_df.to_csv(results_file, index=False)
        print(f"\nDetailed results saved to: {results_file}")
    
    return True

def main():
    print("=== Master Tool Wear Prediction Workflow ===")
    print("This script will run the complete workflow from training to evaluation")
    
    # Get data directory
    data_dir = "C:\\Users\\User\\Documents\\AAU 8. semester\\Projekt\\Data"
    
    if not os.path.exists(data_dir):
        print(f"Directory not found: {data_dir}")
        return
    
    # Get number of windows (used throughout the workflow)
    n_windows = int(input("Enter number of time windows per recording (default 4): ") or "4")
    
    print(f"\n{'='*60}")
    print("WORKFLOW CONFIGURATION")
    print(f"{'='*60}")
    print(f"Data directory: {data_dir}")
    print(f"Number of windows: {n_windows}")
    print(f"Training samples: 1-17")
    print(f"Test samples: 18-23")
    print(f"{'='*60}\n")
    
    # Step 1: Optional failure point analysis
    run_analysis_script(data_dir)
    
    # Step 2: Train model
    print(f"\n{'='*60}")
    success = train_progressive_model(data_dir, n_windows)
    
    if not success:
        print("Training failed. Stopping workflow.")
        return
    
    # Step 3: Evaluate model
    print(f"\n{'='*60}")
    success = evaluate_model(data_dir, n_windows)
    
    if success:
        print(f"\n{'='*60}")
        print("WORKFLOW COMPLETE!")
        print(f"{'='*60}")
        print(" Model trained on samples 1-17")
        print(" Model evaluated on samples 18-23")
        print(" Results and plots generated")
        print(" Detailed results saved to CSV")
        print("\nCheck the 'models/drill_progressive' directory for all output files.")
    else:
        print("Evaluation failed.")

if __name__ == "__main__":
    main()