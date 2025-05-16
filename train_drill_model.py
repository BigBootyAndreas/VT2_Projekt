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
        
        # Apply the SAME preprocessing to BOTH models
        print("\nApplying preprocessing to training data...")
        from data_preparation import advanced_data_preprocessing
        X_train_processed, scaler, feature_selector, preprocessing_info = advanced_data_preprocessing(
            X_train, y_train, remove_outliers=True, handle_correlations=True, verbose=True
        )
        
        # Save the preprocessing components for consistent test evaluation
        import joblib
        joblib.dump(scaler, os.path.join(output_dir, "training_scaler.pkl"))
        joblib.dump(feature_selector, os.path.join(output_dir, "training_feature_selector.pkl"))
        joblib.dump(list(X_train_processed.columns), os.path.join(output_dir, "training_features.pkl"))
        joblib.dump(preprocessing_info, os.path.join(output_dir, "preprocessing_info.pkl"))
        print(f"Saved preprocessing components for consistent evaluation")
        
        # Updated parameter grids to work with smaller feature count
        n_features = X_train_processed.shape[1]
        print(f"Features after preprocessing: {n_features}")
        
        # Adjust feature selection to match available features
        svr_param_grid_adjusted = {
            "feature_selection__k": [min(20, n_features), min(30, n_features), 'all'],
            "svr__kernel": ["rbf"],
            "svr__C": [1, 10, 100],
            "svr__gamma": ["scale", 0.01, 0.1],
            "svr__epsilon": [0.1, 0.2],
            "svr__degree": [3],
        }
        
        gb_param_grid_adjusted = {
            "feature_selection__k": [min(20, n_features), min(30, n_features), 'all'],
            "gb__n_estimators": [50, 100, 200],  
            "gb__learning_rate": [0.015, 0.1, 0.15],
            "gb__max_depth": [3, 4],  
            "gb__subsample": [0.8, 0.9],  
            "gb__min_samples_split": [5, 10],  
            "gb__min_samples_leaf": [3, 5],  
            "gb__max_features": ["sqrt", "log2"],  
        }
        
        # Train models with consistent data
        print("\n=== Training SVR Model ===")
        svr_files = train_final_model(
            X_train_processed, y_train, svr_pipe, svr_param_grid_adjusted, 
            f"SVR_Drill_Progressive_{n_windows}windows", 
            output_dir=output_dir,
            info_df=info_train
        )
        print(f"SVR model saved to: {svr_files[0]}")
        
        print("\n=== Training Gradient Boosting Model ===")
        gb_files = train_final_model(
            X_train_processed, y_train, gb_pipe, gb_param_grid_adjusted, 
            f"GB_Drill_Progressive_{n_windows}windows", 
            output_dir=output_dir,
            info_df=info_train
        )
        print(f"GB model saved to: {gb_files[0]}")
        
        print(f"\n=== Training Complete! ===")
        print(f"All models saved to: {output_dir}")
        
        # Save comprehensive training info
        training_summary = {
            'n_windows': n_windows,
            'n_training_samples': len(X_train),
            'n_features_original': X_train.shape[1],
            'n_features_final': X_train_processed.shape[1],
            'preprocessing_info': preprocessing_info,
            'feature_names': list(X_train_processed.columns),
            'training_samples': sorted(info_train['sample_number'].unique()),
            'target_stats': {
                'mean': y_train.mean(),
                'std': y_train.std(),
                'min': y_train.min(),
                'max': y_train.max()
            }
        }
        
        joblib.dump(training_summary, os.path.join(output_dir, "training_summary.pkl"))
        info_train.to_csv(os.path.join(output_dir, "training_info.csv"), index=False)
        
        # Save test failure map for evaluation
        test_failure_map = create_test_failure_map()
        with open(os.path.join(output_dir, "test_failure_map.pkl"), "wb") as f:
            pickle.dump(test_failure_map, f)
        
        print("\nNext step: Run evaluation script")
        print("Both models will now use exactly the same features for fair comparison.")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()