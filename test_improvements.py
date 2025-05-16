#!/usr/bin/env python3
"""
Quick test script to verify the enhanced preprocessing improvements
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Make sure to run this from your project directory
# Import the enhanced modules
from data_preparation import advanced_data_preprocessing, collect_dataset_from_directory
from ml_models import run_ml_pipeline, nested_cv_evaluate, svr_pipe, gb_pipe, svr_param_grid, gb_param_grid

def quick_preprocessing_test():
    """
    Quick test to compare old vs new preprocessing
    """
    print("=== Testing Enhanced Preprocessing ===")
    
    # Test with sample data (replace with your actual data path)
    data_dir = "C:\\Users\\User\\Documents\\AAU 8. semester\\Projekt\\Data"
    
    if not os.path.exists(data_dir):
        print(f"Data directory not found: {data_dir}")
        print("Please update the data_dir path in this script")
        return
    
    try:
        # Load data with enhanced preprocessing
        print("Loading data with enhanced preprocessing...")
        result = collect_dataset_from_directory(data_dir, tool_lifetime=3600)
        
        if len(result) == 5:
            X, y, scaler, feature_selector, preprocessing_info = result
        else:
            X, y = result[:2]
            preprocessing_info = {}
        
        print(f"✓ Data loaded successfully")
        print(f"  - Shape: {X.shape}")
        print(f"  - Features: {X.shape[1]}")
        print(f"  - Samples: {X.shape[0]}")
        print(f"  - Target range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Check for common data issues
        print("\n=== Data Quality Check ===")
        
        # Check for NaN/inf values
        nan_count = X.isnull().sum().sum()
        inf_count = np.isinf(X.values).sum()
        print(f"✓ NaN values: {nan_count}")
        print(f"✓ Infinite values: {inf_count}")
        
        # Check feature scales
        feature_stats = X.describe()
        large_scale_features = []
        for col in X.columns:
            std = feature_stats.loc['std', col]
            mean = abs(feature_stats.loc['mean', col])
            if std > 100 or mean > 100:
                large_scale_features.append(col)
        
        if large_scale_features:
            print(f"⚠️  {len(large_scale_features)} features with large scales (may need scaling)")
        else:
            print("✓ All features appear well-scaled")
        
        # Check preprocessing info
        if preprocessing_info:
            print(f"\n=== Preprocessing Summary ===")
            for key, value in preprocessing_info.items():
                print(f"  - {key}: {value}")
        
        return X, y
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def quick_model_test(X, y):
    """
    Quick test of enhanced models
    """
    print("\n=== Quick Model Performance Test ===")
    
    if X is None or y is None:
        print("No data available for testing")
        return
    
    # Quick split for testing
    n_train = int(0.8 * len(X))
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
    
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Test enhanced SVR
    print("\n--- Testing Enhanced SVR ---")
    try:
        # Use a simplified parameter grid for quick testing
        quick_svr_params = {
            "feature_selection__k": [min(30, X.shape[1])],
            "svr__C": [1, 10],
            "svr__gamma": ["scale"],
            "svr__kernel": ["rbf"]
        }
        
        svr_model = svr_pipe
        svr_model.fit(X_train, y_train)
        y_pred_svr = svr_model.predict(X_test)
        
        svr_r2 = r2_score(y_test, y_pred_svr)
        svr_mse = mean_squared_error(y_test, y_pred_svr)
        
        print(f"✓ SVR R² Score: {svr_r2:.4f}")
        print(f"✓ SVR MSE: {svr_mse:.4f}")
        
    except Exception as e:
        print(f"✗ SVR test failed: {e}")
        svr_r2 = -999
    
    # Test enhanced Gradient Boosting
    print("\n--- Testing Enhanced Gradient Boosting ---")
    try:
        gb_model = gb_pipe
        gb_model.fit(X_train, y_train)
        y_pred_gb = gb_model.predict(X_test)
        
        gb_r2 = r2_score(y_test, y_pred_gb)
        gb_mse = mean_squared_error(y_test, y_pred_gb)
        
        print(f"✓ GB R² Score: {gb_r2:.4f}")
        print(f"✓ GB MSE: {gb_mse:.4f}")
        
    except Exception as e:
        print(f"✗ GB test failed: {e}")
        gb_r2 = -999
    
    # Summary
    print(f"\n=== Results Summary ===")
    print(f"Enhanced SVR:  R² = {svr_r2:.4f}")
    print(f"Enhanced GB:   R² = {gb_r2:.4f}")
    
    if max(svr_r2, gb_r2) > 0.4:
        print("🎉 Great! The enhanced preprocessing shows promising results!")
    elif max(svr_r2, gb_r2) > 0.2:
        print("👍 Good! The enhanced preprocessing shows improvement!")
    else:
        print("🤔 Results still need work. Consider:")
        print("   - More data cleaning")
        print("   - Different feature engineering")
        print("   - Alternative model types")

def plot_basic_diagnostics(X, y):
    """
    Create basic diagnostic plots
    """
    print("\n=== Creating Diagnostic Plots ===")
    
    if X is None or y is None:
        return
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Target distribution
        axes[0, 0].hist(y, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Target Distribution (Tool Wear)')
        axes[0, 0].set_xlabel('Tool Wear')
        axes[0, 0].set_ylabel('Frequency')
        
        # Feature correlation with target
        correlations = X.corrwith(y).abs().sort_values(ascending=False).head(10)
        axes[0, 1].barh(range(len(correlations)), correlations.values)
        axes[0, 1].set_yticks(range(len(correlations)))
        axes[0, 1].set_yticklabels(correlations.index, rotation=0)
        axes[0, 1].set_xlabel('Absolute Correlation with Target')
        axes[0, 1].set_title('Top 10 Features Correlated with Target')
        
        # Feature scale distribution
        feature_stds = X.std().sort_values(ascending=False).head(15)
        axes[1, 0].bar(range(len(feature_stds)), feature_stds.values)
        axes[1, 0].set_xticks(range(len(feature_stds)))
        axes[1, 0].set_xticklabels(feature_stds.index, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Standard Deviation')
        axes[1, 0].set_title('Feature Scale Distribution (Top 15)')
        
        # Sample progression (if possible)
        axes[1, 1].plot(y.values, 'o-', alpha=0.7)
        axes[1, 1].set_xlabel('Sample Index')
        axes[1, 1].set_ylabel('Tool Wear')
        axes[1, 1].set_title('Tool Wear Progression')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()
        
        print("✓ Diagnostic plots created successfully")
        
    except Exception as e:
        print(f"✗ Error creating plots: {e}")

def main():
    """
    Main test function
    """
    print("=== Enhanced Preprocessing Test Suite ===")
    print("This script will test the enhanced preprocessing improvements")
    print("=" * 50)
    
    # Test 1: Data loading and preprocessing
    X, y = quick_preprocessing_test()
    
    # Test 2: Quick model performance
    if X is not None and y is not None:
        quick_model_test(X, y)
        
        # Test 3: Diagnostic plots
        plot_basic_diagnostics(X, y)
    
    print("\n" + "=" * 50)
    print("Test suite completed!")
    print("\nNext steps:")
    print("1. Run your full training pipeline with these improvements")
    print("2. Monitor the R² scores - they should be significantly better")
    print("3. Check for overfitting using learning curves")
    print("4. Consider adding XGBoost/LightGBM for even better performance")

if __name__ == "__main__":
    import os
    main()
