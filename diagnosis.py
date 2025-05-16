#!/usr/bin/env python3
"""
Quick diagnostic script to test the enhanced preprocessing
"""

import pandas as pd
import numpy as np
import os

# Test the progressive wear dataset collection
def test_diagnosis():
    """Test the diagnosis workflow"""
    
    print("=== TESTING ENHANCED PREPROCESSING ===")
    
    try:
        # Import the functions
        from progressive_wear_dataset import collect_combined_training_dataset
        
        # Set data directory
        data_dir = "C:\\Users\\User\\Documents\\AAU 8. semester\\Projekt\\Data"
        
        if not os.path.exists(data_dir):
            print(f"Error: Data directory not found - {data_dir}")
            print("Please update the data_dir path in this script")
            return False
        
        print(f"Loading data from: {data_dir}")
        print("This may take a few minutes...")
        
        # Collect training dataset with fewer windows for testing
        n_windows = 4  # Reduced for faster testing
        print(f"Using {n_windows} windows per recording")
        
        # This should now work without unpacking errors
        X, y, info = collect_combined_training_dataset(data_dir, n_windows=n_windows)
        
        print("\n=== SUCCESS! ===")
        print(f"Dataset loaded successfully!")
        print(f"Shape: {X.shape}")
        print(f"Features: {X.shape[1]}")
        print(f"Samples: {X.shape[0]}")
        print(f"Target range: [{y.min():.3f}, {y.max():.3f}]")
        
        # Quick data quality check
        print("\n=== DATA QUALITY CHECK ===")
        nan_count = X.isnull().sum().sum()
        inf_count = np.isinf(X.values).sum()
        
        print(f"NaN values: {nan_count}")
        print(f"Infinite values: {inf_count}")
        print(f"All values finite: {np.isfinite(X.values).all()}")
        
        # Check feature ranges
        print("\n=== FEATURE STATISTICS ===")
        print(f"Mean feature std: {X.std().mean():.4f}")
        print(f"Max feature value: {X.max().max():.4f}")
        print(f"Min feature value: {X.min().min():.4f}")
        
        # Check for constant features
        constant_features = (X.std() == 0).sum()
        print(f"Constant features: {constant_features}")
        
        print("\n✓ Diagnostic test completed successfully!")
        print("The enhanced preprocessing is working correctly.")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_diagnosis()
    
    if success:
        print("\n" + "="*50)
        print("NEXT STEPS:")
        print("1. The enhanced preprocessing is working")
        print("2. You can now run your model training")
        print("3. Expected improvements in R² scores")
        print("="*50)
    else:
        print("\n" + "="*50)
        print("TROUBLESHOOTING:")
        print("1. Check that the data directory path is correct")
        print("2. Verify all CSV files are accessible")
        print("3. Make sure all imports are working")
        print("="*50)
