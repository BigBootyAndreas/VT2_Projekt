#!/usr/bin/env python3
"""
Warning suppression script for sklearn quantile transformer warnings
Run this before running your main workflow
"""

import warnings
import os

def suppress_sklearn_warnings():
    """Suppress all sklearn quantile transformer warnings"""
    
    # Suppress the specific n_quantiles warning
    warnings.filterwarnings('ignore', 
                          message='n_quantiles .* is greater than the total number of samples',
                          category=UserWarning)
    
    # Suppress all warnings from sklearn preprocessing data module
    warnings.filterwarnings('ignore', 
                          category=UserWarning, 
                          module='sklearn.preprocessing._data')
    
    # Also suppress the general warning about quantiles
    warnings.filterwarnings('ignore', 
                          message='.*n_quantiles.*',
                          category=UserWarning)
    
    print("✓ Sklearn quantile transformer warnings suppressed")

def create_warnings_config():
    """Create a warnings configuration file"""
    
    config_content = '''
# Add these lines to the top of your main script to suppress warnings:

import warnings

# Suppress sklearn quantile transformer warnings
warnings.filterwarnings('ignore', message='n_quantiles .* is greater than the total number of samples')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing._data')

print("✓ Warnings suppressed")
'''
    
    with open('suppress_warnings.py', 'w') as f:
        f.write(config_content)
    
    print("✓ Created suppress_warnings.py")
    print("  You can import this at the top of your scripts")

if __name__ == "__main__":
    print("=== Sklearn Warning Suppression Tool ===")
    
    # Suppress warnings in this session
    suppress_sklearn_warnings()
    
    # Create config file for future use
    create_warnings_config()
    
    print("\nOptions to permanently suppress warnings:")
    print("1. Import suppress_warnings.py at the top of your scripts")
    print("2. The warnings are already suppressed in the updated files")
    print("3. Add these lines to your main script:")
    print("   import warnings")
    print("   warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing._data')")
    
    print("\nNote: The warnings don't affect functionality - they're just informational")
    print("Your models will still train correctly with the automatic adjustments.")
