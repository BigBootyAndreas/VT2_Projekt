import pandas as pd
import numpy as np
import os
import warnings
from File_reader import read_csv_file
from Acoustic_data import acoustic_processing
from data_preparation import extract_features_from_stft, extract_features_from_imu, advanced_data_preprocessing

# Suppress specific warnings about n_quantiles
warnings.filterwarnings('ignore', message='n_quantiles .* is greater than the total number of samples')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing._data')

def safe_extract_features_from_stft(stft_result, sr):
    """
    Safe wrapper for STFT feature extraction with NaN handling
    """
    try:
        features = extract_features_from_stft(stft_result, sr)
        
        # Check for NaN values in features and replace with 0
        for key, value in features.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                print(f"Warning: Found NaN/Inf in feature {key}, replacing with 0")
                features[key] = 0
        
        return features
    except Exception as e:
        print(f"Error extracting STFT features: {e}")
        # Return default features
        return {
            'energy_low': 0, 'energy_mid': 0, 'energy_high': 0,
            'energy_total': 0, 'spectral_centroid': 0, 'spectral_spread': 0,
            'spectral_flatness': 0, 'spectral_rolloff_75': 0, 'spectral_rolloff_85': 0, 'spectral_rolloff_95': 0
        }

def safe_extract_features_from_imu(x_accel, y_accel, z_accel, sr=100):
    """
    Safe wrapper for IMU feature extraction with NaN handling
    """
    try:
        features = extract_features_from_imu(x_accel, y_accel, z_accel, sr)
        
        # Check for NaN values in features and replace with 0
        for key, value in features.items():
            if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                print(f"Warning: Found NaN/Inf in IMU feature {key}, replacing with 0")
                features[key] = 0
        
        return features
    except Exception as e:
        print(f"Error extracting IMU features: {e}")
        # Return default features for x, y, z axes
        default_features = {}
        for axis in ['x', 'y', 'z']:
            default_features.update({
                f'{axis}_mean': 0, f'{axis}_std': 0, f'{axis}_rms': 0, f'{axis}_peak': 0,
                f'{axis}_kurtosis': 0, f'{axis}_skewness': 0,
                f'{axis}_stft_energy_total': 0, f'{axis}_stft_spectral_centroid': 0,
                f'{axis}_stft_spectral_spread': 0, f'{axis}_stft_spectral_flatness': 0
            })
        return default_features

def extract_progressive_wear_features(acoustic_df, imu_df, failure_time, base_wear=None, n_windows=3, recording_duration=60):
    """
    Extract features from multiple time windows showing progression to failure
    Enhanced with proper wear level calculation that caps at 100%
    
    Parameters:
    -----------
    failure_time : float or None
        Time when tool failed (seconds), or None if no failure
    base_wear : float or None
        Base wear level for tools that don't fail (0-1), or None if failure_time is provided
    """
    features_list = []
    wear_levels = []
    
    # Debug IMU data structure
    if imu_df is not None:
        print(f"    IMU data shape: {imu_df.shape}")
        print(f"    IMU columns: {imu_df.columns.tolist()}")
        if len(imu_df) > 0:
            print(f"    IMU first few rows:\n{imu_df.head(3)}")
    else:
        print("    IMU data is None")
    
    # Split recording into time windows
    for i in range(n_windows):
        window_start_time = i * (recording_duration / n_windows)
        window_end_time = (i + 1) * (recording_duration / n_windows)
        window_center_time = (window_start_time + window_end_time) / 2
        
        print(f"    Processing window {i+1}: {window_start_time:.1f}s - {window_end_time:.1f}s")
        
        # Extract window from acoustic data
        if "Time" in acoustic_df.columns:
            mask = (acoustic_df["Time"] >= window_start_time) & (acoustic_df["Time"] < window_end_time)
            acoustic_window = acoustic_df[mask].copy()
            print(f"      Acoustic window (time-based): {len(acoustic_window)} samples")
        else:
            # If no time column, use indices
            start_idx = int(len(acoustic_df) * window_start_time / recording_duration)
            end_idx = int(len(acoustic_df) * window_end_time / recording_duration)
            acoustic_window = acoustic_df.iloc[start_idx:end_idx].copy()
            print(f"      Acoustic window (index-based): {len(acoustic_window)} samples")
        
        # Extract window from IMU data with improved logic
        imu_window = None
        if imu_df is not None and len(imu_df) > 0:
            try:
                # Method 1: Try time-based windowing if time column exists
                if "epoch" in imu_df.columns:
                    print(f"      Using epoch column for IMU windowing")
                    # Check if epoch looks like absolute timestamps (large numbers)
                    if imu_df["epoch"].iloc[0] > recording_duration * 100:
                        # Convert to relative time
                        imu_time = imu_df["epoch"] - imu_df["epoch"].iloc[0]
                        print(f"        Converted epoch to relative time (first value: {imu_time.iloc[0]})")
                    else:
                        imu_time = imu_df["epoch"].copy()
                    
                    # Apply time mask
                    mask = (imu_time >= window_start_time) & (imu_time < window_end_time)
                    imu_window = imu_df[mask].copy()
                    print(f"        IMU window (time-based): {len(imu_window)} samples")
                    
                    # If time-based windowing gives too few samples, try index-based
                    if len(imu_window) < 10:
                        print(f"        Time-based windowing gave {len(imu_window)} samples, trying index-based")
                        start_idx = int(len(imu_df) * window_start_time / recording_duration)
                        end_idx = int(len(imu_df) * window_end_time / recording_duration)
                        imu_window = imu_df.iloc[start_idx:end_idx].copy()
                        print(f"        IMU window (index-based fallback): {len(imu_window)} samples")
                        
                # Method 2: Use index-based windowing
                else:
                    print(f"      Using index-based windowing for IMU")
                    start_idx = int(len(imu_df) * window_start_time / recording_duration)
                    end_idx = int(len(imu_df) * window_end_time / recording_duration)
                    imu_window = imu_df.iloc[start_idx:end_idx].copy()
                    print(f"        IMU window: {len(imu_window)} samples (indices {start_idx}-{end_idx})")
                
                # Validate the window
                if len(imu_window) == 0:
                    print(f"        Warning: IMU window is empty, creating minimal window")
                    # Create a minimal window from nearby data
                    center_idx = int(len(imu_df) * window_center_time / recording_duration)
                    start_idx = max(0, center_idx - 5)
                    end_idx = min(len(imu_df), center_idx + 5)
                    imu_window = imu_df.iloc[start_idx:end_idx].copy()
                    print(f"        Created minimal IMU window: {len(imu_window)} samples")
                
            except Exception as e:
                print(f"        Error extracting IMU window: {e}")
                # Fallback: use a small section of data
                center_idx = int(len(imu_df) * window_center_time / recording_duration)
                start_idx = max(0, center_idx - 10)
                end_idx = min(len(imu_df), center_idx + 10)
                imu_window = imu_df.iloc[start_idx:end_idx].copy()
                print(f"        Fallback IMU window: {len(imu_window)} samples")
        else:
            print(f"      No IMU data available for windowing")
        
        # FIXED: Calculate wear level for this window with proper capping and base wear usage
        if failure_time is None:
            # No failure during recording - use base wear with slight progression
            if base_wear is not None:
                # Calculate progression from initial wear to slightly higher
                progression_factor = i / max(1, n_windows - 1)  # 0 to 1
                # Add small progression (max 10% increase from base)
                wear_level = base_wear + (0.1 * progression_factor)
                wear_level = min(wear_level, 1.0)  # Cap at 100%
            else:
                # Fallback if base_wear not provided
                wear_level = 0.1 + (0.2 * i / max(1, n_windows - 1))
                wear_level = min(wear_level, 0.3)  # Cap at 30% for no-failure cases
        elif failure_time <= 0:
            # Already failed at start
            wear_level = 1.0
        elif window_center_time < failure_time:
            # Before failure - wear increases towards failure
            progress_to_failure = window_center_time / failure_time
            # Start at 60% wear and progress to 90% just before failure
            wear_level = 0.6 + 0.3 * progress_to_failure
            wear_level = min(wear_level, 0.9)  # Cap just before failure
        else:
            # After failure - tool is 100% worn
            wear_level = 1.0
        
        # Ensure wear level is always between 0 and 1
        wear_level = max(0.0, min(1.0, wear_level))
        
        print(f"  Window {i+1}: {window_start_time:.1f}s - {window_end_time:.1f}s, Wear: {wear_level:.1%}")
        
        # Extract features if window has enough data
        if len(acoustic_window) > 100:  # Minimum data check
            # Extract acoustic features safely
            try:
                stft_result, sr = acoustic_processing(acoustic_window, show_plot=False)
                acoustic_features = safe_extract_features_from_stft(stft_result, sr)
                print(f"      ✓ Extracted {len(acoustic_features)} acoustic features")
            except Exception as e:
                print(f"    Warning: Failed to extract acoustic features for window {i+1}: {e}")
                acoustic_features = safe_extract_features_from_stft(None, 1000)  # Default features
            
            # Extract IMU features safely with improved logic
            imu_features = {}
            if imu_window is not None and len(imu_window) > 0:
                print(f"      Processing IMU window with {len(imu_window)} samples")
                try:
                    # Check for required columns
                    if "X (g)" in imu_window.columns and "Y (g)" in imu_window.columns and "Z (g)" in imu_window.columns:
                        print(f"        Found X, Y, Z acceleration columns")
                        # Clean the data first
                        x_accel = imu_window["X (g)"].values
                        y_accel = imu_window["Y (g)"].values
                        z_accel = imu_window["Z (g)"].values
                        
                        # Remove any NaN or infinite values
                        x_accel = np.nan_to_num(x_accel, nan=0.0, posinf=0.0, neginf=0.0)
                        y_accel = np.nan_to_num(y_accel, nan=0.0, posinf=0.0, neginf=0.0)
                        z_accel = np.nan_to_num(z_accel, nan=0.0, posinf=0.0, neginf=0.0)
                        
                        print(f"        Data ranges: X=[{x_accel.min():.3f}, {x_accel.max():.3f}], "
                              f"Y=[{y_accel.min():.3f}, {y_accel.max():.3f}], "
                              f"Z=[{z_accel.min():.3f}, {z_accel.max():.3f}]")
                        
                        # Extract features with additional validation
                        if len(x_accel) > 0 and len(y_accel) > 0 and len(z_accel) > 0:
                            imu_features = safe_extract_features_from_imu(x_accel, y_accel, z_accel)
                            print(f"        ✓ Extracted {len(imu_features)} IMU features")
                        else:
                            print(f"        Warning: Empty acceleration arrays")
                            imu_features = safe_extract_features_from_imu(np.array([0]), np.array([0]), np.array([0]))
                    else:
                        print(f"        Warning: Missing required columns in IMU data")
                        print(f"        Available columns: {imu_window.columns.tolist()}")
                        # Try alternative column names
                        alt_cols = {}
                        for col in imu_window.columns:
                            col_lower = col.lower()
                            if 'x' in col_lower and ('accel' in col_lower or 'acc' in col_lower):
                                alt_cols['x'] = col
                            elif 'y' in col_lower and ('accel' in col_lower or 'acc' in col_lower):
                                alt_cols['y'] = col
                            elif 'z' in col_lower and ('accel' in col_lower or 'acc' in col_lower):
                                alt_cols['z'] = col
                        
                        if len(alt_cols) == 3:
                            print(f"        Found alternative columns: {alt_cols}")
                            x_accel = np.nan_to_num(imu_window[alt_cols['x']].values, nan=0.0)
                            y_accel = np.nan_to_num(imu_window[alt_cols['y']].values, nan=0.0)
                            z_accel = np.nan_to_num(imu_window[alt_cols['z']].values, nan=0.0)
                            imu_features = safe_extract_features_from_imu(x_accel, y_accel, z_accel)
                            print(f"        ✓ Extracted {len(imu_features)} IMU features using alternative columns")
                        else:
                            print(f"        Using default IMU features")
                            imu_features = safe_extract_features_from_imu(np.array([0]), np.array([0]), np.array([0]))
                            
                except Exception as e:
                    print(f"    Warning: Failed to extract IMU features for window {i+1}: {e}")
                    import traceback
                    traceback.print_exc()
                    imu_features = safe_extract_features_from_imu(np.array([0]), np.array([0]), np.array([0]))
            else:
                # No IMU data available or window too small
                if imu_window is None:
                    print(f"      No IMU window extracted")
                else:
                    print(f"      IMU window too small: {len(imu_window)} samples")
                print(f"      Using default IMU features")
                imu_features = safe_extract_features_from_imu(np.array([0]), np.array([0]), np.array([0]))
            
            # Combine features
            window_features = {**acoustic_features, **imu_features}
            
            # Add window-specific identifier
            window_features['window_number'] = i + 1
            window_features['window_center_time'] = window_center_time
            
            # Final check: ensure no NaN values made it through
            for key, value in window_features.items():
                if isinstance(value, (int, float)) and (np.isnan(value) or np.isinf(value)):
                    print(f"    Final check: Replacing NaN/Inf in {key}")
                    window_features[key] = 0
            
            features_list.append(window_features)
            wear_levels.append(wear_level)
        else:
            print(f"    Warning: Not enough data in window {i+1} (only {len(acoustic_window)} samples)")
    
    return features_list, wear_levels

def create_enhanced_failure_map_with_times():
    """
    Enhanced failure map with detailed timing information
    """
    failure_map = {
        # Format: sample_number: (failure_time_seconds_or_None, base_wear_if_no_failure)
        1: (None, 0.7),   # No failure, moderate wear
        2: (26.0, None),  # Failed at 26 seconds 
        3: (25.0, None),  # Failed at 25 seconds
        4: (None, 0.8),   # No failure, higher wear
        5: (15.0, None),  # Failed at 15 seconds
        6: (0.0, None),   # Already broken
        7: (None, 0.2),   # Light wear
        8: (None, 0.5),   # Moderate wear
        9: (45.0, None),  # Failed at 45 seconds 
        10: (0.0, None),  # Already broken
        11: (None, 0.3),  # Light wear
        12: (None, 0.6),  # Moderate wear
        13: (50.0, None), # Failed at 50 seconds
        14: (0, None),    # Already broken
        15: (None, 0.2),  # Very light wear
        16: (None, 0.8),  # Moderate wear
        17: (20.0, None), # Failed at 20 seconds
        18: (None, 0.2),  # Light wear
        19: (None, 0.9),  # Moderate wear
        20: (5.0, None),  # Failed at 5 seconds
        21: (None, 0.3),  # Light wear
        22: (35.0, None), # Failed at 35 seconds
        23: (0, None),    # Already broken
    }
    return failure_map

def create_training_failure_map():
    """
    Training failure map for samples 1-17
    """
    training_failure_map = {
        # Samples 1-17 for training
        1: (None, 0.7),   # No failure, moderate wear
        2: (26.0, None),  # Failed at 26 seconds 
        3: (25.0, None),  # Failed at 25 seconds
        4: (None, 0.8),   # No failure, higher wear
        5: (15.0, None),  # Failed at 15 seconds
        6: (0.0, None),   # Already broken
        7: (None, 0.2),   # Light wear
        8: (None, 0.5),   # Moderate wear
        9: (45.0, None),  # Failed at 45 seconds 
        10: (0.0, None),  # Already broken
        11: (None, 0.3),  # Light wear
        12: (None, 0.6),  # Moderate wear
        13: (50.0, None), # Failed at 50 seconds
        14: (0, None),    # Already broken
        15: (None, 0.2),  # Very light wear
        16: (None, 0.8),  # Moderate wear
        17: (20.0, None), # Failed at 20 seconds
    }
    return training_failure_map

def create_test_failure_map():
    """
    Test failure map for samples 18-23
    """
    test_failure_map = {
        # Samples 18-23 for testing
        18: (None, 0.2),  # Light wear
        19: (None, 0.9),  # Moderate wear
        20: (5.0, None),  # Failed at 5 seconds
        21: (None, 0.3),  # Light wear
        22: (35.0, None), # Failed at 35 seconds
        23: (0, None),    # Already broken
    }
    return test_failure_map

def collect_progressive_wear_dataset(base_dir, failure_map, n_windows=3):
    """
    Collect dataset with progressive wear windows and enhanced error handling
    FIXED: Properly pass base wear level to extract function
    """
    # Find directories
    acoustic_dir = os.path.join(base_dir, "Acoustic Data")
    imu_dir = os.path.join(base_dir, "IMU Data")
    
    # Find all files
    acoustic_files = []
    imu_files = []
    
    for root, dirs, files in os.walk(acoustic_dir):
        acoustic_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
    
    for root, dirs, files in os.walk(imu_dir):
        imu_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
    
    acoustic_files = sorted(acoustic_files)
    imu_files = sorted(imu_files)
    
    all_features = []
    wear_states = []
    sample_info = []  # Track which sample and window each feature came from
    
    for i, acoustic_file in enumerate(acoustic_files):
        imu_file = imu_files[i] if i < len(imu_files) else None
        
        # Extract sample number
        import re
        match = re.search(r'(\d+)', os.path.basename(acoustic_file))
        sample_number = int(match.group(1)) if match else i + 1
        
        # Skip if sample not in failure map (for filtering training/test sets)
        if sample_number not in failure_map:
            continue
        
        print(f"\nProcessing Sample {sample_number}...")
        
        # Get failure information
        failure_info = failure_map.get(sample_number, (None, 0.5))
        failure_time, base_wear = failure_info
        
        # Debug failure info
        if failure_time is None:
            print(f"  No failure during recording, base wear: {base_wear:.1%}")
        elif failure_time <= 0:
            print(f"  Tool already failed at start")
        else:
            print(f"  Tool failed at {failure_time:.1f} seconds")
        
        # Read data with error handling
        try:
            acoustic_df = read_csv_file(acoustic_file, '2')
            if acoustic_df is None:
                print(f"    Error: Could not read acoustic file {acoustic_file}")
                continue
                
            # Clean acoustic data
            acoustic_df = acoustic_df.dropna()
            if len(acoustic_df) == 0:
                print(f"    Error: No valid acoustic data in {acoustic_file}")
                continue
        except Exception as e:
            print(f"    Error reading acoustic file {acoustic_file}: {e}")
            continue
        
        imu_df = None
        if imu_file:
            try:
                print(f"    Reading IMU file: {os.path.basename(imu_file)}")
                imu_df = read_csv_file(imu_file, '1')
                if imu_df is not None:
                    print(f"    IMU data loaded: {imu_df.shape} shape")
                    print(f"    IMU columns: {imu_df.columns.tolist()}")
                    
                    # Check for standard column names
                    if "X (g)" in imu_df.columns:
                        print(f"    ✓ Found standard IMU columns")
                    else:
                        print(f"    ⚠ Non-standard IMU columns, will try to detect")
                    
                    # Clean IMU data
                    initial_len = len(imu_df)
                    imu_df = imu_df.dropna()
                    final_len = len(imu_df)
                    
                    if final_len < initial_len:
                        print(f"    Cleaned IMU data: {initial_len} → {final_len} rows")
                    
                    if len(imu_df) == 0:
                        print(f"    Warning: No valid IMU data after cleaning")
                        imu_df = None
                    else:
                        print(f"    ✓ IMU data ready for processing")
                else:
                    print(f"    Error: read_csv_file returned None for IMU")
            except Exception as e:
                print(f"    Error reading IMU file {imu_file}: {e}")
                import traceback
                traceback.print_exc()
                imu_df = None
        else:
            print(f"    No IMU file provided")
        
        # Extract progressive features with error handling
        try:
            # FIXED: Pass both failure_time and base_wear to the function
            window_features, window_wear_levels = extract_progressive_wear_features(
                acoustic_df, imu_df, failure_time, base_wear, n_windows
            )
            
            # Add to dataset
            for j, (features, wear) in enumerate(zip(window_features, window_wear_levels)):
                # Final validation of features
                valid_features = {}
                for key, value in features.items():
                    if isinstance(value, (int, float)):
                        if np.isnan(value) or np.isinf(value):
                            print(f"    Replacing invalid value in {key}")
                            valid_features[key] = 0
                        else:
                            valid_features[key] = value
                    else:
                        valid_features[key] = value
                
                all_features.append(valid_features)
                wear_states.append(wear)
                sample_info.append((sample_number, j + 1))
                
        except Exception as e:
            print(f"    Error extracting features for sample {sample_number}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Convert to DataFrame with error handling
    if not all_features:
        raise ValueError("No features extracted from any samples!")
    
    try:
        X = pd.DataFrame(all_features)
        y = pd.Series(wear_states, name="tool_wear")
        
        # Final data cleaning
        print("\nFinal data cleaning...")
        
        # Check for and handle any remaining NaN values
        nan_count = X.isnull().sum().sum()
        if nan_count > 0:
            print(f"Found {nan_count} NaN values in final dataset, filling with 0")
            X = X.fillna(0)
        
        # Check for infinite values
        inf_count = np.isinf(X.select_dtypes(include=[np.number]).values).sum()
        if inf_count > 0:
            print(f"Found {inf_count} infinite values in final dataset, replacing with 0")
            X = X.replace([np.inf, -np.inf], 0)
        
        # Create info DataFrame for tracking
        info_df = pd.DataFrame(sample_info, columns=['sample_number', 'window_number'])
        
        print(f"\n=== Dataset Summary ===")
        print(f"Total windows: {len(X)}")
        print(f"Features per window: {len(X.columns)}")
        print(f"Windows per sample: {n_windows}")
        print("\nWear distribution:")
        for i, wear in enumerate(wear_states):
            sample_num, window_num = sample_info[i]
            print(f"  Sample {sample_num}, Window {window_num}: {wear:.1%}")
        
        # Apply enhanced preprocessing
        print(f"\nApplying enhanced preprocessing...")
        X_processed, scaler, feature_selector, preprocessing_info = advanced_data_preprocessing(
            X, y, remove_outliers=True, handle_correlations=True, verbose=True
        )
        
        print(f"\nPreprocessing summary:")
        for key, value in preprocessing_info.items():
            print(f"  {key}: {value}")
        
        return X_processed, y, info_df
        
    except Exception as e:
        print(f"Error creating final dataset: {e}")
        import traceback
        traceback.print_exc()
        raise

def collect_combined_training_dataset(data_dir, n_windows=4):
    """
    Collect combined training dataset from samples 1-17 (all in one directory)
    """
    print("=== Collecting Combined Training Dataset ===")
    
    # Get training failure map (samples 1-17)
    training_failure_map = create_training_failure_map()
    
    print("Processing training data (samples 1-17)...")
    X_train, y_train, info_train = collect_progressive_wear_dataset(
        data_dir, training_failure_map, n_windows
    )
    
    print(f"\nCombined training dataset summary:")
    print(f"- Total training samples: {len(X_train)} windows")
    print(f"- Training samples (1-17): {len(info_train)} windows")
    
    # Show sample distribution
    unique_samples = sorted(info_train['sample_number'].unique())
    print(f"- Sample numbers used: {unique_samples}")
    
    # Show wear distribution
    print("\nWear distribution in training data:")
    wear_bins = [(0, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for low, high in wear_bins:
        mask = (y_train >= low) & (y_train < high)
        count = mask.sum()
        print(f"  {low:.1f}-{high:.1f}: {count} windows ({count/len(y_train)*100:.1f}%)")
    
    return X_train, y_train, info_train

def collect_test_dataset(data_dir, n_windows=4):
    """
    Collect test dataset from samples 18-23 (same directory as training)
    """
    print("=== Collecting Test Dataset ===")
    
    # Get test failure map (samples 18-23)
    test_failure_map = create_test_failure_map()
    
    # Process test data (samples 18-23)
    print("Processing test drill data (samples 18-23)...")
    X_test, y_test, info_test = collect_progressive_wear_dataset(
        data_dir, test_failure_map, n_windows
    )
    
    print(f"\nTest dataset summary:")
    print(f"- Total test samples: {len(X_test)} windows")
    print(f"- Test data (18-23): {len(info_test)} windows")
    
    # Show sample distribution
    unique_samples = sorted(info_test['sample_number'].unique())
    print(f"- Sample numbers used: {unique_samples}")
    
    # Show wear distribution
    print("\nWear distribution in test data:")
    wear_bins = [(0, 0.3), (0.3, 0.7), (0.7, 0.9), (0.9, 1.0)]
    for low, high in wear_bins:
        mask = (y_test >= low) & (y_test < high)
        count = mask.sum()
        print(f"  {low:.1f}-{high:.1f}: {count} windows ({count/len(y_test)*100:.1f}%)")
    
    return X_test, y_test, info_test

if __name__ == "__main__":
    base_dir = "C:\\Users\\User\\Documents\\AAU 8. semester\\Projekt\\Data"
    failure_map = create_enhanced_failure_map_with_times()
    
    X, y, info = collect_progressive_wear_dataset(base_dir, failure_map, n_windows=3)
    
    # Save the dataset
    X.to_csv("progressive_wear_features.csv", index=False)
    y.to_csv("progressive_wear_labels.csv", index=False)
    info.to_csv("progressive_wear_info.csv", index=False)
    
    print("Dataset saved!")
