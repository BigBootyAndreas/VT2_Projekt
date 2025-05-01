import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler
import os
import sys
import glob
import librosa

# Import the existing processing functions
# Add the current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the existing signal processing functions
from Acoustic_data import acoustic_processing  
from IMU_data import imu_processing

def extract_features_from_stft(stft_result, sr):
    """
    Extract features from STFT result
    Uses the output from the existing acoustic_processing function
    
    Parameters:
    -----------
    stft_result : numpy.ndarray
        STFT result from acoustic_processing
    sr : int
        Sampling rate
        
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    # Convert to magnitude
    magnitude = np.abs(stft_result)
    
    # Calculate frequency bins
    n_fft = 2 * (magnitude.shape[0] - 1)
    freq_bins = np.fft.rfftfreq(n_fft, 1/sr)
    
    # Extract features
    features = {}
    
    # Mean magnitude across time for each frequency
    mean_spectrum = np.mean(magnitude, axis=1)
    
    # Energy in different frequency bands
    # Low (0-500 Hz), Mid (500-2000 Hz), High (2000+ Hz)
    low_idx = np.where(freq_bins <= 500)[0]
    mid_idx = np.where((freq_bins > 500) & (freq_bins <= 2000))[0]
    high_idx = np.where(freq_bins > 2000)[0]
    
    features['energy_low'] = np.sum(mean_spectrum[low_idx]) if len(low_idx) > 0 else 0
    features['energy_mid'] = np.sum(mean_spectrum[mid_idx]) if len(mid_idx) > 0 else 0
    features['energy_high'] = np.sum(mean_spectrum[high_idx]) if len(high_idx) > 0 else 0
    
    # Total energy
    features['energy_total'] = np.sum(mean_spectrum)
    
    # Spectral centroid (weighted mean of frequencies)
    if len(freq_bins) == len(mean_spectrum):
        features['spectral_centroid'] = np.sum(freq_bins * mean_spectrum) / np.sum(mean_spectrum) if np.sum(mean_spectrum) > 0 else 0
    
    # Spectral spread (weighted std of frequencies)
    if len(freq_bins) == len(mean_spectrum) and features['spectral_centroid'] > 0:
        features['spectral_spread'] = np.sqrt(np.sum(((freq_bins - features['spectral_centroid'])**2) * mean_spectrum) / np.sum(mean_spectrum)) if np.sum(mean_spectrum) > 0 else 0
    
    # Spectral flatness (ratio of geometric mean to arithmetic mean)
    epsilon = 1e-10  # Small value to avoid log(0)
    features['spectral_flatness'] = np.exp(np.mean(np.log(mean_spectrum + epsilon))) / (np.mean(mean_spectrum) + epsilon)
    
    # Spectral roll-off (frequency below which 85% of energy is contained)
    cumsum = np.cumsum(mean_spectrum)
    threshold = 0.85 * cumsum[-1]
    rolloff_idx = np.where(cumsum >= threshold)[0][0] if len(np.where(cumsum >= threshold)[0]) > 0 else 0
    features['spectral_rolloff'] = freq_bins[rolloff_idx] if rolloff_idx < len(freq_bins) else 0
    
    return features


def extract_features_from_stft_imu(stft_result, sr=100):
    """
    Extract features from IMU STFT result
    Similar to extract_features_from_stft but optimized for IMU data
    
    Parameters:
    -----------
    stft_result : numpy.ndarray
        STFT result from IMU data processing
    sr : int
        Sampling rate (default: 100 Hz for typical IMU sensors)
        
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    # Convert to magnitude
    magnitude = np.abs(stft_result)
    
    # Calculate frequency bins
    n_fft = 2 * (magnitude.shape[0] - 1)
    freq_bins = np.fft.rfftfreq(n_fft, 1/sr)
    
    # Extract features
    features = {}
    
    # Mean magnitude across time for each frequency
    mean_spectrum = np.mean(magnitude, axis=1)
    
    # Energy in different frequency bands relative to Nyquist frequency
    nyquist = sr / 2
    low_idx = np.where(freq_bins <= 0.2 * nyquist)[0]  # 0-20% of Nyquist
    mid_idx = np.where((freq_bins > 0.2 * nyquist) & (freq_bins <= 0.5 * nyquist))[0]  # 20-50% of Nyquist
    high_idx = np.where(freq_bins > 0.5 * nyquist)[0]  # 50-100% of Nyquist
    
    features['energy_low'] = np.sum(mean_spectrum[low_idx]) if len(low_idx) > 0 else 0
    features['energy_mid'] = np.sum(mean_spectrum[mid_idx]) if len(mid_idx) > 0 else 0
    features['energy_high'] = np.sum(mean_spectrum[high_idx]) if len(high_idx) > 0 else 0
    
    # Total energy
    features['energy_total'] = np.sum(mean_spectrum)
    
    # Spectral centroid (weighted mean of frequencies)
    if len(freq_bins) == len(mean_spectrum):
        features['spectral_centroid'] = np.sum(freq_bins * mean_spectrum) / np.sum(mean_spectrum) if np.sum(mean_spectrum) > 0 else 0
    
    # Spectral spread (weighted std of frequencies)
    if len(freq_bins) == len(mean_spectrum) and features['spectral_centroid'] > 0:
        features['spectral_spread'] = np.sqrt(np.sum(((freq_bins - features['spectral_centroid'])**2) * mean_spectrum) / np.sum(mean_spectrum)) if np.sum(mean_spectrum) > 0 else 0
    
    # Spectral flatness (ratio of geometric mean to arithmetic mean)
    epsilon = 1e-10  # Small value to avoid log(0)
    features['spectral_flatness'] = np.exp(np.mean(np.log(mean_spectrum + epsilon))) / (np.mean(mean_spectrum) + epsilon)
    
    # Spectral roll-off (frequency below which 85% of energy is contained)
    cumsum = np.cumsum(mean_spectrum)
    threshold = 0.85 * cumsum[-1] if len(cumsum) > 0 and cumsum[-1] > 0 else 0
    rolloff_idx = np.where(cumsum >= threshold)[0][0] if len(np.where(cumsum >= threshold)[0]) > 0 else 0
    features['spectral_rolloff'] = freq_bins[rolloff_idx] if rolloff_idx < len(freq_bins) else 0
    
    # Peak frequency
    peak_idx = np.argmax(mean_spectrum) if len(mean_spectrum) > 0 else 0
    features['peak_freq'] = freq_bins[peak_idx] if peak_idx < len(freq_bins) else 0
    
    return features

def extract_features_from_imu(x_accel, y_accel, z_accel):
    """
    Extract features from IMU data
    
    Parameters:
    -----------
    x_accel, y_accel, z_accel : numpy.ndarray
        Acceleration values for each axis
        
    Returns:
    --------
    features : dict
        Dictionary of extracted features
    """
    features = {}
    
    # Time domain features for each axis
    for axis_name, axis_data in zip(['x', 'y', 'z'], [x_accel, y_accel, z_accel]):
        features[f'{axis_name}_mean'] = np.mean(axis_data)
        features[f'{axis_name}_std'] = np.std(axis_data)
        features[f'{axis_name}_rms'] = np.sqrt(np.mean(np.square(axis_data)))
        features[f'{axis_name}_peak'] = np.max(np.abs(axis_data))
        features[f'{axis_name}_kurtosis'] = stats.kurtosis(axis_data)
        features[f'{axis_name}_skewness'] = stats.skew(axis_data)
        
        # Frequency domain features
        fft_result = np.abs(np.fft.rfft(axis_data))
        features[f'{axis_name}_energy'] = np.sum(fft_result**2)
        
        if len(fft_result) > 0:
            # Peak frequency
            features[f'{axis_name}_peak_freq'] = np.argmax(fft_result)
            
            # Frequency band energy
            freq_bins = np.fft.rfftfreq(len(axis_data))
            low_idx = np.where(freq_bins <= 0.1)[0]  # Low frequency (0-10% of Nyquist)
            mid_idx = np.where((freq_bins > 0.1) & (freq_bins <= 0.4))[0]  # Mid frequency
            high_idx = np.where(freq_bins > 0.4)[0]  # High frequency
            
            features[f'{axis_name}_energy_low'] = np.sum(fft_result[low_idx]**2) if len(low_idx) > 0 else 0
            features[f'{axis_name}_energy_mid'] = np.sum(fft_result[mid_idx]**2) if len(mid_idx) > 0 else 0
            features[f'{axis_name}_energy_high'] = np.sum(fft_result[high_idx]**2) if len(high_idx) > 0 else 0
    
    # Combined magnitude features
    magnitude = np.sqrt(x_accel**2 + y_accel**2 + z_accel**2)
    features['magnitude_mean'] = np.mean(magnitude)
    features['magnitude_std'] = np.std(magnitude)
    features['magnitude_rms'] = np.sqrt(np.mean(np.square(magnitude)))
    features['magnitude_peak'] = np.max(magnitude)
    
    return features

def prepare_data_for_ml(acoustic_file, imu_file, tool_lifetime):
    """
    Prepare data for machine learning using the existing processing functions
    
    Parameters:
    -----------
    acoustic_file : str
        Path to acoustic data file
    imu_file : str
        Path to IMU data file
    tool_lifetime : float
        Tool lifetime in seconds
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable (tool wear)
    features_dict : dict
        Dictionary of feature dictionaries
    """
    # Read files using the existing File_reader.py
    from File_reader import read_csv_file
    
    acoustic_df = read_csv_file(acoustic_file, '2')  # '2' for acoustic data
    
    # For IMU data, we need to read it directly since it has a different format
    try:
        imu_df = pd.read_csv(imu_file)
        print("IMU data loaded successfully")
        print("IMU columns:", imu_df.columns.tolist())
    except Exception as e:
        print(f"Error reading IMU file: {e}")
        imu_df = None
    
    if acoustic_df is None or imu_df is None:
        raise ValueError("Failed to read input files")
    
    # Process acoustic data using the existing function
    stft_result, sr = acoustic_processing(acoustic_df)
    
    # Extract features from STFT
    acoustic_features = extract_features_from_stft(stft_result, sr)
    
    # Extract IMU features if the required columns exist
    imu_features = {}
    imu_stft_features = {}  # New: for storing STFT-based IMU features
    try:
        # Check if we have the expected columns from the IMU data
        if "X (g)" in imu_df.columns and "Y (g)" in imu_df.columns and "Z (g)" in imu_df.columns:
            print("Using X (g), Y (g), Z (g) columns for IMU data processing")
            x_accel = imu_df["X (g)"].values
            y_accel = imu_df["Y (g)"].values
            z_accel = imu_df["Z (g)"].values
            imu_features = extract_features_from_imu(x_accel, y_accel, z_accel)
            
            # Add STFT-based feature extraction for IMU data
            print("Calculating STFT features for IMU data...")
            
            # Process each axis with STFT and extract features
            imu_stft_features_x = {}
            imu_stft_features_y = {}
            imu_stft_features_z = {}
            
            # Use the same parameters as in IMU_data.py
            n_fft = 8192
            hop_length = 2048
            win_length = 8192
            
            # Compute STFT for each axis
            stft_x = librosa.stft(x_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            stft_y = librosa.stft(y_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            stft_z = librosa.stft(z_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            
            # Estimate sampling rate if needed
            if not 'sr' in locals() or sr is None:
                # This is an approximation - you might want to adjust based on the data
                sr = 100  # Typical IMU sampling rate
            
            # Extract features from STFT results
            x_features = extract_features_from_stft_imu(stft_x, sr)
            y_features = extract_features_from_stft_imu(stft_y, sr)
            z_features = extract_features_from_stft_imu(stft_z, sr)
            
            # Prefix the features with axis name
            imu_stft_features_x = {'x_' + k: v for k, v in x_features.items()}
            imu_stft_features_y = {'y_' + k: v for k, v in y_features.items()}
            imu_stft_features_z = {'z_' + k: v for k, v in z_features.items()}
            
            # Combine all IMU STFT features
            imu_stft_features = {**imu_stft_features_x, **imu_stft_features_y, **imu_stft_features_z}
            
        # Fallback to traditional column names if they exist
        elif "Accel_X" in imu_df.columns and "Accel_Y" in imu_df.columns and "Accel_Z" in imu_df.columns:
            print("Using Accel_X, Accel_Y, Accel_Z columns for IMU data processing")
            x_accel = imu_df["Accel_X"].values
            y_accel = imu_df["Accel_Y"].values
            z_accel = imu_df["Accel_Z"].values
            imu_features = extract_features_from_imu(x_accel, y_accel, z_accel)
            
            # Add STFT-based feature extraction for IMU data
            print("Calculating STFT features for IMU data...")
            
            # Process each axis with STFT and extract features
            imu_stft_features_x = {}
            imu_stft_features_y = {}
            imu_stft_features_z = {}
            
            # Use the same parameters as in IMU_data.py
            n_fft = 8192
            hop_length = 2048
            win_length = 8192
            
            # Compute STFT for each axis
            stft_x = librosa.stft(x_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            stft_y = librosa.stft(y_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            stft_z = librosa.stft(z_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            
            # Estimate sampling rate if needed
            if not 'sr' in locals() or sr is None:
                # This is an approximation - you might want to adjust based on the data
                sr = 100  # Typical IMU sampling rate
            
            # Extract features from STFT results
            x_features = extract_features_from_stft_imu(stft_x, sr)
            y_features = extract_features_from_stft_imu(stft_y, sr)
            z_features = extract_features_from_stft_imu(stft_z, sr)
            
            # Prefix the features with axis name
            imu_stft_features_x = {'x_' + k: v for k, v in x_features.items()}
            imu_stft_features_y = {'y_' + k: v for k, v in y_features.items()}
            imu_stft_features_z = {'z_' + k: v for k, v in z_features.items()}
            
            # Combine all IMU STFT features
            imu_stft_features = {**imu_stft_features_x, **imu_stft_features_y, **imu_stft_features_z}
            
        else:
            print("Warning: IMU file doesn't have expected acceleration columns.")
            print("Available columns:", imu_df.columns.tolist())
            
            # Create synthetic IMU features based on acoustic data
            print("Creating synthetic IMU features for demonstration.")
            # Use the amplitude as a basis for synthetic acceleration
            amplitude = acoustic_df["Amplitude"].values
            x_accel = amplitude * 0.5 + np.random.normal(0, 0.1, len(amplitude))
            y_accel = amplitude * 0.3 + np.random.normal(0, 0.1, len(amplitude))
            z_accel = amplitude * 0.7 + np.random.normal(0, 0.1, len(amplitude))
            imu_features = extract_features_from_imu(x_accel, y_accel, z_accel)
            
            # Add STFT-based feature extraction for IMU data
            print("Calculating STFT features for IMU data...")
            
            # Process each axis with STFT and extract features
            imu_stft_features_x = {}
            imu_stft_features_y = {}
            imu_stft_features_z = {}
            
            # Use the same parameters as in IMU_data.py
            n_fft = 8192
            hop_length = 2048
            win_length = 8192
            
            # Compute STFT for each axis
            stft_x = librosa.stft(x_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            stft_y = librosa.stft(y_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            stft_z = librosa.stft(z_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
            
            # Estimate sampling rate if needed
            if not 'sr' in locals() or sr is None:
                # This is an approximation - you might want to adjust based on the data
                sr = 100  # Typical IMU sampling rate
            
            # Extract features from STFT results
            x_features = extract_features_from_stft_imu(stft_x, sr)
            y_features = extract_features_from_stft_imu(stft_y, sr)
            z_features = extract_features_from_stft_imu(stft_z, sr)
            
            # Prefix the features with axis name
            imu_stft_features_x = {'x_' + k: v for k, v in x_features.items()}
            imu_stft_features_y = {'y_' + k: v for k, v in y_features.items()}
            imu_stft_features_z = {'z_' + k: v for k, v in z_features.items()}
            
            # Combine all IMU STFT features
            imu_stft_features = {**imu_stft_features_x, **imu_stft_features_y, **imu_stft_features_z}
            
    except Exception as e:
        print(f"Error extracting IMU features: {e}")
        # Create synthetic IMU features for demonstration
        print("Creating synthetic IMU features after error.")
        # Generate random acceleration values
        n_samples = 1000
        x_accel = np.random.normal(0, 1, n_samples)
        y_accel = np.random.normal(0, 1, n_samples)
        z_accel = np.random.normal(0, 1, n_samples)
        imu_features = extract_features_from_imu(x_accel, y_accel, z_accel)
        
        # Add STFT-based feature extraction for synthetic IMU data
        print("Calculating STFT features for synthetic IMU data...")
        n_fft = 8192
        hop_length = 2048
        win_length = 8192
        
        # Compute STFT for each axis
        stft_x = librosa.stft(x_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_y = librosa.stft(y_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        stft_z = librosa.stft(z_accel, n_fft=n_fft, hop_length=hop_length, win_length=win_length)
        
        # Extract features from STFT results
        x_features = extract_features_from_stft_imu(stft_x, 100)  # Assuming 100Hz for synthetic data
        y_features = extract_features_from_stft_imu(stft_y, 100)
        z_features = extract_features_from_stft_imu(stft_z, 100)
        
        # Prefix the features with axis name
        imu_stft_features_x = {'x_' + k: v for k, v in x_features.items()}
        imu_stft_features_y = {'y_' + k: v for k, v in y_features.items()}
        imu_stft_features_z = {'z_' + k: v for k, v in z_features.items()}
        
        # Combine all IMU STFT features
        imu_stft_features = {**imu_stft_features_x, **imu_stft_features_y, **imu_stft_features_z}
    
    # Combine features: now including the STFT-based IMU features
    all_features = {**acoustic_features, **imu_features, **imu_stft_features}
    
    # Convert to dataframe with one row
    features_df = pd.DataFrame([all_features])
    
    # Create target variable (assuming fresh tool at recording)
    wear_state = 0.0  # 0% wear for a new tool
    
    # Return features and target
    X = features_df
    y = pd.Series([wear_state])
    
    return X, y, all_features

def scale_features(X, scaler=None):
    """
    Scale features for ML
    
    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    scaler : sklearn.preprocessing.StandardScaler, optional
        Pre-fitted scaler
        
    Returns:
    --------
    X_scaled : pandas.DataFrame
        Scaled features
    scaler : sklearn.preprocessing.StandardScaler
        Fitted scaler
    """
    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)
    
    # Convert back to DataFrame
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    return X_scaled_df, scaler

def collect_dataset_from_directory(data_dir, tool_lifetime):
    """
    Collect dataset from directory containing multiple recording pairs
    
    Parameters:
    -----------
    data_dir : str
        Directory containing acoustic and IMU data files
    tool_lifetime : float
        Tool lifetime in seconds
        
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable (tool wear)
    """
    # Check if we're in a sensor-specific subfolder and need to look for the other sensor data
    folder_name = os.path.basename(data_dir).lower()
    parent_dir = os.path.dirname(data_dir)
    
    # Initialize paths for acoustic and IMU directories
    acoustic_dir = data_dir
    imu_dir = data_dir
    
    # Check if we're in a sensor-specific folder
    if 'acoustic' in folder_name:
        # We're in acoustic folder, look for IMU folder
        potential_imu_dir = os.path.join(parent_dir, 'IMU Data')
        if os.path.exists(potential_imu_dir):
            # Look for matching tool subfolder
            tool_name = os.path.basename(data_dir).lower()
            for item in os.listdir(potential_imu_dir):
                if item.lower() == tool_name.lower():
                    imu_dir = os.path.join(potential_imu_dir, item)
                    print(f"Found matching IMU directory: {imu_dir}")
                    break
                    
    elif 'imu' in folder_name:
        # We're in IMU folder, look for acoustic folder
        potential_acoustic_dir = os.path.join(parent_dir, 'Acoustic Data')
        if os.path.exists(potential_acoustic_dir):
            # Look for matching tool subfolder
            tool_name = os.path.basename(data_dir).lower()
            for item in os.listdir(potential_acoustic_dir):
                if item.lower() == tool_name.lower():
                    acoustic_dir = os.path.join(potential_acoustic_dir, item)
                    print(f"Found matching acoustic directory: {acoustic_dir}")
                    break
    
    # Find all acoustic and IMU files in their respective directories
    acoustic_files = sorted(glob.glob(os.path.join(acoustic_dir, "*.csv")))
    imu_files = sorted(glob.glob(os.path.join(imu_dir, "*.csv")))
    
    print(f"Found {len(acoustic_files)} acoustic files in {acoustic_dir}")
    print(f"Found {len(imu_files)} IMU files in {imu_dir}")
    
    if len(acoustic_files) == 0 or len(imu_files) == 0:
        print("Error: Missing data files. Need both acoustic and IMU data.")
        print("Make sure you have CSV files in both sensor directories.")
        raise ValueError("No data files found in the directory structure")
    
    # Assuming files are matched by index position (in alphabetical order)
    # If numbers don't match, use the minimum
    n_pairs = min(len(acoustic_files), len(imu_files))
    
    # Initialize lists for features and targets
    all_features = []
    wear_states = []
    
    # Process each pair
    for i in range(n_pairs):
        acoustic_file = acoustic_files[i]
        imu_file = imu_files[i]
        
        print(f"Processing pair {i+1}/{n_pairs}: ")
        print(f"  - Acoustic: {os.path.basename(acoustic_file)}")
        print(f"  - IMU:      {os.path.basename(imu_file)}")
        
        try:
            # Calculate wear state based on position in sequence
            # Assuming files are sorted by time/wear progression
            wear_state = i / (n_pairs - 1) if n_pairs > 1 else 0
            
            # Extract features
            X, _, features = prepare_data_for_ml(acoustic_file, imu_file, tool_lifetime)
            
            # Add to lists
            all_features.append(features)
            wear_states.append(wear_state)
            
        except Exception as e:
            print(f"Error processing pair: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not all_features:
        raise ValueError("Failed to extract features from any file pairs")
    
    # Convert to dataframe
    X = pd.DataFrame(all_features)
    y = pd.Series(wear_states, name="tool_wear")
    
    print(f"Created dataset with {X.shape[0]} samples and {X.shape[1]} features")
    
    return X, y

if __name__ == "__main__":
    # Example usage
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python data_preparation.py acoustic_file.csv imu_file.csv tool_lifetime_seconds")
        sys.exit(1)
    
    acoustic_file = sys.argv[1]
    imu_file = sys.argv[2]
    tool_lifetime = float(sys.argv[3])
    
    X, y, features = prepare_data_for_ml(acoustic_file, imu_file, tool_lifetime)
    
    print("Extracted features:")
    for feature, value in features.items():
        print(f"  {feature}: {value}")
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable: {y.values[0]:.2f} (tool wear)")
    
    # Example scaling
    X_scaled, scaler = scale_features(X)
    print("\nScaled features (first few):")
    print(X_scaled.iloc[0][:5])