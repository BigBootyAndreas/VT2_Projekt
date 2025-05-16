import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import StandardScaler, RobustScaler, QuantileTransformer, PowerTransformer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
import os
import sys
import glob
import librosa
import warnings

# Suppress specific warnings about n_quantiles
warnings.filterwarnings('ignore', message='n_quantiles .* is greater than the total number of samples')
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn.preprocessing._data')

# Import the existing processing functions
# Add the current directory to path to ensure imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import the existing signal processing functions
from Acoustic_data import acoustic_processing  
from IMU_data import imu_processing
from File_reader import read_csv_file

def extract_features_from_stft(stft_result, sr):
    """
    Robust STFT feature extraction with comprehensive NaN prevention
    """
    if stft_result is None:
        print("Warning: STFT result is None, returning default features")
        return _get_default_acoustic_features()
    
    try:
        magnitude = np.abs(stft_result)
        
        # Check if magnitude has valid data
        if magnitude.size == 0 or np.all(magnitude == 0):
            print("Warning: Empty or zero magnitude spectrum, using defaults")
            return _get_default_acoustic_features()
        
        n_fft = 2 * (magnitude.shape[0] - 1)
        freq_bins = np.fft.rfftfreq(n_fft, 1/sr)
        
        features = {}
        
        # Mean magnitude across time for each frequency
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_spectrum = np.mean(magnitude, axis=1)
            
            # Replace any NaN/Inf values
            mean_spectrum = np.nan_to_num(mean_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Enhanced frequency band analysis with safety checks
        bands = {
            'low': (0, 500),
            'mid_low': (500, 1000), 
            'mid': (1000, 2000),
            'mid_high': (2000, 4000),
            'high': (4000, 8000),
            'very_high': (8000, sr/2)
        }
        
        total_energy = np.sum(mean_spectrum)
        if total_energy == 0:
            total_energy = 1e-10  # Prevent division by zero
        
        for band_name, (low_freq, high_freq) in bands.items():
            # Ensure frequency bounds are valid
            low_freq = max(0, min(low_freq, sr/2))
            high_freq = max(low_freq, min(high_freq, sr/2))
            
            band_idx = np.where((freq_bins >= low_freq) & (freq_bins <= high_freq))[0]
            if len(band_idx) > 0:
                band_energy = np.sum(mean_spectrum[band_idx])
                features[f'energy_{band_name}'] = _safe_float(band_energy)
                features[f'peak_{band_name}'] = _safe_float(np.max(mean_spectrum[band_idx]))
                # Relative energy
                features[f'energy_{band_name}_rel'] = _safe_float(band_energy / total_energy)
            else:
                features[f'energy_{band_name}'] = 0.0
                features[f'peak_{band_name}'] = 0.0
                features[f'energy_{band_name}_rel'] = 0.0
        
        # Total energy
        features['energy_total'] = _safe_float(total_energy)
        
        # Spectral centroid (weighted mean of frequencies)
        if len(freq_bins) == len(mean_spectrum) and total_energy > 1e-10:
            centroid = np.sum(freq_bins * mean_spectrum) / total_energy
            features['spectral_centroid'] = _safe_float(centroid)
        else:
            features['spectral_centroid'] = 0.0
        
        # Spectral spread (weighted std of frequencies)
        if features['spectral_centroid'] > 0 and total_energy > 1e-10:
            spread = np.sqrt(np.sum(((freq_bins - features['spectral_centroid'])**2) * mean_spectrum) / total_energy)
            features['spectral_spread'] = _safe_float(spread)
        else:
            features['spectral_spread'] = 0.0
        
        # Spectral flatness (ratio of geometric mean to arithmetic mean)
        epsilon = 1e-10
        # Use only positive values for geometric mean calculation
        positive_spectrum = mean_spectrum[mean_spectrum > epsilon]
        if len(positive_spectrum) > 0:
            geo_mean = np.exp(np.mean(np.log(positive_spectrum + epsilon)))
            arith_mean = np.mean(mean_spectrum) + epsilon
            features['spectral_flatness'] = _safe_float(geo_mean / arith_mean)
        else:
            features['spectral_flatness'] = 0.0
        
        # Multiple spectral roll-off points
        for percentile in [0.75, 0.85, 0.95]:
            try:
                cumsum = np.cumsum(mean_spectrum)
                if cumsum[-1] > 0:
                    threshold = percentile * cumsum[-1]
                    rolloff_idx = np.where(cumsum >= threshold)[0]
                    if len(rolloff_idx) > 0:
                        rolloff_idx = rolloff_idx[0]
                        features[f'spectral_rolloff_{int(percentile*100)}'] = _safe_float(freq_bins[rolloff_idx])
                    else:
                        features[f'spectral_rolloff_{int(percentile*100)}'] = 0.0
                else:
                    features[f'spectral_rolloff_{int(percentile*100)}'] = 0.0
            except:
                features[f'spectral_rolloff_{int(percentile*100)}'] = 0.0
        
        # Spectral flux (measure of how quickly the power spectrum changes)
        if magnitude.shape[1] > 1:
            try:
                spectral_flux = np.mean(np.diff(magnitude, axis=1)**2)
                features['spectral_flux'] = _safe_float(spectral_flux)
                
                spectral_var = np.std(np.sum(magnitude, axis=0))
                features['spectral_variability'] = _safe_float(spectral_var)
            except:
                features['spectral_flux'] = 0.0
                features['spectral_variability'] = 0.0
        else:
            features['spectral_flux'] = 0.0
            features['spectral_variability'] = 0.0
        
        # Final check: ensure all features are valid
        for key, value in features.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                print(f"Warning: Invalid value in feature {key}, replacing with 0")
                features[key] = 0.0
        
        return features
        
    except Exception as e:
        print(f"Error in extract_features_from_stft: {e}")
        import traceback
        traceback.print_exc()
        return _get_default_acoustic_features()

def _safe_float(value):
    """Convert value to safe float, handling NaN and Inf"""
    if np.isnan(value) or np.isinf(value):
        return 0.0
    return float(value)

def extract_features_from_imu(x_accel, y_accel, z_accel, sr=100):
    """
    Robust IMU feature extraction with comprehensive NaN prevention
    """
    try:
        # Input validation and cleaning
        if len(x_accel) == 0 or len(y_accel) == 0 or len(z_accel) == 0:
            print("Warning: Empty IMU data, using defaults")
            return _get_default_imu_features()
        
        # Clean input data
        x_accel = np.asarray(x_accel, dtype=float)
        y_accel = np.asarray(y_accel, dtype=float)
        z_accel = np.asarray(z_accel, dtype=float)
        
        # Replace NaN/Inf values
        x_accel = np.nan_to_num(x_accel, nan=0.0, posinf=0.0, neginf=0.0)
        y_accel = np.nan_to_num(y_accel, nan=0.0, posinf=0.0, neginf=0.0)
        z_accel = np.nan_to_num(z_accel, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Calculate magnitude safely
        magnitude = np.sqrt(x_accel**2 + y_accel**2 + z_accel**2)
        magnitude = np.nan_to_num(magnitude, nan=0.0, posinf=0.0, neginf=0.0)
        
        features = {}
        
        # Enhanced time-domain features for each axis
        for axis_name, axis_data in zip(['x', 'y', 'z', 'magnitude'], [x_accel, y_accel, z_accel, magnitude]):
            # Basic statistics with safety checks
            features[f'{axis_name}_mean'] = _safe_float(np.mean(axis_data))
            features[f'{axis_name}_std'] = _safe_float(np.std(axis_data))
            features[f'{axis_name}_var'] = _safe_float(np.var(axis_data))
            
            # RMS with safety check
            rms = np.sqrt(np.mean(np.square(axis_data)))
            features[f'{axis_name}_rms'] = _safe_float(rms)
            
            features[f'{axis_name}_peak'] = _safe_float(np.max(np.abs(axis_data)))
            features[f'{axis_name}_peak_to_peak'] = _safe_float(np.max(axis_data) - np.min(axis_data))
            
            # Higher-order statistics with error handling
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    features[f'{axis_name}_kurtosis'] = _safe_float(stats.kurtosis(axis_data))
                    features[f'{axis_name}_skewness'] = _safe_float(stats.skew(axis_data))
            except:
                features[f'{axis_name}_kurtosis'] = 0.0
                features[f'{axis_name}_skewness'] = 0.0
            
            # Percentiles
            try:
                features[f'{axis_name}_p25'] = _safe_float(np.percentile(axis_data, 25))
                features[f'{axis_name}_p75'] = _safe_float(np.percentile(axis_data, 75))
                features[f'{axis_name}_iqr'] = features[f'{axis_name}_p75'] - features[f'{axis_name}_p25']
            except:
                features[f'{axis_name}_p25'] = 0.0
                features[f'{axis_name}_p75'] = 0.0
                features[f'{axis_name}_iqr'] = 0.0
            
            # Shape factor and crest factor with safety checks
            mean_abs = np.mean(np.abs(axis_data))
            if mean_abs > 1e-10 and rms > 1e-10:
                features[f'{axis_name}_shape_factor'] = _safe_float(rms / mean_abs)
                features[f'{axis_name}_crest_factor'] = _safe_float(features[f'{axis_name}_peak'] / rms)
            else:
                features[f'{axis_name}_shape_factor'] = 0.0
                features[f'{axis_name}_crest_factor'] = 0.0
            
            # Zero crossing rate
            try:
                zero_crossings = np.where(np.diff(np.sign(axis_data)))[0]
                features[f'{axis_name}_zcr'] = _safe_float(len(zero_crossings) / len(axis_data))
            except:
                features[f'{axis_name}_zcr'] = 0.0
        
        # Cross-axis correlations with safety checks
        try:
            if len(x_accel) > 1 and len(y_accel) > 1:
                features['xy_correlation'] = _safe_float(np.corrcoef(x_accel, y_accel)[0, 1])
            else:
                features['xy_correlation'] = 0.0
                
            if len(x_accel) > 1 and len(z_accel) > 1:
                features['xz_correlation'] = _safe_float(np.corrcoef(x_accel, z_accel)[0, 1])
            else:
                features['xz_correlation'] = 0.0
                
            if len(y_accel) > 1 and len(z_accel) > 1:
                features['yz_correlation'] = _safe_float(np.corrcoef(y_accel, z_accel)[0, 1])
            else:
                features['yz_correlation'] = 0.0
        except:
            features['xy_correlation'] = 0.0
            features['xz_correlation'] = 0.0
            features['yz_correlation'] = 0.0
        
        # Enhanced STFT frequency-domain features for each axis
        for axis_name, axis_data in zip(['x', 'y', 'z', 'magnitude'], [x_accel, y_accel, z_accel, magnitude]):
            # Use appropriate window size for IMU data
            n_fft = min(8192, len(axis_data))
            if n_fft < 32:  # Minimum window size
                # Not enough data for STFT, use defaults
                for band_name in ['very_low', 'low', 'mid', 'high']:
                    features[f'{axis_name}_stft_energy_{band_name}'] = 0.0
                    features[f'{axis_name}_stft_energy_{band_name}_rel'] = 0.0
                features[f'{axis_name}_stft_energy_total'] = 0.0
                features[f'{axis_name}_stft_spectral_centroid'] = 0.0
                features[f'{axis_name}_stft_spectral_spread'] = 0.0
                features[f'{axis_name}_stft_spectral_flatness'] = 0.0
                for percentile in [75, 85, 95]:
                    features[f'{axis_name}_stft_spectral_rolloff_{percentile}'] = 0.0
                continue
            
            hop_length = max(n_fft // 4, 1)
            
            try:
                # Ensure we have enough data
                if len(axis_data) < hop_length:
                    axis_data = np.pad(axis_data, (0, hop_length - len(axis_data)), 'constant')
                
                stft_result = librosa.stft(axis_data, n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
                magnitude_spec = np.abs(stft_result)
                
                # Safety check
                if magnitude_spec.size == 0 or np.all(magnitude_spec == 0):
                    magnitude_spec = np.ones((n_fft//2 + 1, 1)) * 1e-10
                
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    mean_spectrum = np.mean(magnitude_spec, axis=1)
                    mean_spectrum = np.nan_to_num(mean_spectrum, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Frequency bins
                freq_bins = np.fft.rfftfreq(n_fft, 1/sr)
                
                # Enhanced frequency band analysis
                nyquist = sr / 2
                bands = {
                    'very_low': (0, 0.1 * nyquist),
                    'low': (0.1 * nyquist, 0.25 * nyquist),
                    'mid': (0.25 * nyquist, 0.5 * nyquist),
                    'high': (0.5 * nyquist, nyquist)
                }
                
                total_energy = np.sum(mean_spectrum)
                if total_energy == 0:
                    total_energy = 1e-10
                
                for band_name, (low_freq, high_freq) in bands.items():
                    band_idx = np.where((freq_bins >= low_freq) & (freq_bins <= high_freq))[0]
                    if len(band_idx) > 0:
                        band_energy = np.sum(mean_spectrum[band_idx])
                        features[f'{axis_name}_stft_energy_{band_name}'] = _safe_float(band_energy)
                        features[f'{axis_name}_stft_energy_{band_name}_rel'] = _safe_float(band_energy / total_energy)
                    else:
                        features[f'{axis_name}_stft_energy_{band_name}'] = 0.0
                        features[f'{axis_name}_stft_energy_{band_name}_rel'] = 0.0
                
                # Total energy
                features[f'{axis_name}_stft_energy_total'] = _safe_float(total_energy)
                
                # Spectral centroid
                if len(freq_bins) == len(mean_spectrum) and total_energy > 1e-10:
                    centroid = np.sum(freq_bins * mean_spectrum) / total_energy
                    features[f'{axis_name}_stft_spectral_centroid'] = _safe_float(centroid)
                else:
                    features[f'{axis_name}_stft_spectral_centroid'] = 0.0
                    
                # Spectral spread
                if features[f'{axis_name}_stft_spectral_centroid'] > 0 and total_energy > 1e-10:
                    spread = np.sqrt(
                        np.sum(((freq_bins - features[f'{axis_name}_stft_spectral_centroid'])**2) * mean_spectrum) / total_energy
                    )
                    features[f'{axis_name}_stft_spectral_spread'] = _safe_float(spread)
                else:
                    features[f'{axis_name}_stft_spectral_spread'] = 0.0
                    
                # Spectral flatness
                epsilon = 1e-10
                positive_spectrum = mean_spectrum[mean_spectrum > epsilon]
                if len(positive_spectrum) > 0:
                    geo_mean = np.exp(np.mean(np.log(positive_spectrum + epsilon)))
                    arith_mean = np.mean(mean_spectrum) + epsilon
                    features[f'{axis_name}_stft_spectral_flatness'] = _safe_float(geo_mean / arith_mean)
                else:
                    features[f'{axis_name}_stft_spectral_flatness'] = 0.0
                
                # Multiple spectral roll-off points
                for percentile in [0.75, 0.85, 0.95]:
                    try:
                        cumsum = np.cumsum(mean_spectrum)
                        if cumsum[-1] > 0:
                            threshold = percentile * cumsum[-1]
                            rolloff_idx = np.where(cumsum >= threshold)[0]
                            if len(rolloff_idx) > 0:
                                rolloff_idx = rolloff_idx[0]
                                features[f'{axis_name}_stft_spectral_rolloff_{int(percentile*100)}'] = _safe_float(freq_bins[rolloff_idx])
                            else:
                                features[f'{axis_name}_stft_spectral_rolloff_{int(percentile*100)}'] = 0.0
                        else:
                            features[f'{axis_name}_stft_spectral_rolloff_{int(percentile*100)}'] = 0.0
                    except:
                        features[f'{axis_name}_stft_spectral_rolloff_{int(percentile*100)}'] = 0.0
        
            except Exception as e:
                print(f"Warning: Failed to extract STFT features for {axis_name}: {e}")
                # Add default values for failed features
                for band_name in ['very_low', 'low', 'mid', 'high']:
                    features[f'{axis_name}_stft_energy_{band_name}'] = 0.0
                    features[f'{axis_name}_stft_energy_{band_name}_rel'] = 0.0
                features[f'{axis_name}_stft_energy_total'] = 0.0
                features[f'{axis_name}_stft_spectral_centroid'] = 0.0
                features[f'{axis_name}_stft_spectral_spread'] = 0.0
                features[f'{axis_name}_stft_spectral_flatness'] = 0.0
                for percentile in [75, 85, 95]:
                    features[f'{axis_name}_stft_spectral_rolloff_{percentile}'] = 0.0
        
        # Final check: ensure all features are valid
        for key, value in features.items():
            if not isinstance(value, (int, float)) or np.isnan(value) or np.isinf(value):
                print(f"Warning: Invalid value in IMU feature {key}, replacing with 0")
                features[key] = 0.0
        
        return features
        
    except Exception as e:
        print(f"Error in extract_features_from_imu: {e}")
        import traceback
        traceback.print_exc()
        return _get_default_imu_features()

# In data_preparation.py, update _get_default_imu_features to include all missing features

def _get_default_imu_features():
    """Get default IMU features when extraction fails - ENHANCED VERSION"""
    features = {}
    axes = ['x', 'y', 'z', 'magnitude']
    
    for axis in axes:
        # Time domain features
        features.update({
            f'{axis}_mean': 0.0, f'{axis}_std': 0.0, f'{axis}_var': 0.0,
            f'{axis}_rms': 0.0, f'{axis}_peak': 0.0, f'{axis}_peak_to_peak': 0.0,
            f'{axis}_kurtosis': 0.0, f'{axis}_skewness': 0.0,
            f'{axis}_p25': 0.0, f'{axis}_p75': 0.0, f'{axis}_iqr': 0.0,
            f'{axis}_shape_factor': 0.0, f'{axis}_crest_factor': 0.0, f'{axis}_zcr': 0.0
        })
        
        # STFT frequency domain features (complete set)
        for band in ['very_low', 'low', 'mid', 'high']:
            features[f'{axis}_stft_energy_{band}'] = 0.0
            features[f'{axis}_stft_energy_{band}_rel'] = 0.0
        
        features.update({
            f'{axis}_stft_energy_total': 0.0,
            f'{axis}_stft_spectral_centroid': 0.0,
            f'{axis}_stft_spectral_spread': 0.0,
            f'{axis}_stft_spectral_flatness': 0.0,
            f'{axis}_stft_spectral_rolloff_75': 0.0,
            f'{axis}_stft_spectral_rolloff_85': 0.0,
            f'{axis}_stft_spectral_rolloff_95': 0.0
        })
    
    # Cross-correlations
    features.update({
        'xy_correlation': 0.0, 'xz_correlation': 0.0, 'yz_correlation': 0.0
    })
    
    # Add missing spectral features for magnitude
    features.update({
        'magnitude_spectral_flux': 0.0,
        'magnitude_spectral_variability': 0.0
    })
    
    return features

def _get_default_acoustic_features():
    """Get default acoustic features when extraction fails - ENHANCED VERSION"""
    features = {}
    bands = ['low', 'mid_low', 'mid', 'mid_high', 'high', 'very_high']
    
    for band_name in bands:
        features[f'energy_{band_name}'] = 0.0
        features[f'peak_{band_name}'] = 0.0
        features[f'energy_{band_name}_rel'] = 0.0
    
    features.update({
        'energy_total': 0.0,
        'spectral_centroid': 0.0,
        'spectral_spread': 0.0,
        'spectral_flatness': 0.0,
        'spectral_rolloff_75': 0.0,
        'spectral_rolloff_85': 0.0,
        'spectral_rolloff_95': 0.0,
        'spectral_flux': 0.0,
        'spectral_variability': 0.0
    })
    
    return features

def advanced_data_preprocessing(X, y=None, scaler=None, feature_selector=None, 
                               remove_outliers=True, handle_correlations=True, 
                               verbose=True):
    """
    Enhanced preprocessing pipeline with better scaling and NaN handling
    """
    
    if verbose:
        print(f"Starting preprocessing with {X.shape[0]} samples and {X.shape[1]} features")
    
    preprocessing_info = {}
    X_processed = X.copy()
    
    # Step 1: Handle missing values
    if X_processed.isnull().any().any():
        if verbose:
            print("Found missing values, filling with median...")
        # Use median for more robust imputation
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                median_val = X_processed[col].median()
                if np.isnan(median_val):
                    # If median is NaN, use 0
                    median_val = 0.0
                X_processed[col].fillna(median_val, inplace=True)
        preprocessing_info['missing_values_handled'] = True
    
    # Step 2: Handle infinite values
    if np.isinf(X_processed.select_dtypes(include=[np.number]).values).any():
        if verbose:
            print("Found infinite values, replacing...")
        X_processed = X_processed.replace([np.inf, -np.inf], np.nan)
        # Fill newly created NaN values
        for col in X_processed.columns:
            if X_processed[col].isnull().any():
                median_val = X_processed[col].median()
                if np.isnan(median_val):
                    median_val = 0.0
                X_processed[col].fillna(median_val, inplace=True)
        preprocessing_info['infinite_values_handled'] = True
    
    # Step 3: Remove features with zero or near-zero variance
    initial_features = X_processed.shape[1]
    variance_selector = VarianceThreshold(threshold=0.01)
    X_var_filtered = variance_selector.fit_transform(X_processed)
    feature_names_after_variance = X_processed.columns[variance_selector.get_support()]
    X_processed = pd.DataFrame(X_var_filtered, columns=feature_names_after_variance)
    
    if verbose:
        removed_features = initial_features - X_processed.shape[1]
        print(f"Removed {removed_features} low-variance features")
    preprocessing_info['low_variance_features_removed'] = initial_features - X_processed.shape[1]
    
    # Step 4: Handle outliers using IQR method
    if remove_outliers:
        if verbose:
            print("Handling outliers using IQR method...")
        
        Q1 = X_processed.quantile(0.25)
        Q3 = X_processed.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier bounds
        outlier_multiplier = 1.5
        lower_bound = Q1 - outlier_multiplier * IQR
        upper_bound = Q3 + outlier_multiplier * IQR
        
        # Count outliers before handling
        outliers_count = 0
        for col in X_processed.columns:
            outliers_mask = (X_processed[col] < lower_bound[col]) | (X_processed[col] > upper_bound[col])
            outliers_count += outliers_mask.sum()
        
        # Cap outliers instead of removing them (to preserve sample count)
        for col in X_processed.columns:
            X_processed[col] = np.clip(X_processed[col], lower_bound[col], upper_bound[col])
        
        if verbose:
            print(f"Capped {outliers_count} outlier values")
        preprocessing_info['outliers_handled'] = outliers_count
    
    # Step 5: Remove highly correlated features
    if handle_correlations:
        if verbose:
            print("Removing highly correlated features...")
        
        initial_features = X_processed.shape[1]
        correlation_matrix = X_processed.corr().abs()
        
        # Create mask for upper triangular matrix
        upper_triangular = correlation_matrix.where(
            np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
        )
        
        # Find pairs with correlation > 0.95
        high_corr_features = []
        correlation_threshold = 0.95
        
        for column in upper_triangular.columns:
            correlated_features = list(upper_triangular.index[upper_triangular[column] > correlation_threshold])
            high_corr_features.extend(correlated_features)
        
        # Remove highly correlated features
        high_corr_features = list(set(high_corr_features))
        X_processed = X_processed.drop(high_corr_features, axis=1)
        
        if verbose:
            removed_features = initial_features - X_processed.shape[1]
            print(f"Removed {removed_features} highly correlated features")
        preprocessing_info['correlated_features_removed'] = initial_features - X_processed.shape[1]
    
    # Step 6: Feature selection (if target provided)
    if y is not None and feature_selector is None and X_processed.shape[1] > 50:
        if verbose:
            print(f"Performing feature selection (current: {X_processed.shape[1]} features)...")
        
        # Use SelectKBest with f_regression
        k_features = min(50, X_processed.shape[1])
        feature_selector = SelectKBest(score_func=f_regression, k=k_features)
        X_selected = feature_selector.fit_transform(X_processed, y)
        selected_features = X_processed.columns[feature_selector.get_support()]
        X_processed = pd.DataFrame(X_selected, columns=selected_features)
        
        if verbose:
            print(f"Selected {k_features} best features")
        preprocessing_info['feature_selection_applied'] = True
        preprocessing_info['features_selected'] = k_features
    elif feature_selector is not None:
        # Apply pre-fitted selector
        X_selected = feature_selector.transform(X_processed)
        selected_features = X_processed.columns[feature_selector.get_support()]
        X_processed = pd.DataFrame(X_selected, columns=selected_features)
    
    # Step 7: Smarter scaling based on data characteristics
    if scaler is None:
        if verbose:
            print("Choosing appropriate scaler...")
        
        # Adjust quantile transformer parameters based on sample size
        n_samples = X_processed.shape[0]
        
        # Check if data looks normally distributed
        normality_scores = []
        for col in X_processed.columns[:min(10, len(X_processed.columns))]:
            try:
                _, p_value = stats.normaltest(X_processed[col])
                normality_scores.append(p_value)
            except:
                normality_scores.append(0.0)
        
        avg_normality = np.mean(normality_scores) if normality_scores else 0.0
        
        if avg_normality > 0.05 and n_samples >= 100:  # Data appears normal and enough samples
            scaler = PowerTransformer(method='yeo-johnson', standardize=True)
            if verbose:
                print("Using PowerTransformer (data appears normal)")
        elif n_samples >= 100:  # Enough samples for quantile transform
            # Adjust n_quantiles based on sample size
            n_quantiles = min(1000, max(10, n_samples // 2))
            scaler = QuantileTransformer(output_distribution='normal', n_quantiles=n_quantiles)
            if verbose:
                print(f"Using QuantileTransformer with {n_quantiles} quantiles")
        else:  # Small sample size, use robust scaler
            scaler = RobustScaler()
            if verbose:
                print("Using RobustScaler (small sample size or robust to outliers)")
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_scaled = scaler.fit_transform(X_processed)
    else:
        # Apply pre-fitted scaler
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            X_scaled = scaler.transform(X_processed)
    
    # Convert back to DataFrame
    X_final = pd.DataFrame(X_scaled, columns=X_processed.columns)
    
    # Final check for any issues
    if X_final.isnull().any().any():
        if verbose:
            print("Warning: Found NaN after scaling, filling with zeros...")
        X_final = X_final.fillna(0)
    
    if np.isinf(X_final.values).any():
        if verbose:
            print("Warning: Found infinite values after scaling, clipping...")
        X_final = X_final.replace([np.inf, -np.inf], 0)
    
    preprocessing_info['final_feature_count'] = X_final.shape[1]
    preprocessing_info['scaler_type'] = type(scaler).__name__
    
    if verbose:
        print(f"Preprocessing complete: {X_final.shape[1]} features remaining")
        print(f"Final feature count: {X_final.shape[1]}")
    
    return X_final, scaler, feature_selector, preprocessing_info

# Rest of the functions remain the same...
def scale_features(X, scaler=None, preprocessing_info=None):
    """
    Enhanced feature scaling with robust preprocessing
    """
    # Use advanced preprocessing
    X_scaled, scaler, _, info = advanced_data_preprocessing(
        X, scaler=scaler, verbose=True
    )
    
    return X_scaled, scaler

def prepare_data_for_ml(acoustic_file, imu_file, tool_lifetime):
    """
    Prepare data for machine learning using enhanced processing functions
    """
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
    
    # Extract enhanced features from STFT
    acoustic_features = extract_features_from_stft(stft_result, sr)
    
    # Extract enhanced IMU features if the required columns exist
    imu_features = {}
    if "X (g)" in imu_df.columns and "Y (g)" in imu_df.columns and "Z (g)" in imu_df.columns:
        print("Using X (g), Y (g), Z (g) columns for IMU data processing")
        x_accel = imu_df["X (g)"].values
        y_accel = imu_df["Y (g)"].values
        z_accel = imu_df["Z (g)"].values
        imu_features = extract_features_from_imu(x_accel, y_accel, z_accel)
    else:
        print("IMU data does not contain expected columns.")
    
    # Combine features
    all_features = {**acoustic_features, **imu_features}
    
    # Convert to dataframe with one row
    features_df = pd.DataFrame([all_features])
    
    # Create target variable (assuming fresh tool at recording)
    wear_state = 0.0  # 0% wear for a new tool
    
    # Return features and target
    X = features_df
    y = pd.Series([wear_state])
    
    return X, y, all_features

def collect_dataset_from_directory(data_dir, tool_lifetime, failure_map=None):
    """
    Collect dataset with enhanced preprocessing and actual failure information
    """
    # [Rest of the function remains the same as before...]
    # Check if we're in a sensor-specific subfolder and need to look for the other sensor data
    folder_name = os.path.basename(data_dir).lower()
    parent_dir = os.path.dirname(data_dir)
    
    # Initialize paths for acoustic and IMU directories
    acoustic_dir = data_dir
    imu_dir = data_dir
    
    # Check if we're in a sensor-specific folder
    if 'acoustic' in folder_name or any(x in folder_name for x in ['drill', 'reamer', 'emill']):
        # We're in an acoustic folder
        acoustic_dir = data_dir
        
        # Look for IMU folder at the same level or parent level
        potential_imu_dirs = [
            os.path.join(parent_dir, 'IMU Data', os.path.basename(data_dir)),
            os.path.join(parent_dir, 'IMU_Data', os.path.basename(data_dir)),
            os.path.join(parent_dir, 'IMU'),
            os.path.join(parent_dir, folder_name.replace('acoustic', 'imu')),
            os.path.join(os.path.dirname(parent_dir), 'IMU Data', os.path.basename(data_dir))
        ]
        
        for potential_dir in potential_imu_dirs:
            if os.path.exists(potential_dir):
                imu_dir = potential_dir
                print(f"Found IMU directory: {imu_dir}")
                break
        else:
            print(f"Warning: Could not find IMU data directory. Checked:")
            for pd in potential_imu_dirs:
                print(f"  - {pd}")
            print("Will use dummy IMU data")
            imu_dir = None
    
    # Find all acoustic and IMU files in their respective directories
    acoustic_files = sorted(glob.glob(os.path.join(acoustic_dir, "*.csv")))
    
    if imu_dir and os.path.exists(imu_dir):
        imu_files = sorted(glob.glob(os.path.join(imu_dir, "*.csv")))
    else:
        imu_files = []
        print("No IMU directory found, will create dummy IMU features")
    
    print(f"Found {len(acoustic_files)} acoustic files in {acoustic_dir}")
    print(f"Found {len(imu_files)} IMU files in {imu_dir if imu_dir else 'N/A'}")
    
    if len(acoustic_files) == 0:
        raise ValueError(f"No acoustic CSV files found in {acoustic_dir}")
    
    # Use acoustic files count for processing
    n_pairs = len(acoustic_files)
    
    # Initialize lists for features and targets
    all_features = []
    wear_states = []
    
    # Process each pair
    for i in range(n_pairs):
        acoustic_file = acoustic_files[i]
        
        # Try to find matching IMU file
        if imu_files:
            # Try to match by filename pattern
            acoustic_basename = os.path.basename(acoustic_file)
            imu_file = None
            
            # First, try exact name matching
            for imu_candidate in imu_files:
                if os.path.basename(imu_candidate) == acoustic_basename:
                    imu_file = imu_candidate
                    break
            
            # If not found, try index-based matching
            if not imu_file and i < len(imu_files):
                imu_file = imu_files[i]
                
            # If still not found, use the next available
            if not imu_file and imu_files:
                imu_file = imu_files[min(i, len(imu_files)-1)]
        else:
            imu_file = None
        
        print(f"Processing pair {i+1}/{n_pairs}: ")
        print(f"  - Acoustic: {os.path.basename(acoustic_file)}")
        print(f"  - IMU:      {os.path.basename(imu_file) if imu_file else 'DUMMY DATA'}")
        
        try:
            # Extract sample number from filename if possible
            sample_number = i + 1  # Default to index + 1
            
            # Try to extract number from filename
            import re
            filename = os.path.basename(acoustic_file)
            match = re.search(r'(\d+)', filename)
            if match:
                sample_number = int(match.group(1))
            
            # Get wear state from failure map
            if failure_map and sample_number in failure_map:
                wear_state = failure_map[sample_number]
                print(f"  - Sample {sample_number}: Using failure map → {wear_state:.2%} wear")
            else:
                # Fallback to linear assumption if no map provided
                wear_state = i / (n_pairs - 1) if n_pairs > 1 else 0
                print(f"  - Sample {sample_number}: Using linear assumption → {wear_state:.2%} wear")
            
            # Extract features
            if imu_file:
                X, _, features = prepare_data_for_ml(acoustic_file, imu_file, tool_lifetime)
            else:
                print("  - No IMU data found, creating dummy features")
                # Create dummy IMU features when no IMU file is available
                acoustic_df = read_csv_file(acoustic_file, '2')
                stft_result, sr = acoustic_processing(acoustic_df)
                acoustic_features = extract_features_from_stft(stft_result, sr)
                
                # Add dummy IMU features
                dummy_imu_features = _get_default_imu_features()
                features = {**acoustic_features, **dummy_imu_features}
                X = pd.DataFrame([features])
            
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
    
    print(f"\nCreated dataset with {X.shape[0]} samples and {X.shape[1]} features")
    print("\nFinal wear labels:")
    for i, wear in enumerate(y):
        print(f"Sample {i+1}: {wear:.2%} wear")
    
    # Apply enhanced preprocessing
    print(f"\nApplying enhanced preprocessing...")
    X_processed, scaler, feature_selector, preprocessing_info = advanced_data_preprocessing(
        X, y, remove_outliers=True, handle_correlations=True, verbose=True
    )
    
    print(f"\nPreprocessing summary:")
    for key, value in preprocessing_info.items():
        print(f"  {key}: {value}")
    
    return X_processed, y, scaler, feature_selector, preprocessing_info

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
    
    # Example enhanced preprocessing
    X_processed, scaler, feature_selector, preprocessing_info = advanced_data_preprocessing(
        X, y, verbose=True
    )
    print(f"\nAfter preprocessing: {X_processed.shape}")
    print("Preprocessing info:", preprocessing_info)
