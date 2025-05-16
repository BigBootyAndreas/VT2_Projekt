import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import os
import sys
import glob

# Ensure current directory is in path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)


from File_reader import read_csv_file
from Acoustic_data import acoustic_processing
from data_preparation import prepare_data_for_ml, scale_features

class ToolWearPredictor:
    """
    Tool wear prediction system that integrates the rest
    """
    def __init__(self, model_path=None, scaler_path=None, features_path=None):
        """
        Initialize the predictor
        
        Parameters:
        -----------
        model_path : str
            Path to the saved model file
        scaler_path : str
            Path to the saved scaler file
        features_path : str
            Path to the saved feature list file
        """
        # Load model and preprocessing components if provided
        self.model_loaded = False
        if model_path and os.path.exists(model_path) and scaler_path and features_path:
            self.load_model(model_path, scaler_path, features_path)
        
        # Tool life parameters
        self.tool_lifetime = 3600  # seconds
        self.replacement_threshold = 0.8  # 80% wear threshold
        
        # Results storage
        self.predictions = []
        self.analyzed_files = []
    
    def load_model(self, model_path, scaler_path, features_path):
        """Load model and preprocessing components"""
        try:
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.feature_cols = joblib.load(features_path)
            self.model_loaded = True
            print(f"Model loaded from {model_path}")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def set_tool_parameters(self, lifetime_seconds, replacement_threshold=0.8):
        """Set tool lifetime and replacement threshold"""
        self.tool_lifetime = lifetime_seconds
        self.replacement_threshold = replacement_threshold
        print(f"Tool lifetime set to {lifetime_seconds} seconds")
        print(f"Replacement threshold set to {replacement_threshold*100}%")
    
    def analyze_recording(self, acoustic_file, imu_file, current_time=None):
        """
        Analyze recorded data and predict tool wear
        
        Parameters:
        -----------
        acoustic_file : str
            Path to acoustic data CSV file
        imu_file : str
            Path to IMU data CSV file
        current_time : float
            Current time in seconds (for tool wear calculation)
            
        Returns:
        --------
        predicted_wear : float
            Predicted tool wear (0-1)
        remaining_time : float
            Estimated remaining time in seconds
        needs_replacement : bool
            Whether tool needs replacement soon
        """
        try:
            # Use the existing File_reader.py to load data
            print(f"Loading acoustic data from {acoustic_file}")
            acoustic_df = read_csv_file(acoustic_file, '2')  # '2' for acoustic data
            
            print(f"Loading IMU data from {imu_file}")
            imu_df = read_csv_file(imu_file, '1')  # '1' for IMU data
            
            if acoustic_df is None or imu_df is None:
                print("Error: Failed to load data files")
                return None, None, False
            
            # Extract features using data_preparation (which uses the existing processing code)
            X, _, _ = prepare_data_for_ml(acoustic_file, imu_file, self.tool_lifetime)
            
            if not self.model_loaded:
                print("Model not loaded. Cannot predict tool wear.")
                return None, None, False
            
            # Check if we have all required features
            missing_features = [f for f in self.feature_cols if f not in X.columns]
            if missing_features:
                print(f"Warning: Missing features: {missing_features}")
                # For missing features, add zeros
                for f in missing_features:
                    X[f] = 0
            
            # Select only features used by the model
            X = X[self.feature_cols]
            
            # Scale features
            X_scaled, _ = scale_features(X, self.scaler)
            
            # Predict tool wear
            predictions = self.model.predict(X_scaled)
            mean_prediction = predictions[0]  # Single prediction
            
            # Calculate remaining time
            if current_time is None:
                # If no current time provided, estimate based on prediction
                elapsed_time = mean_prediction * self.tool_lifetime
            else:
                elapsed_time = current_time
                
            remaining_time = self.tool_lifetime * (1 - mean_prediction)
            needs_replacement = mean_prediction >= self.replacement_threshold
            
            # Store result
            result = {
                'acoustic_file': os.path.basename(acoustic_file),
                'imu_file': os.path.basename(imu_file),
                'predicted_wear': mean_prediction,
                'remaining_time': remaining_time,
                'needs_replacement': needs_replacement,
                'elapsed_time': elapsed_time,
                'timestamp': pd.Timestamp.now()
            }
            self.predictions.append(result)
            self.analyzed_files.append((acoustic_file, imu_file))
            
            print(f"Analysis complete:")
            print(f"- Predicted wear: {mean_prediction:.2%}")
            print(f"- Estimated remaining time: {self._format_time(remaining_time)}")
            print(f"- Needs replacement: {'YES' if needs_replacement else 'No'}")
            
            return mean_prediction, remaining_time, needs_replacement
            
        except Exception as e:
            print(f"Error analyzing recording: {e}")
            import traceback
            traceback.print_exc()
            return None, None, False
    
    def analyze_directory(self, directory_path, acoustic_pattern="acoustic*.csv", imu_pattern="IMU*.csv"):
        """
        Analyze all matching files in a directory
        
        Parameters:
        -----------
        directory_path : str
            Path to directory containing recordings
        acoustic_pattern : str
            Pattern to match acoustic files
        imu_pattern : str
            Pattern to match IMU files
            
        Returns:
        --------
        results : list
            List of analysis results
        """
        # Find all acoustic and IMU files
        acoustic_files = sorted(glob.glob(os.path.join(directory_path, acoustic_pattern)))
        imu_files = sorted(glob.glob(os.path.join(directory_path, imu_pattern)))
        
        # If no files found with the primary pattern, try alternative patterns
        if not acoustic_files:
            acoustic_files = sorted(glob.glob(os.path.join(directory_path, "*_ac*.csv")))
        if not imu_files:
            imu_files = sorted(glob.glob(os.path.join(directory_path, "*_imu*.csv")))
        
        print(f"Found {len(acoustic_files)} acoustic files and {len(imu_files)} IMU files")
        
        if len(acoustic_files) != len(imu_files):
            print(f"Warning: Number of acoustic files ({len(acoustic_files)}) " 
                  f"doesn't match number of IMU files ({len(imu_files)})")
        
        # Use the minimum number of files
        n_pairs = min(len(acoustic_files), len(imu_files))
        
        results = []
        for i in range(n_pairs):
            acoustic_file = acoustic_files[i]
            imu_file = imu_files[i]
            
            print(f"Analyzing recording {i+1}/{n_pairs}: {os.path.basename(acoustic_file)}")
            wear, remaining, needs_replacement = self.analyze_recording(acoustic_file, imu_file)
            
            if wear is not None:
                results.append({
                    'acoustic_file': acoustic_file,
                    'imu_file': imu_file,
                    'wear': wear,
                    'remaining_time': remaining,
                    'needs_replacement': needs_replacement
                })
        
        return results
    
    def plot_wear_history(self):
        """Plot wear history from analyzed files"""
        if not self.predictions:
            print("No analysis history to plot")
            return
        
        wear_values = [p['predicted_wear'] for p in self.predictions]
        timestamps = [p['timestamp'] for p in self.predictions]
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, wear_values, 'o-', linewidth=2)
        plt.axhline(y=self.replacement_threshold, color='r', linestyle='--', 
                   label=f'Replacement Threshold ({self.replacement_threshold:.0%})')
        
        plt.title('Tool Wear History')
        plt.xlabel('Analysis Time')
        plt.ylabel('Predicted Tool Wear')
        plt.grid(True)
        plt.legend()
        
        # Format y-axis as percentage
        plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1.0))
        
        plt.tight_layout()
        plt.show()
    
    def save_results(self, filename="tool_wear_analysis.csv"):
        """Save analysis results to CSV file"""
        if not self.predictions:
            print("No results to save")
            return
        
        # Convert to DataFrame
        results_df = pd.DataFrame(self.predictions)
        
        # Save to CSV
        results_df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")
        
        return results_df
    
    def _format_time(self, seconds):
        """Format seconds as hours, minutes, seconds string"""
        hours, remainder = divmod(int(seconds), 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {seconds}s"
        elif minutes > 0:
            return f"{minutes}m {seconds}s"
        else:
            return f"{seconds}s"


def evaluate_progressive_model(model_dir="models/drill_progressive", data_dir=None):
    """
    Evaluate progressive wear model on held-out test data (samples 18-23)
    """
    print("=== Evaluating Progressive Drill Model ===")
    
    # Import needed functions
    from progressive_wear_dataset import collect_test_dataset
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    if not os.path.exists(model_dir):
        print(f"Model directory not found: {model_dir}")
        print("Train the model first using train_drill_model.py")
        return None
    
    if data_dir is None:
        data_dir = "C:\\Users\\User\\Documents\\AAU 8. semester\\Projekt\\Data"
    
    # Load test failure map
    test_failure_map_path = os.path.join(model_dir, "test_failure_map.pkl")
    if not os.path.exists(test_failure_map_path):
        print("Test failure map not found. Train the model first.")
        return None
    
    # Number of windows
    n_windows = int(input("Enter number of windows used in training (default 4): ") or "4")
    
    # Collect test data (automatically filters for samples 18-23)
    print("\nCollecting test data...")
    X_test, y_test, info_test = collect_test_dataset(data_dir, n_windows)
    
    print(f"Test dataset: {len(X_test)} windows from samples 18-23")
    
    # Find trained models
    model_files = [f for f in os.listdir(model_dir) if f.endswith('_model.pkl')]
    
    if not model_files:
        print("No trained models found")
        return None
    
    print("\nAvailable models:")
    for i, model_file in enumerate(model_files):
        print(f"{i+1}. {model_file}")
    
    # Evaluate each model
    results = {}
    
    for model_file in model_files:
        base_name = model_file.replace('_model.pkl', '')
        print(f"\nEvaluating {base_name}...")
        
        # Create a temporary predictor for this model
        model_path = os.path.join(model_dir, f"{base_name}_model.pkl")
        scaler_path = os.path.join(model_dir, f"{base_name}_scaler.pkl")
        features_path = os.path.join(model_dir, f"{base_name}_features.pkl")
        
        temp_predictor = ToolWearPredictor(model_path, scaler_path, features_path)
        
        if not temp_predictor.model_loaded:
            print(f"  Error loading {base_name}")
            continue
        
        # Handle missing features
        X_test_model = X_test.copy()
        missing_features = [f for f in temp_predictor.feature_cols if f not in X_test_model.columns]
        for feature in missing_features:
            X_test_model[feature] = 0
            print(f"  Warning: Missing feature {feature}, filled with 0")
        
        # Scale and predict
        X_test_scaled = temp_predictor.scaler.transform(X_test_model[temp_predictor.feature_cols])
        y_pred = temp_predictor.model.predict(X_test_scaled)
        
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
    
    if not results:
        print("No models could be evaluated")
        return None
    
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
    
    return results


def train_model_from_recordings(data_dir, output_dir="models", tool_lifetime=3600):
    """
    Train a new model from a directory of recordings
    
    Parameters:
    -----------
    data_dir : str
        Directory containing acoustic and IMU recordings
    output_dir : str
        Directory to save trained model and associated files
    tool_lifetime : float
        Tool lifetime in seconds
        
    Returns:
    --------
    predictor : ToolWearPredictor
        Initialized predictor with trained model
    """
    from ml_models import run_ml_pipeline
    
    # Run ML pipeline
    print(f"Training model using data in {data_dir}")
    best_model_files = run_ml_pipeline(data_dir, tool_lifetime, output_dir)
    
    if best_model_files:
        model_path, scaler_path, features_path = best_model_files
        
        # Initialize predictor with trained model
        predictor = ToolWearPredictor(model_path, scaler_path, features_path)
        predictor.set_tool_parameters(tool_lifetime)
        
        return predictor
    else:
        print("Error: Failed to train model")
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Tool Wear Prediction System")
    parser.add_argument("--analyze", help="Path to directory containing recordings to analyze")
    parser.add_argument("--train", help="Path to directory containing recordings for training")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate progressive model on test data")
    parser.add_argument("--model", help="Path to trained model file")
    parser.add_argument("--scaler", help="Path to scaler file")
    parser.add_argument("--features", help="Path to feature list file")
    parser.add_argument("--lifetime", type=float, default=3600, 
                       help="Tool lifetime in seconds (default: 3600)")
    
    args = parser.parse_args()
    
    if args.evaluate:
        # Evaluate progressive model
        evaluate_progressive_model()
        
    elif args.train:
        # Train new model
        predictor = train_model_from_recordings(args.train, tool_lifetime=args.lifetime)
        
        if args.analyze and predictor:
            # Analyze recordings with newly trained model
            predictor.analyze_directory(args.analyze)
            predictor.plot_wear_history()
            predictor.save_results()
            
    elif args.analyze and args.model and args.scaler and args.features:
        # Load existing model and analyze recordings
        predictor = ToolWearPredictor(args.model, args.scaler, args.features)
        predictor.set_tool_parameters(args.lifetime)
        predictor.analyze_directory(args.analyze)
        predictor.plot_wear_history()
        predictor.save_results()
        
    else:
        print("Choose an option:")
        print("1. Evaluate progressive model")
        print("2. Train new model")
        print("3. Analyze recordings with existing model")
        
        choice = input("Enter choice: ")
        if choice == "1":
            evaluate_progressive_model()
        elif choice == "2":
            data_dir = "C:\\Users\\User\\Documents\\AAU 8. semester\\Projekt\\Data"
            predictor = train_model_from_recordings(data_dir)
        elif choice == "3":
            model_path = input("Enter path to model file: ")
            scaler_path = input("Enter path to scaler file: ")
            features_path = input("Enter path to features file: ")
            data_dir = "C:\\Users\\User\\Documents\\AAU 8. semester\\Projekt\\Data"
            
            predictor = ToolWearPredictor(model_path, scaler_path, features_path)
            predictor.analyze_directory(data_dir)
            predictor.plot_wear_history()
            predictor.save_results()