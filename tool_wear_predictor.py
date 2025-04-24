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
    Tool wear prediction system that integrates with your existing processing code
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
    parser.add_argument("--model", help="Path to trained model file")
    parser.add_argument("--scaler", help="Path to scaler file")
    parser.add_argument("--features", help="Path to feature list file")
    parser.add_argument("--lifetime", type=float, default=3600, 
                       help="Tool lifetime in seconds (default: 3600)")
    
    args = parser.parse_args()
    
    if args.train:
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
        print("Please specify either --train or --analyze with --model, --scaler, and --features")