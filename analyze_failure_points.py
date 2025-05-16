import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from File_reader import read_csv_file
import os

def create_failure_analysis_plots(data_dir, output_dir="failure_analysis_plots"):
    """
    Create plots for each sample and save them as images for manual review
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find acoustic and IMU directories
    acoustic_dir = os.path.join(data_dir, "Acoustic Data")
    imu_dir = os.path.join(data_dir, "IMU Data")
    
    # Find all files
    acoustic_files = []
    imu_files = []
    
    for root, dirs, files in os.walk(acoustic_dir):
        acoustic_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
    
    for root, dirs, files in os.walk(imu_dir):
        imu_files.extend([os.path.join(root, f) for f in files if f.endswith('.csv')])
    
    acoustic_files = sorted(acoustic_files)
    imu_files = sorted(imu_files)
    
    print(f"Found {len(acoustic_files)} acoustic files")
    print(f"Found {len(imu_files)} IMU files")
    
    for i, acoustic_file in enumerate(acoustic_files):
        imu_file = imu_files[i] if i < len(imu_files) else None
        
        # Extract sample number
        import re
        match = re.search(r'(\d+)', os.path.basename(acoustic_file))
        sample_number = int(match.group(1)) if match else i + 1
        
        print(f"Creating plot for Sample {sample_number}...")
        
        # Read data
        acoustic_df = read_csv_file(acoustic_file, '2')
        imu_df = read_csv_file(imu_file, '1') if imu_file else None
        
        # Create time arrays
        if "Time" in acoustic_df.columns:
            acoustic_time = acoustic_df["Time"].values
        else:
            acoustic_time = np.arange(len(acoustic_df)) / 1200
        
        acoustic_amplitude = acoustic_df["Amplitude"].values
        
        # Create plot
        fig, axes = plt.subplots(2 if imu_file else 1, 1, figsize=(15, 8))
        if not isinstance(axes, np.ndarray):
            axes = [axes]
        
        # Plot acoustic data
        axes[0].plot(acoustic_time, acoustic_amplitude, 'b-', linewidth=0.5)
        axes[0].set_title(f'Sample {sample_number} - Acoustic Signal')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True)
        
        # Add time markers
        max_time = acoustic_time[-1]
        for t in range(0, int(max_time), 10):
            axes[0].axvline(x=t, color='gray', linestyle='--', alpha=0.3)
            axes[0].text(t, axes[0].get_ylim()[1], f'{t}s', rotation=90, va='top')
        
        # Plot IMU data if available
        if imu_file and imu_df is not None:
            if "epoch" in imu_df.columns:
                imu_time = imu_df["epoch"].values
                if imu_time[0] > 1000000:
                    imu_time = imu_time - imu_time[0]
            else:
                imu_time = np.arange(len(imu_df)) / 400
            
            if "X (g)" in imu_df.columns:
                x_accel = imu_df["X (g)"].values
                y_accel = imu_df["Y (g)"].values
                z_accel = imu_df["Z (g)"].values
                imu_magnitude = np.sqrt(x_accel**2 + y_accel**2 + z_accel**2)
            else:
                imu_magnitude = np.zeros(len(imu_df))
            
            axes[1].plot(imu_time, imu_magnitude, 'r-', linewidth=0.5)
            axes[1].set_title(f'Sample {sample_number} - IMU Magnitude')
            axes[1].set_xlabel('Time (s)')
            axes[1].set_ylabel('Acceleration Magnitude (g)')
            axes[1].grid(True)
            
            # Add time markers
            #for t in range(0, int(imu_time[-1]), 10):
                #axes[1].axvline(x=t, color='gray', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        output_file = os.path.join(output_dir, f"sample_{sample_number}_analysis.png")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"  Saved: {output_file}")
    
    print(f"\nAll plots saved to {output_dir}")
    print("Review the plots and identify failure times manually.")

def collect_failure_times_manually():
    """
    Simple function to collect failure times manually
    """
    failure_times = {}
    
    print("\nBased on the analysis of the plots, enter the failure times:")
    print("(Enter 'none' if the tool didn't fail during that recording)")
    
    # Known samples that should have failures
    samples_with_failures = [2, 3, 5,6, 9, 10,]
    
    for sample in range(1, 24):
        if sample in samples_with_failures:
            while True:
                try:
                    user_input = input(f"Sample {sample} failure time (seconds): ")
                    if user_input.lower() == 'none':
                        break
                    failure_time = float(user_input)
                    failure_times[sample] = failure_time
                    break
                except ValueError:
                    print("Please enter a valid number or 'none'")
        else:
            print(f"Sample {sample}: No failure expected (skipping)")
    
    return failure_times

if __name__ == "__main__":
    data_dir = "C:\\Users\\User\\Documents\\AAU 8. semester\\Projekt\\Data"
    
    # Create plots
    print("Creating analysis plots...")
    create_failure_analysis_plots(data_dir)
    
    # Collect failure times manually
    failure_times = collect_failure_times_manually()
    
    print(f"\n{'='*50}")
    print("SUMMARY OF FAILURE TIMES")
    print(f"{'='*50}")
    for sample, time in failure_times.items():
        print(f"Sample {sample}: {time}s")
    
    # Save results
    if failure_times:
        with open("failure_times.txt", "w") as f:
            f.write("Failure Times Analysis Results\n")
            f.write("=" * 30 + "\n")
            for sample, time in failure_times.items():
                f.write(f"Sample {sample}: {time}s\n")
        print(f"\nResults saved to failure_times.txt")