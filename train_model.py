import os
import sys
from tool_wear_predictor import train_model_from_recordings

def main():
    # Get the directory path from user
    data_dir = input("Enter the path to your data directory (containing all acoustic files): ")
    
    if not os.path.exists(data_dir):
        print(f"Error: Directory {data_dir} does not exist")
        return
    
    # Select tool type
    print("\nSelect tool type:")
    print("1. Reamer (200 min lifetime)")
    print("2. E-mill (100 min lifetime)")
    
    tool_type = input("Enter your choice (1/2): ")
    
    if tool_type == "1":
        tool_lifetime = 12000  # 200 minutes
        tool_name = "reamer"
    elif tool_type == "2":
        tool_lifetime = 6000   # 100 minutes
        tool_name = "emill"
    else:
        print("Invalid selection")
        return
    
    # Create output directory
    output_dir = os.path.join("models", tool_name)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nTraining model on all files in {data_dir}")
    print(f"Tool lifetime: {tool_lifetime} seconds")
    
    # Train the model
    predictor = train_model_from_recordings(
        data_dir,
        output_dir=output_dir,
        tool_lifetime=tool_lifetime
    )
    
    if predictor and predictor.model_loaded:
        print(f"\nSuccess! Model trained and saved to {output_dir}")
    else:
        print("\nError: Failed to train model")

if __name__ == "__main__":
    main()