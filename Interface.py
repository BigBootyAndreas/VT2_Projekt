import os
import pandas as pd
from file_path import *
import test_reader




def main():
    print("Welcome to TCM system")
    print("Please enter your name")

    # Path til personen
    while True:
        person = input("Who is running the script: ")  # Get the name
        directory = path(person)  # Get the path

        if directory:  
            print(f"You selected {person}. Data directory: {directory}")
            print("Checking directory contents...")
            files = [f for f in os.listdir(directory) if f.endswith(('.txt', '.csv', '.xlsx'))]
            print("Files in directory:", files)
            # Call the text reading function from the imported script
            test_reader.read_text_file(directory)  
            break  

        else:
            print("Invalid user, please try again.")

if __name__ == "__main__":
    main()