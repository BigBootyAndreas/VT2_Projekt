import os
from fernetfile import open as fernet_open  # assuming you're using fernetfile
from cryptography.fernet import Fernet

# Key
key = b'SUPER SECRET ENCRYPTION KEY' # Placeholder

# Folders
base_dir = os.path.dirname(os.path.abspath(__file__))
input_folder = os.path.join(base_dir, 'encrypted_files/reamer/')
output_folder = os.path.join(base_dir, 'imu_decrypted_files')

# Create output folder
os.makedirs(output_folder, exist_ok=True)

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):

        dc_source = os.path.join(input_folder, filename)
        base_name = os.path.splitext(filename)[0]
        base_name = base_name.replace("_encrypted", "")  # Remove encrypted prefix
        dc_destination = os.path.join(output_folder, f'{base_name}.csv')

        file_size = os.path.getsize(dc_source)
        estimated_decrypted_size = int(file_size * 0.749) # Multiplier added due to decrypted files being smaller (For the procentage reading)

        with fernet_open(dc_source, mode='rb', fernet_key=key) as fin, open(dc_destination, 'wb') as fout:
            total_bytes = 0
            chunk_size = 65536

            while True:
                data = fin.read(chunk_size)
                if not data:
                    break
                fout.write(data)
                total_bytes += len(data)
                percent = (total_bytes / estimated_decrypted_size) * 100
                print(f"Decrypting: {percent:.1f}% ({total_bytes}/{estimated_decrypted_size} bytes)", end='\r')

        print("\nDecryption complete!")

        if os.path.exists(dc_source):
            os.remove(dc_source)
            print("Removed")
        else:
            print("File doesn't exist")
