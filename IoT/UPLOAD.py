import os
from google.cloud import storage

#Text colors
CYELLOW = '\033[93m'
CGREEN = '\033[92m'
CBLUE = '\033[94m'
CRED = '\033[91m'
CBLINK = '\033[5m'
CEND = '\033[0m'

def upload_to_gcs(bucket_name, source_file_path, destination_blob_name, credentials_file, filename):

    # Initialize the Google Cloud Storage client with the credentials
    storage_client = storage.Client.from_service_account_json(credentials_file)

    # Get the target bucket
    bucket = storage_client.bucket(bucket_name)

    # Upload the file to the bucket
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_path)

    print(CBLUE + f"File {source_file_path} uploaded to gs://{bucket_name}/audio/data/{destination_blob_name}" + CEND)

    if os.path.exists(source_file_path):
        os.remove(source_file_path)
        os.remove(source_file_path.replace('_encrypted',''))
        print(CRED + filename + " removed" + CEND)
    else:
        print("File doesn't exist")
