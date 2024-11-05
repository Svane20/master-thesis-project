import os
import requests
import zipfile

from constants.directories import DATA_DIRECTORY, IMAGE_DIRECTORY

DATASET_URL = "http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"

def download_and_extract_ade20k(dataset_url=DATASET_URL, data_dir=DATA_DIRECTORY):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    zip_path = os.path.join(data_dir, "ADEChallengeData2016.zip")

    # Download the dataset if not already downloaded
    if not os.path.exists(zip_path):
        print("Downloading ADE20K dataset...")
        response = requests.get(dataset_url, stream=True)
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        print("Download completed.")

    # Extract the dataset if not already extracted
    if not os.path.exists(IMAGE_DIRECTORY):
        print("Extracting ADE20K dataset...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        print("Extraction completed.")

        # Remove the zip file
        os.remove(zip_path)

download_and_extract_ade20k()