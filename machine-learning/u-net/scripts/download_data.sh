#!/bin/bash

# Define the URL and output file name
DATASET_URL="http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
OUTPUT_FILE="ADEChallengeData2016.zip"
DESTINATION_DIR="./data"

# Create the destination directory if it doesn't exist
mkdir -p "$DESTINATION_DIR"

# Download the dataset
echo "Downloading ADE20K dataset..."
curl -L "$DATASET_URL" -o "$OUTPUT_FILE"
echo "Download completed."

# Extract the dataset
echo "Extracting the dataset..."
TEMP_DIR=$(mktemp -d)
unzip -q "$OUTPUT_FILE" -d "$TEMP_DIR"
mv "$TEMP_DIR/ADEChallengeData2016/"* "$DESTINATION_DIR/"
rm -rf "$TEMP_DIR"
echo "Extraction completed."

# Remove the zip file
echo "Removing zip file..."
rm -f "$OUTPUT_FILE"
echo "Zip file removed."


