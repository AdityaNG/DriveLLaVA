#!/bin/bash

# Check if two arguments are given
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_folder> <output_folder>"
    exit 1
fi

# Assign arguments to variables
input_folder=$1
output_folder=$2

# Ensure the output directory exists
mkdir -p "$output_folder"

# Iterate over zip files in the input directory
for zip_file in "$input_folder"/*.zip; do
    # Extract the filename without the path and extension
    base_name=$(basename "$zip_file" .zip)

    # Special case for val.zip and pose_val.zip
    if [ "$base_name" == "val" ] || [ "$base_name" == "pose_val" ]; then
        mkdir -p "$output_folder/$base_name"
        unzip -o "$zip_file" -d "$output_folder/$base_name"
    else
        # Create a directory with the same name as the zip file in the output directory
        mkdir -p "$output_folder/$base_name"
        unzip -o "$zip_file" -d "$output_folder/$base_name"
    fi
done

echo "Extraction complete."
