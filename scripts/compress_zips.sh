#!/bin/bash

trap 'cleanup_and_exit' INT

cleanup_and_exit() {
    echo "Interrupt received. Cleaning up..."
    # Return to the original directory if needed
    popd > /dev/null 2>&1
    echo "Exiting due to user interruption."
    exit 1
}

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

# Iterate over directories in the input directory
for dir in "$input_folder"/*; do
    if [ -d "$dir" ]; then # Check if it is a directory
        # Extract the directory name without the path
        dir_name=$(basename "$dir")
        
        # Change to the parent directory of the target directory
        pushd "$(dirname "$dir")" > /dev/null

        # Compress the directory into a zip file with the same name, excluding specific patterns
        zip -r "$output_folder/$dir_name.zip" "$dir_name" -x "data_*_to_*/*.npy" -x "data_*_to_*/*.json"
        
        # Return to the original directory
        popd > /dev/null
    fi
done

echo "Compression complete."
