#!/bin/bash

# Set source and target directories
source_dir="/d/ovf/ovf_laz/raw_data"
target_dir="/e/ovf/laz"

# Set batch size (500 files per archive)
batch_size=500

# Create the target directory if it doesn't exist
mkdir -p "$target_dir"

# Counter for zip files
zip_counter=1

# Initialize an empty array to store the current batch of files
batch_files=()

# Iterate over all .laz files in the source directory
for laz_file in "$source_dir"/*.laz; do
    # Add the current laz file to the batch array
    batch_files+=("$laz_file")

    # Check if the batch size is reached
    if [ ${#batch_files[@]} -eq "$batch_size" ]; then
        # Create a zip file for the current batch using 7z
        zip_filename="$target_dir/laz_batch_$zip_counter.7z"
        /c/Program\ Files/7-Zip/7z.exe a "$zip_filename" "${batch_files[@]}"
        echo "Created $zip_filename"

        # Clear the batch array for the next set of files
        batch_files=()

        # Increment the zip counter
        zip_counter=$((zip_counter + 1))
    fi
done

# Check if there are any remaining files in the batch after the loop
if [ ${#batch_files[@]} -gt 0 ]; then
    # Create a zip file for the remaining files using 7z
    zip_filename="$target_dir/laz_batch_$zip_counter.7z"
    /c/Program\ Files/7-Zip/7z.exe a "$zip_filename" "${batch_files[@]}"
    echo "Created $zip_filename"
fi

echo "Compression completed."
