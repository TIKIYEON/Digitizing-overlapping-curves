#!/bin/bash

# Define the expected source directory (external HDD)
EXPECTED_SOURCE="/run/media/tikki/Seagate Hub"
# Define the target directory where .las files will be copied
TARGET_DIR="/home/tikki/Documents/School/BachelorProject/Digitizing-overlapping-curves/tif-files"

# Function to find the actual mount point of the external HDD
find_mount_point() {
    local drive_name="Seagate Hub"  # Adjust this if your drive has a different label or name
    MOUNT_POINT=$(lsblk -o NAME,LABEL,MOUNTPOINT | grep -i "$drive_name" | tail -n 1 | awk '{print $NF}')
    if [ -z "$MOUNT_POINT" ] || [ "$MOUNT_POINT" = "-" ]; then
        echo "Error: Could not find mount point for drive '$drive_name'. Is it plugged in and mounted?"
        exit 1
    fi
    echo "Found mount point: $MOUNT_POINT"
    echo "$MOUNT_POINT"
}

# Check if the expected source directory exists, or find the actual mount point
if [ -d "$EXPECTED_SOURCE" ]; then
    SOURCE_DIR="$EXPECTED_SOURCE"
    echo "Using source directory: $SOURCE_DIR"
else
    echo "Expected source directory '$EXPECTED_SOURCE' not found. Attempting to find mount point..."
    SOURCE_DIR=$(find_mount_point)
    if [ $? -ne 0 ]; then
        exit 1
    fi
fi

# Create target directory if it doesn't exist
mkdir -p "$TARGET_DIR"

# Temporary store for eligible .tif files
TEMP_FILE_LIST=$(mktemp)

# Find all .tif files
find "$SOURCE_DIR" -type f -name "*.tif" | while read -r tif_file; do
    tif_dir=$(dirname "$tif_file")
    parent_dir=$(dirname "$tif_dir")

    # Check if there are any .las files in the current directory
    if find "$tif_dir" -maxdepth 1 -name "*.las" -type f | grep -q .; then
        continue # Skip to the next file if .las files are found
    fi

    # Check if there are any .las files in sibling directories
    las_in_siblings=0
    for sibling_dir in "$parent_dir"/*; do
        if [ -d "$sibling_dir" ] && [ "$sibling_dir" != "$tif_dir" ]; then
            if find "$sibling_dir" -maxdepth 1 -name "*.las" -type f | grep -q .; then
                las_in_siblings=1
                break
            fi
        fi
    done

    if [ "$las_in_siblings" -gt 0 ]; then
        continue # Skip to the next file if .las files are found in sibling directories
    fi

    # Current .tif file is eligible for copying, add it to the list
    echo "$tif_file" >> "$TEMP_FILE_LIST"
done

# Count the number of eligible .tif files
FILE_COUNT=$(wc -l < "$TEMP_FILE_LIST")

if [ "$FILE_COUNT" -eq 0 ]; then
    echo "No eligible .tif files found in $SOURCE_DIR."
    rm "$TEMP_FILE_LIST"
    exit 1
fi

echo "Found $FILE_COUNT eligible .tif files in $SOURCE_DIR."

echo "Calculating the total size of up to 1000 eligible .tif files..."
# Use find to locate 1000 eligible .tif files and calculate their total size
TOTAL_SIZE=$(head -n 1000 "$TEMP_FILE_LIST" | xargs du -cb 2>/dev/null | grep total$ | cut -f1)

echo "Total size of the first 1000 .tif files: $TOTAL_SIZE"
echo "Proceed with copying the first 1000 .tif files? (y/n)"
read -r response

if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Copying the first 1000 .tif files to $TARGET_DIR..."

    # Copy only the first 1000 .tif files
    head -n 1000 "$TEMP_FILE_LIST" | while read -r file; do
        cp "$file" "$TARGET_DIR" 2>/dev/null
        echo "Copied: $file"
    done

    echo "Copy completed. Verifying..."

    # Verify the copy by checking the number of files
    FILES_COPIED=$(ls "$TARGET_DIR"/*.tif 2>/dev/null | wc -l)
    if [ "$FILE_COUNT" -gt 1000 ]; then
        EXPECTED_COUNT=1000
    else
        EXPECTED_COUNT=$FILE_COUNT
    fi

    if [ "$FILES_COPIED" -eq "$EXPECTED_COUNT" ]; then
        echo "All $FILES_COPIED files copied successfully."
    elif [ "$FILES_COPIED" -gt 0 ]; then
        echo "Only $FILES_COPIED files copied. Check for errors."
    else
        echo "No files were copied. Check for errors."
    fi
else
    echo "Copy cancelled."
    exit 0
fi
