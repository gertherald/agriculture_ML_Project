#!/bin/bash

# Get the current working directory
current_dir=$(pwd)

# Append /src to the current directory
target_dir="$current_dir/src"

# Change to the target directory
cd "$target_dir" || { echo "Directory not found!"; exit 1; }

# Run the main.py script located in the 'src' folder
python main.py
