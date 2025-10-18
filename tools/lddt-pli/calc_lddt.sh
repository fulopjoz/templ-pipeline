#!/bin/bash

# Check if correct number of arguments provided
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <input_csv> <output_csv>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$2"

# Start Docker container if not already running
if ! sudo docker ps -a --format '{{.Names}}' | grep -q '^ost$'; then
  sudo docker run -dit --name ost -v "$(pwd)":/home registry.scicore.unibas.ch/schwede/openstructure:latest
fi

# Initialize output file with header
echo "prot,lig,score" > "$OUTPUT_FILE"

# Read input file line by line (skip header)
{
  read   # skip header
  while IFS=',' read -r model ref wrong; do
    [[ -z "$model" || -z "$ref" || -z "$wrong" ]] && continue

    echo "Processing: $model, $ref, $wrong"
    sudo docker exec ost ost pli_lddt.py "$model" "$ref" "$wrong" "$OUTPUT_FILE"
  done
} < "$INPUT_FILE"

# Clean up Docker container
sudo docker rm -f ost

