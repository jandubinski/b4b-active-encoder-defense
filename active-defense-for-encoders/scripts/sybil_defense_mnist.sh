#!/bin/bash

# Change directory to the location of the Python script
cd ..

#transforms=("affine" "shuffle_pad" "affine_shuffle_pad" "binary")
benchmark_datasets=(
"mnist_embeddings" "mnist_embeddings_shuffle_pad" "mnist_embeddings_affine_shuffle_pad" "mnist_embeddings_binary"
)

# Loop through the list of benchmark datasets
for benchmark in "${benchmark_datasets[@]}"; do
    # Loop through the list of additional arguments and run the Python script with each argument and benchmark dataset
    output_dir="./results/${benchmark}"
    echo "Output directory: $output_dir"
    python src/sybil_defense.py benchmark=$benchmark output_dir=$output_dir
    echo "Finished running with output:  $output_dir for benchmark: $benchmark"
done
# Exit with a status code indicating successful execution (0) or an error (non-zero)
exit 0
