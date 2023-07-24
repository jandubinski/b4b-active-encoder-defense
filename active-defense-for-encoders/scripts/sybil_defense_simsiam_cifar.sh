#!/bin/bash

# Change directory to the location of the Python script
cd ..

#transforms=("affine" "shuffle_pad" "affine_shuffle_pad" "binary")
benchmark_datasets=(
 "cifar10_simsiam_embeddings" "cifar10_simsiam_embeddings_shuffle_pad" "cifar10_simsiam_embeddings_affine_shuffle_pad" "cifar10_simsiam_embeddings_binary"
)

# Loop through the list of benchmark datasets
for benchmark in "${benchmark_datasets[@]}"; do
    # Loop through the list of additional arguments and run the Python script with each argument and benchmark dataset
    output_dir="./results/simsiam/${benchmark}"
    echo "Output directory: $output_dir"
    python src/sybil_defense_sim_siam.py benchmark=$benchmark output_dir=$output_dir ++benchmark.embeddings.transform.base_dim=2048 ++benchmark.embeddings.transform.pad_dim=2048
    echo "Finished running with output:  $output_dir for benchmark: $benchmark"
done
# Exit with a status code indicating successful execution (0) or an error (non-zero)
exit 0
