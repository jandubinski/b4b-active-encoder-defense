#!/bin/bash

echo "Running sybil_defense_cifar.sh..."
./sybil_defense_cifar.sh

echo "Running sybil_defense_mnist.sh..."
./sybil_defense_mnist.sh

echo "Running sybil_defense_imagenet.sh..."
./sybil_defense_imagenet.sh

echo "All scripts completed."
