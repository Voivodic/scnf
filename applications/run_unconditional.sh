#!/bin/bash

# -----------------------------------------------------------------------------
# This script runs the full pipeline for generating data and training the SCNF
# model.
#
# Usage:
#   ./run_unconditional.sh [options]
#
# Options:
#   --train-only  Skip the data generation step and only run training.
#
# You can customize the parameters below before running the script.
# -----------------------------------------------------------------------------

# --- Configuration Variables ---

# Data Generation Parameters
NUM_SAMPLES=1000
DIMENSIONS=3
POLY_ORDER=3
THETA_OUTPUT_PATH="data/thetas.hdf5"

# Training Parameters
TRAIN_CONFIG_FILE="config_unconditional.json"

# --- Argument Parsing ---

TRAIN_ONLY=false
if [[ "$1" == "--train-only" ]]; then
    TRAIN_ONLY=true
fi

# --- Pipeline Execution ---

if [ "$TRAIN_ONLY" = false ]; then
    # Step 1: Generate the synthetic data
    echo "------------------------------------"
    echo "STEP 1: Generating synthetic data..."
    echo "------------------------------------"
    echo "Number of samples: $NUM_SAMPLES"
    echo "Dimensions: $DIMENSIONS"
    echo "Polynomial order: $POLY_ORDER"
    echo "Output file: $THETA_OUTPUT_PATH"
    echo ""

    python generate_theta.py \
        --num-samples "$NUM_SAMPLES" \
        --dim "$DIMENSIONS" \
        --poly-order "$POLY_ORDER" \
        --output "$THETA_OUTPUT_PATH"

    # Check if the data generation command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Data generation failed. Exiting."
        exit 1
    fi

    echo "Data generation completed successfully."
    echo ""
fi

# Step 2: Train the CNF model
echo "----------------------------------"
echo "STEP 2: Training the CNF model..."
echo "----------------------------------"
echo "Number of samples: $NUM_SAMPLES"
echo "Dimensions: $DIMENSIONS"
echo "Using configuration file: $TRAIN_CONFIG_FILE"
echo ""

python train.py \
    --num-samples "$NUM_SAMPLES" \
    --dim "$DIMENSIONS" \
    --config "$TRAIN_CONFIG_FILE"

# Check if the training command was successful
if [ $? -ne 0 ]; then
    echo "Error: Model training failed. Exiting."
    exit 1
fi

echo "Model training completed successfully."
echo ""
echo "Pipeline finished."

