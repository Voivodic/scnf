#!/bin/bash

# -----------------------------------------------------------------------------
# This script runs the full pipeline for generating data, training the SCNF
# model, sampling from the trained model, and plotting the results.
#
# Usage:
#   ./run_unconditional.sh [options]
#
# Options:
#   --no-generate    Skip the data generation step.
#   --no-train       Skip the training step.
#   --no-sample      Skip the sampling step.
#   --no-plot        Skip the comparison plotting step.
#   --no-loss-plot   Skip the loss plotting step.
#
# You can customize the parameters below before running the script.
# -----------------------------------------------------------------------------

# --- Configuration Variables ---

# Training Parameters
CONFIG_FILE="config_unconditional.json"

# --- Argument Parsing ---

RUN_GENERATE=true
RUN_TRAIN=true
RUN_SAMPLE=true
RUN_PLOT=true
RUN_LOSS_PLOT=true

for arg in "$@"
do
    case $arg in
        --no-generate)
        RUN_GENERATE=false
        shift
        ;;
        --no-train)
        RUN_TRAIN=false
        shift
        ;;
        --no-sample)
        RUN_SAMPLE=false
        shift
        ;;
        --no-sample-plot)
        RUN_PLOT=false
        shift
        ;;
        --no-loss-plot)
        RUN_LOSS_PLOT=false
        shift
        ;;
    esac
done

# --- Pipeline Execution ---

if [ "$RUN_GENERATE" = true ]; then
    # Step 1: Generate the synthetic data
    echo "------------------------------------"
    echo "STEP 1: Generating synthetic data..."
    echo "------------------------------------"
    echo "Using configuration file: $CONFIG_FILE"
    echo ""

    python generate_theta.py \
        --config "$CONFIG_FILE" 

    # Check if the data generation command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Data generation failed. Exiting."
        exit 1
    fi

    echo "Data generation completed successfully."
    echo ""
fi

if [ "$RUN_TRAIN" = true ]; then
    # Step 2: Train the CNF model
    echo "----------------------------------"
    echo "STEP 2: Training the CNF model..."
    echo "----------------------------------"
    echo "Using configuration file: $CONFIG_FILE"
    echo ""

    python train.py \
        --config "$CONFIG_FILE"

    # Check if the training command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Model training failed. Exiting."
        exit 1
    fi

    echo "Model training completed successfully."
    echo ""
fi

if [ "$RUN_SAMPLE" = true ]; then
    # Step 3: Sample from the trained model
    echo "-------------------------------------------"
    echo "STEP 3: Sampling from the trained model..."
    echo "-------------------------------------------"
    echo "Using configuration file: $CONFIG_FILE"
    echo ""

    python sample.py \
        --config "$CONFIG_FILE"

    # Check if the sampling command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Model sampling failed. Exiting."
        exit 1
    fi

    echo "Model sampling completed successfully."
    echo ""
fi

if [ "$RUN_PLOT" = true ]; then
    # Step 4: Plot the comparison
    echo "------------------------------------"
    echo "STEP 4: Plotting the comparison..."
    echo "------------------------------------"
    echo "Using configuration file: $CONFIG_FILE"
    echo ""

    python plot_samples.py \
        --config "$CONFIG_FILE"

    # Check if the plotting command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Plotting failed. Exiting."
        exit 1
    fi

    echo "Plotting completed successfully."
    echo ""
fi

if [ "$RUN_LOSS_PLOT" = true ]; then
    # Step 5: Plot the loss
    echo "------------------------------------"
    echo "STEP 5: Plotting the loss..."
    echo "------------------------------------"
    echo "Using configuration file: $CONFIG_FILE"
    echo ""

    python plot_losses.py \
        --config "$CONFIG_FILE"

    # Check if the plotting command was successful
    if [ $? -ne 0 ]; then
        echo "Error: Loss plotting failed. Exiting."
        exit 1
    fi

    echo "Loss plotting completed successfully."
    echo ""
fi

echo "Pipeline finished."
