#!/bin/bash

# Cause the script to exit on any errors
# Reference: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euo pipefail

# Initialize conda
source "a2oc/scripts/_common/init_conda.sh"

# Activate the conda environment
conda activate "$CONDA_ENV_NAME"

# Set the Theano flags
export THEANO_FLAGS="floatX=float32"

# Make sure the log directory exists
mkdir -p "$LOG_DIR"

# Make sure the model directory exists
mkdir -p "$MODEL_DIR"

# Make sure the root zips directory exists
mkdir -p "$ROOT_ZIPS_DIR"

# Print the Theano flags
echo "\$THEANO_FLAGS=$THEANO_FLAGS"
