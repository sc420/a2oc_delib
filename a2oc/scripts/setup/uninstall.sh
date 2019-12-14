#!/bin/bash

# Cause the script to exit on any errors
# Reference: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euo pipefail

# Set common variables
source "a2oc/scripts/_common/set_common_variables.sh"

# Initialize conda
source "a2oc/scripts/_common/init_conda.sh"

# Remove the conda environment
conda remove -n "$CONDA_ENV_NAME" -y --all
