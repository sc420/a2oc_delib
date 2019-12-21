#!/bin/bash

# Cause the script to exit on any errors
# Reference: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euo pipefail

################################################################################
# Initialize
################################################################################

# Set the common variables
source "a2oc/scripts/_common/set_common_variables.sh"

# Initialize the training variables
source "a2oc/scripts/train/_common/init_training_variables.sh"

# Initialize the training environment
source "a2oc/scripts/train/_common/init_training_environment.sh"

################################################################################
# Set Fixed Training Arguments
################################################################################

# Set the total number of frames for training
export MAX_NUM_FRAMES=500000

# Set the number of options
export NUM_OPTIONS=8

################################################################################
# Start Training
################################################################################

# 1. Run training in a Tmux detached session
# 2. Zip the experimental results
tmux new -d -s "$TMUX_SESSION_NAME" \
    "THEANO_FLAGS=\"$THEANO_FLAGS\" \
    python a2oc/train.py \
    --env=$ENV_ID \
    --folder-name=$MODEL_DIR \
    --max-num-frames=$MAX_NUM_FRAMES \
    --num-options=$NUM_OPTIONS \
    --num-threads=16 \
    --seed=$SEED \
    > >(tee \"$LOG_STDOUT_PATH\") \
    2> >(tee \"$LOG_STDERR_PATH\" >&2) && \
    rm -f \"$ZIP_PATH\" && \
    zip -r \"$ZIP_PATH\" \"$LOG_DIR\" \"$MODEL_DIR\""
