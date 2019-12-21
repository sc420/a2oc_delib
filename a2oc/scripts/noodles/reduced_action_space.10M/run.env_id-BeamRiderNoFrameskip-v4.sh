#!/bin/bash

# Cause the script to exit on any errors
# Reference: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euo pipefail

################################################################################
# Set Actions
################################################################################

# Set the Noodles action
export NOODLES_ACTION="run"

################################################################################
# Set Paths
################################################################################

# Set the path of the spec
export SPEC_PATH='a2oc/specs/train/reduced_action_space.10M/env_id-BeamRiderNoFrameskip-v4.yml'

################################################################################
# Start Training
################################################################################

# Run the common script
bash "a2oc/scripts/noodles/_common/$NOODLES_ACTION.sh"
