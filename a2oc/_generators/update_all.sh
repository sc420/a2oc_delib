#!/bin/bash

# Cause the script to exit on any errors
# Reference: https://vaneyckt.io/posts/safer_bash_scripts_with_set_euxo_pipefail/
set -euxo pipefail

################################################################################
# Update Training Scripts
################################################################################

python a2oc/scripts/train/test/_generators/gen_scripts.run.py

################################################################################
# Update Noodles Scripts
################################################################################

python a2oc/scripts/noodles/test/_generators/gen_scripts.py

################################################################################
# Update Noodles Specs
################################################################################

python a2oc/specs/train/test/_generators/gen_specs.py
