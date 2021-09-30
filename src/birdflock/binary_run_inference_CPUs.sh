#!/usr/bin/env bash

USAGE="Usage: binary_run_inference_CPUs.sh experiments-root samples_root"
if [[ $1 == -h ]]
then
    echo $USAGE
    exit 0
fi    

if [[ $# != 2 ]]
then
    echo $USAGE
    exit 1
fi    

# Ensure that experiments does not have
# a trailing slash:

EXPERIMENTS=$(echo $1 | sed 's:/*$::')
SAMPLES=$2

THIS_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

parallel $THIS_DIR/binary_run_inference.py ::: --device ::: cpu ::: -- ::: $EXPERIMENTS/Classifier* ::: $SAMPLES
echo Done
