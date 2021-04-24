#!/usr/bin/env bash

USAGE="$0 <root of species directories>"

if [[ $# != 1 ]]
then
    echo Usage: $USAGE
    exit 0
fi

if [[ $1 == '-h' ]]
then
    echo $USAGE
    exit 0
fi    
    
DATA_ROOT=$1

if [[ ! -e $DATA_ROOT ]]
then
    echo "Data root directory '$DATA_ROOT' does not exist"
    exit 1
fi

SCRIPT_DIR=$(dirname $0)

#*************
# echo "DATA_ROOT: $DATA_ROOT"
# echo "SCRIPT_DIR: $SCRIPT_DIR"
# exit 0
#*************

cd $DATA_ROOT
for species_dir in $(ls .)
do
    for file in $($SCRIPT_DIR/random_file_selector.py --full_path --directory $species_dir 20)
    do
       [ -d "../Testing/$species_dir" ] || mkdir -p ../Testing/$species_dir; mv $file ../Testing/$species_dir
    done
done
