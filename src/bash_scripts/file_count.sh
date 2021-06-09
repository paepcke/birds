#!/usr/bin/env bash

# Lists pairs:
#
#    <subdirectoryName>,<num_files_in_subdir>
# 
# for each subdirectory

USAGE=$'USAGE: file_count.sh [directory]\n  Lists number of files in subdirs; default is pwd'

# Turn args into an array (without the script name in position 0)
args=("$@")

# Asking for help?
if [[ ${args[0]} == "-h" ]] || [[ ${args[0]} == "--help" ]]
then
    echo "$USAGE"
    exit
fi

# If no arg provided, use current dir:
if [[ -z ${args[0]} ]]
then
    dir=`pwd`
else
    dir=${args[0]}
fi

for file in $dir/*
   do
       file_count=$(ls -l $file | wc -l)
       echo "$file,$(expr $file_count - 1)"
   done
