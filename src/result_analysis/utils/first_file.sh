#!/usr/bin/env bash

# Echoes the first file in the given directory.
# Used by saliency_maps.py in a system call to
# avoid calling walk, or listdir.

ls $1 | head -1

#files=( $(ls $1) )
#echo ${files[0]}
