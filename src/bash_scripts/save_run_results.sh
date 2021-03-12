#!/usr/bin/env bash

# Rudimentary test whether cwd is birds:

CURR_DIR=$(pwd)

if [[ $(basename `pwd`) != 'birds' ||  ! -d `pwd`/src ]]
then
    echo "Must run in birds root"
    exit 1
fi    

USAGE="USAGE: $0 [-r|--remove] dest_root"

PARAMS=""

# Default: do not remove the 
# originals:

REMOVE=0

while (( "$#" )); do
  case "$1" in
    -r|--remove)
      REMOVE=1
      shift
      ;;
    *) # preserve positional arguments
      PARAMS="$PARAMS $1"
      shift
      ;;
  esac
done
# Set positional arguments in their proper place
eval set -- "$PARAMS"

if [[ -z $PARAMS ]]
then
	echo $USAGE
	exit
else
	DEST=$PARAMS
fi

#echo "Removal: $REMOVE; Dest: $DEST"

mkdir -p $DEST
echo "Copying run(s)..."
cp -r src/birdsong/runs $DEST
echo "Copying csv file(s)..."
cp -r src/birdsong/runs_raw_results/ $DEST
echo "Copying csv models..."
cp -r src/birdsong/runs_models/ $DEST

if [[ $REMOVE ]]
then
    echo "Removing run/csv/model files"
    rm src/birdsong/runs/*
    rm src/birdsong/runs_raw_results/*
    rm src/birdsong/runs_models/*
fi

echo "All done"

