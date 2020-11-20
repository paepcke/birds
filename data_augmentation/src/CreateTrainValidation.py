"""
A script to split a directory of all samples into a training and validation sets in new directories.

File Layout:

* train folder

* validation folder

* FILEPATH

        -- species 1 folder

        -- species 2 folder

        -- ................
"""

import os
import random
import sys

FILEPATH = '/home/data/birds/NEW_BIRDSONG/current_files/'
SPLIT = 0.8


def manual_split(split, starting_filepath):
    """
    A script to split a directory of all samples into a training and validation sets in new directories.

    :param split: the fraction of the samples that should be used for training. 0.8 by default
    :type split: float
    :param starting_filepath: the directory containing the species subdirectories
    :type starting_filepath: str
    """
    birdlist = os.listdir(starting_filepath)
    print(birdlist)
    if '.DS_STORE' in birdlist:
        birdlist.remove('.DS_Store')  # ghost file
    print(birdlist)
    for bird in birdlist:
        for file in os.listdir(starting_filepath + bird):
            print(file)
            if random.random() >= split:
                os.rename(starting_filepath + bird + '/' + file, starting_filepath +
                          "../validation/" + bird + "/" + file)
            else:
                os.rename(starting_filepath + bird + '/' + file, starting_filepath +
                          "../train/" + bird + "/" + file)


if __name__ == '__main__':
    """main method"""
    if len(sys.argv) > 1:
        manual_split(SPLIT, sys.argv[1])
    else:
        manual_split(SPLIT, FILEPATH)
