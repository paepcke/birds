#!/usr/bin/env python3
'''
Created on Jul 28, 2021

@author: paepcke

Command line script to move all non-focus species from
one subdirectory to another.
'''

import argparse
import os
import sys

from birdsong.utils.utilities import FileUtils


if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Move non-focus species dir to another place"
                                     )

    parser.add_argument('from_root',
                        help='path to root of sources species subdirectories')
    parser.add_argument('to_root',
                        help='path to root of destination species subdirectories')

    args = parser.parse_args()

    FileUtils.mv_non_focus_species(args.from_root, args.to_root)

