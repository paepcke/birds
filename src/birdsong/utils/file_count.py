#!/usr/bin/env python3
'''
Created on Jun 9, 2021

@author: paepcke

Used on command line to list number of
files in subdirectories: 

     file_counter.py
     file_counter.py -f
     file_counter.py dirname
     file_counter.py -f dirname
     
Subdirectory names and the number of files
in them are listed to stdout as comma separated
values:

     mydir,10
     yourdir,4
       ...

With -f, the absolute path is listed for each
subdirectory.

Works recursively.

'''
import argparse
import os
import sys

# ------------------------ Main ------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="List number of files in subdirectories."
                                     )

    parser.add_argument('-d', '--directory',
                        help='path to directory whose files to count; default: current dir',
                        default=os.getcwd())
    parser.add_argument('-f', '--full',
                        help='list the full subdirectory paths; default is just the dir name',
                        action='store_true')

    args = parser.parse_args()
    
    for root, dirs, files in os.walk(args.directory):
        for subdir in dirs:
            num_files = len(files)
            if args.full:
                print(f"{os.path.join(root, subdir)},{num_files}")
            else:
                print(f"{subdir},{num_files}")
