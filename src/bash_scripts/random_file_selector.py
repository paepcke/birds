#!/usr/bin/env python3
'''
Created on Apr 20, 2021

@author: paepcke
'''
import argparse
import os
import random
import sys


class RandomFilePicker:
    '''
    Takes a percentage P, and a directory D.
    Lists P percent of the files in D, randomly
    chosen
    '''

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, perc, directory=None, full_path=False, csv=False):
        '''
        Constructor
        '''
        if directory is None:
            directory = os.curdir

        # Allow dollar env vars like $HOME
        directory = os.path.realpath(os.path.expandvars(directory))
        
        if not os.path.exists(directory) or not os.path.isdir(directory):
            print(f"Location {dir} either does not exist, or is not a directory")
            sys.exit(1)

        files      = os.listdir(directory)
        full_paths = list(filter(os.path.isfile,
                                 [os.path.join(directory, fname) for fname in files]
                                 ))
        
        num_to_choose = len(full_paths) * perc // 100
        indicies      = random.sample(range(len(full_paths)), num_to_choose)
        
        # Get list of file names:
        selection = [full_paths[i] for i in indicies]
        if full_path:
            final_selection = selection
        else:
            final_selection = [os.path.basename(f) for f in selection]

        if csv:
            out_str = ','.join(final_selection)
        else:
            out_str = ' '.join(final_selection)
            
        print(out_str)
        
# ------------------------ Main ------------
if __name__ == '__main__':

    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Randomly select files from directory."
                                     )

    parser.add_argument('-f', '--full_path',
                        action='store_true',
                        help='whether or not to output full paths of file or just file names; default: just names',
                        default=False)
    
    parser.add_argument('-c', '--csv',
                        action='store_true',
                        help='output file names comma separated; default: space separated',
                        default=False)
    
    parser.add_argument('-d', '--directory',
                        help='directory from which to select files; default: current dir',
                        default=None)
    
    parser.add_argument('percentage',
                        type=int,
                        help='percentage of directory content to choose')

    args = parser.parse_args()

    if not args.percentage in range(0,100):
        print(f"Percentage must be an integer between 0 and 100")
        sys.exit(-1)


    RandomFilePicker(args.percentage,
                     directory=args.directory, 
                     full_path=args.full_path, 
                     csv=args.csv)