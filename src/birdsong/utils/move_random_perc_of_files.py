'''
Created on Jan 15, 2021

@author: paepcke
'''
import argparse
import os
import sys

import shutil
import numpy as np


class FileMover(object):
    '''
    Moves a given percentage of files below
    a given directory to a target directory.
    
    Used to separate test sets
    '''

    #------------------------------------
    # Constructor
    #-------------------

    def __init__(self, srcdir, dstdir, perc):
        '''
        Constructor
        '''
        self.srcdir = srcdir
        self.dstdir = dstdir
        self.perc = perc
        
    #------------------------------------
    # move 
    #-------------------
    
    def move(self, move_fn=shutil.move):
        '''
        Computes random sequence of indices into
        the result of walking the srcdir subtree.
        The lenght of this sequence correspondes 
        to desired percentage of files to move.
        
        Moves files, and returns number of files 
        moved.
        
        @param move_fn: function to use when moving files.
            Primarily for testing without actually moving
            files
        @type move_fn: Function(src_file, dst_dir)
        @return: number of files moved
        @rtype: int
        '''
        
        # Fast method for finding 
        # the number of files below srcdir
        # without loading a list of file names:
        
        file_iter = os.walk(self.srcdir)
        num_files = 0
        for _root, _dirs, files in file_iter:
            num_files += len(files)
        
        num_to_move = np.round(100 * perc / num_files)
        
        # A generator of numbers:
        rng = np.random.default_rng()

        # Get as many ints between 0 and the
        # total number of files below srcdir
        # as the percentage of the files to 
        # be moved. Without replacement implies
        # no duplicates. The sort reversal is
        # convienience so that we can use
        # pop() on the sequence later on:
        
        
        file_indices = sorted(rng.choice(num_files, num_to_move, replace=False),
                              reverse=True)
        
        # Go through the same file listings
        # as when the num of files was determined
        # above. Watch for files corresponding to
        # the random numbers in file_idx:
        
        file_iter = os.walk(self.srcdir)
        file_idx = 0
        next_to_move = file_indices.pop()
        for root, _dirs, files in file_iter:
            addtl_idx_range = len(files)
            while next_to_move in range(file_idx, file_idx + addtl_idx_range):
                # A file to be moved is among
                # the files in this current dir;
                # find its position in this batch
                # of files:
                file_idx_to_move = next_to_move - file_idx
                file_to_move = os.path.join(root, files[file_idx_to_move])
                
                shutil.move(file_to_move, self.dstdir)
                next_to_move = file_indices.pop()
                
            file_idx += addtl_idx_range

        return num_to_move

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Move random percentage of files to destination"
                                     )

    parser.add_argument('srcdir',
                        help='directory below which some files are to be moved'
                        )
    parser.add_argument('dstdir',
                        help='directory where files are to be moved'
                        )
    parser.add_argument('percentage',
                        type=int,
                        help='percentage of files to move'
                        )

    args = parser.parse_args()

    srcdir = args.srcdir
    perc   = args.percentage
    dstdir = args.dstdir
    
    if not os.path.exists(srcdir):
        print(f"Source directory {srcdir} does not exist")
        sys.exit(1)
        
    if not os.path.exists(dstdir):
        os.makedirs(dstdir)
        
    if perc < 0:
        print(f"Percentage must be >= zero, not {perc}")
        sys.exit(1)
        
    mover = FileMover(srcdir, dstdir, perc)
    num_files_moved = mover.move() 
    
