#!/usr/bin/env python3
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
    
    def move(self, move_fn=shutil.move, testing=False):
        '''
        Computes random sequence of indices into
        the result of walking the srcdir subtree.
        The lenght of this sequence correspondes 
        to desired percentage of files to move.
        
        Moves files, and returns number of files 
        moved.
        
        The optional move function will be used to 
        execute a move. Args must be (src_path, dst_dir).
        
        If testing is True, the following intermediate
        results will be available as instance variables
        after execution:
        
            num_to_move
            file_indices   # Indices into list of files
                           # as discovered by os.walk
            files_moved    # List of the files moved 
            
        
        @param move_fn: function to use when moving files.
            Primarily for testing without actually moving
            files
        @type move_fn: Function(src_file, dst_dir)
        @param testing: if True, store intermediates in self
        @type testing: bool
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
        
        num_to_move = int(np.round(num_files * self.perc / 100))
        
        if num_to_move == 0:
            print(f"With only {num_files} in src, {self.perc}% means no files to move")
            return

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
        
        if testing:
            # Need to copy, b/c file_indices
            # will be popped below:
            self.file_indices = file_indices.copy()
            self.num_to_move  = num_to_move
        
        # Go through the same file listings
        # as when the num of files was determined
        # above. Watch for files corresponding to
        # the random numbers in file_idx:

        if testing:
            files_moved = []
            
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
                
                move_fn(file_to_move, self.dstdir)
                
                if testing:
                    files_moved.append(file_to_move)
                
                try:
                    next_to_move = file_indices.pop()
                except IndexError:
                    # Moved all files
                    if testing:
                        self.files_moved = files_moved
                    return num_to_move
                
            file_idx += addtl_idx_range

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
    
