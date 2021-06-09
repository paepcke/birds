#!/usr/bin/env python3
'''
Created on Jun 8, 2021

@author: paepcke
'''
import argparse
import os
from pathlib import Path
import random
import shutil
import sys

from data_augmentation.utils import Utils


class TestsetCreator(object):
    '''
    Move a percentage of files in from a training
    set to a test set.
    
    Ex: starting with
    
       source_tree
       
     dir1   dir2   dir3
     ------------------
     f1_1   f2_1   f3_1
     f1_2   f2_2   f3_2
     f1_3   f2_3
            f2_4
    
        # Move 20% of files from directories
        # under /path/to/source_tree to same-named
        # directories under path_to_dest_tree:
        
        testset_creator /path/to/source_tree path_to_dest_tree 20
        
    After the command 
    (the percentages are not correct; just for illustration) 
    
       source_tree                    dest_tree
                                                                 
     dir1   dir2   dir3           dir1   dir2   dir3             
     ------------------           ------------------             
            f2_1                  f1_1          f3_1             
     f1_2   f2_2   f3_2                         
     f1_3   f2_3                                                  
                                         f2_4                    

    Directories dir1/dir2/dir3 have been created by the
    command, and some of the files were moved from under
    source_tree to dest_tree.
    
    Procedure: Given the root of a directory tree
    randomly select a percentage of files
    from each subdirectory, and move those
    files to a same-named subdirectory under
    a different root.
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, from_dir, to_dir, percentage, unittesting=False):
        '''
        Constructor
        '''
        
        if unittesting:
            return
        
        if not os.path.exists(from_dir) and os.path.isdir(from_dir):
            raise FileNotFoundError(f"Argument from_dir must be a dir, not {from_dir}")
        # Sanity:
        file_distribution = self.get_file_distrib(from_dir)
        # Must have at least two files in each subdir
        # to make sense:
        for subdir, num_files in file_distribution.items():
            if num_files < 2:
                raise ValueError(f"Subdirectory {subdir} has fewer than 2 files")
        
        # Be nice and create to_dir if necessary:
        os.makedirs(to_dir, exist_ok=True)
        
        self.move_files(file_distribution, to_dir, percentage)

    #------------------------------------
    # move_files
    #-------------------

    def move_files(self, file_distribution, to_dir, percentage):
        '''
        Move percentage files from each subdirectory
        to its corresponding subdirectory under to_dir.
        Create that destination dir if needed.
        
        :param file_distribution: map absolute paths of
            source subdirectories to the number of files
            within them.
        :type file_distribution: {str : int}
        :param to_dir: root of destination directory tree
        :type to_dir: str
        :param percentage: percentage of files to move from
            each source subdirectory
        :type percentage: int
        '''
        
        for subdir, num_files in file_distribution.items():
            
            # Create destination subdir (no problem if exists):
            dest_subdir = os.path.join(to_dir, Path(subdir).name)
            os.makedirs(dest_subdir, exist_ok=True)

            # Absolute paths of files to move:
            files = Utils.listdir_abs(subdir)
            
            # Must have at least two files in the 
            # source directory, so we can move 1 of
            # those.
            if len(files) < 2:
                raise ValueError(f"Subdirectory {subdir} has fewer than 2 files")
            
            # Compute number of files to move based on
            # percentage of files that are in the source
            # subdir; but move at least 1 file:
            num_files_to_move = max(int(num_files * percentage / 100), 1)
            
            # Choice of which files to move is random: 
            files_to_move = random.sample(files, num_files_to_move)
            
            # Do it:
            for file in files_to_move:
                shutil.move(file, dest_subdir)

    #------------------------------------
    # get_file_distrib 
    #-------------------
    
    def get_file_distrib(self, root_dir):
        '''
        Given a directory root, return a dict
        whose keys are the absolute path of all
        subdirectories (only down 1 level). The
        values are the number for files in the
        respective subdir.
         
        :param root_dir: root of the subdirs 
        :type root_dir: src
        :return: dictionary mapping absolute subdirectory
            paths to the number of files they contain
        :rtype: {src : int}
        '''
        
        subdir_population = {}
        
        for root, dirs, _files in os.walk(root_dir):
            # Only go one level deep:
            if len(dirs) == 0:
                raise ValueError(f"Directory {root_dir} has no subdirectories from which to draw files")
            for subdir in dirs:
                subdir_abs = os.path.join(root, subdir)
                subdir_population[subdir_abs] = len(os.listdir(subdir_abs))
            # Only cover the first set of subidirs:
            return subdir_population


# ------------------------ Main ------------
if __name__ == '__main__':
    
    examples = '''
    Examples:
    
        # Move 20 percent of files within each
        # subdirectory: 
        testset_creator /path/to/my/from_root path_to_my_to_root 20
    '''
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Randomly move given percentage of directory tree",
                                     epilog=examples
                                     )


    parser.add_argument('from_dir',
                        help='directory root of all sample files')

    parser.add_argument('to_dir',
                        help='destinatino directory root')
    
    parser.add_argument('percentage',
                        type=int,
                        help='percentage of files to move')

    args = parser.parse_args()

    from_dir = args.from_dir
    
    if not os.exists(from_dir) or not os.path.isdir(from_dir):
        print(f"Directory {from_dir} does not exist")
        sys.exit(1)

    TestsetCreator(from_dir, args.to_dir, args.percentage)