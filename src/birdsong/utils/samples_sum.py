#!/usr/bin/env python3
'''
Created on Jul 25, 2021

@author: paepcke
'''
import argparse
import os
import sys

from statistics import median, mean

class SamplesInventory:
    '''
    A samples management utility for use from the
    command line. Not part of training.
    
    Given two root directories with subdirectories of
    species samples, print the name of each species,
    and the sum of the samples in each, like:
    
    Given dir1 = /foo/bar, and dir2 /blue/green, each
    containing subdirectories BANA and SHWC, printed output
    might be:
    
		  BANA: 221 + 0 = 221
		  SHWC: 44 + 6 = 50
    '''

    def __init__(self, root1, root2):
        '''
        Constructor
        '''
        sums = []
        
        for species_dir in os.listdir(root2):

            try:
                num_root1_species = len(os.listdir(os.path.join(root1, species_dir)))
            except FileNotFoundError:
                num_root1_species = 0
            try:
                num_root2_species = len(os.listdir(os.path.join(root2, species_dir)))
            except FileNotFoundError:
                num_root2_species = 0
                
            species_sum = num_root1_species + num_root2_species
            sums.append(species_sum)
            print(f"{species_dir}: {num_root1_species} + {num_root2_species} = {species_sum}")

        print()
        print(f"Mean: {mean(sums)}; median: {median(sums)}")
        

# ------------------------ Main ------------
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(prog=os.path.basename(sys.argv[0]),
                                     formatter_class=argparse.RawTextHelpFormatter,
                                     description="Lists combined number of sample files in two directory roots"
                                     )

    parser.add_argument('root1',
                        help='Root of first directories of sample species')

    parser.add_argument('root2',
                        help='Root of second directories of sample species')


    args = parser.parse_args()

    root1 = args.root1
    root2 = args.root2
    
    if not os.path.exists(root1):
        print(f"Cannot open {root1}")
    if not os.path.exists(root2):
        print(f"Cannot open {root2}")
        

    SamplesInventory(root1, root2)

