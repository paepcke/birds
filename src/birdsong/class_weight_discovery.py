'''
Created on Jan 21, 2021

@author: paepcke
'''

import os

import natsort
import torch

from birdsong.utils.file_utils import FileUtils


class ClassWeightDiscovery(object):
    '''
    Given a root directory, finds all 
    subdirectories that do not begin with 
    a dot, and contain at least one non-directory
    file that does not begin with a dot. It is 
    assumed that the last path element of such 
    diretories name a class. This arrangment is 
    common in pytorch.
    
    For each directory, find the number of non-directory,
    non-dot-starting files. These numbers imply
    a weight, with the number of files in the most
    populated directory assigned the weight 1.0 

    The class names are sorted in natural sort order.
    Natural order means that Foo30 is placed after
    Foo3.  
    
    Each class is then associated with a weight. The
    result is a tensor of weight, arranged to correspond
    to the natural-sorted classes. 
    '''


    @classmethod
    def get_weights(cls, file_root):
        '''
        Given to root of a subdirectory,
        return a tensor of weights. The order
        of the weights corresponds to the 
        naturally sorted class names.
        
        @param file_root: full path to root
            of data file subtree
        @type file_root: str
        @return weights in naturally sorted class order
        @rtype: Tensor
        '''

        
        # Full paths of all the non-dot-starting 
        # dirs under file_root:
        
        dirs = FileUtils.find_class_paths(file_root)
        
        # Create:
        #  {'class1' : <num_samples>,
        #   'class2' : <num_samples>,
        #         ...
        #   } 
        class_populations = {os.path.basename(class_dir) : len(list(os.listdir(class_dir)))
                        		for class_dir in dirs
                        		if not os.path.basename(class_dir).startswith('.')
                        		}

        majority_class_population = max(class_populations.values())
        sorted_classes = natsort.natsorted(class_populations.keys())
        weights = [class_populations[class_name] / majority_class_population
                     for class_name in sorted_classes
                     ]
        return torch.tensor(weights) 

