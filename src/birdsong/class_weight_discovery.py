'''
Created on Jan 21, 2021

@author: paepcke
'''

import os
from pathlib import Path
import sys

from logging_service.logging_service import LoggingService
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

        #   OrderedDict{class_name : [Path(dir1), Path(dir2)]
        # The class names are already sorted:
        class_name_paths_dir = FileUtils.find_class_paths(file_root)
        
        # Create:
        #  {'class1' : <num_samples>,
        #   'class2' : <num_samples>,
        #         ...
        #   }
        
        class_populations = {}
        for class_name in class_name_paths_dir.keys():
            num_samples = 0
            # Each class may have samples in multiple
            # directories; add them up:
            for class_dir in class_name_paths_dir[class_name]:
                num_samples += len([file_name 
                                     for file_name 
                                     in os.listdir(class_dir)
                                     if Path(file_name).suffix in FileUtils.IMG_EXTENSIONS
                                     ])
            class_populations[class_name] = num_samples
            
        if len(class_populations) == 0:
            LoggingService().err(f"No target classes found under {file_root}")
            sys.exit(1)
        majority_class_population = max(class_populations.values())
        weights = []
        for class_name in class_name_paths_dir.keys():
            weights.append(class_populations[class_name] / majority_class_population)

        return torch.tensor(weights) 

