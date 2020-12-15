"""
A file that uses sci-kit learn's StratifiedKFold module to split the training data
and then uses cross validation to train and validate models with different sets of data.
"""
from _collections import OrderedDict
import os
from pathlib import Path
import sys

from sklearn.model_selection import StratifiedKFold
from torchvision import datasets
from torchvision import transforms
from torchvision.datasets.folder import ImageFolder

import numpy as np
from training import Training


# Should actually be 1:3 but broke the system:
SAMPLE_WIDTH  = 400 # pixels
SAMPLE_HEIGHT = 400 # pixels

class BirdDataset(ImageFolder):
    
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    #------------------------------------
    # Constructor 
    #-------------------

    def __init__(self, 
                 filepath,
                 sample_width=SAMPLE_WIDTH,
                 sample_height=SAMPLE_HEIGHT
                 ):
        transform_img = transforms.Compose([
            transforms.Resize((sample_width, sample_height)),  # should actually be 1:3 but broke the system
            transforms.ToTensor()])

        super().__init__(filepath,
                       transform=transform_img,
                       target_transform=None,
                       loader=datasets.folder.default_loader,
                       is_valid_file = lambda filepath: 
                            Path(filepath).suffix in BirdDataset.IMG_EXTENSIONS
                       )
        # Build three data structures:
        #     class_name --> class id int needed for model building
        #                    No strings allowed for class_to_id
        #     sample_id  --> sample file path
        #     sample_id  --> class id
        #
        # Use OrderedDict for some of the above maps to
        # ensure that calls to keys() or values() always
        # return the same sequence.
        
        # The name of each folder in the train subdir
        # is the name of a class (i.e. species); go through this
        # folder/species sequence:
        
        # The class_name --> class id map: for IDs, use 
        # the superclass's 'classes' property, which is a list
        # of class names (a.k.a. folder names):
        
        self.class_to_id = OrderedDict({class_name : class_id 
                            for class_id, class_name 
                             in enumerate(self.classes)})
        self.sample_id_to_path = OrderedDict({})
        self.sample_id_to_class = OrderedDict({}) 
        
        # First sample ID:
        sample_id_start = 0
        
        # Go through the samples in each folder (i.e. class):
        
        for sample_folder in [os.path.join(filepath, folder_name)
                                for folder_name in self.classes]:
            class_name    = os.path.basename(sample_folder)
            class_id      = self.class_to_id[class_name]
            
            # List of full paths to each sample of current class:
            folder_content  = [os.path.join(filepath, class_name, sample_path)
                                 for sample_path in os.listdir(sample_folder)]
            # IDs we will assign to the samples in this folder:
            sample_id_range = range(sample_id_start, sample_id_start + len(folder_content)) 
            
            # Create sample id --> filename map for just the 
            # samples in this folder:
            sample_id_map = OrderedDict({sample_id : folder_content[i]
                                            for i, sample_id in enumerate(sample_id_range)
                                         })
            
            # Append this folder's sample id --> filename to
            # our emerging final map (dict1.update(dict2) appends):
            self.sample_id_to_path.update(sample_id_map)
            
            # Build sample id --> class ID; the class ID is 
            # the same for all samples in this folder:
            self.sample_id_to_class.update({sample_id : class_id
                                            for sample_id in sample_id_range 
                                            })
            # Update where to start the IDs for the
            # next folder
            sample_id_start += len(sample_id_range)

    #------------------------------------
    # sample_ids
    #-------------------

    def sample_ids(self):
        '''
        Return an iterator over all 
        sample IDs. Multiple calls guaranteed
        to return IDs in the same order each time
        
        :returns List of sample identifiers
        :rtype [int]
        '''
        
        return self.sample_id_to_class.keys()

    #------------------------------------
    # sample_classes 
    #-------------------
    
    def sample_classes(self):
        '''
        Returns an np array of class ids
        of all samples. The sequence order is
        guaranteed to be the same for each call.
        Furthermore, the sequence matches that of
        the sequence returned from sample_ids()
        
        :return: Array over all class labels in
        sample_id order.
        :rtype: np.array
        '''
        
        return np.array(list(self.sample_id_to_class.values()))


    #------------------------------------
    # __len__
    #-------------------

    def __len__(self):
        return len(self.sample_id_to_path)

    #------------------------------------
    # __getitem__ 
    #-------------------

    def __getitem__(self, sample_id):

        img = datasets.folder.default_loader(self.sample_id_to_path[sample_id])
        return (img, self.sample_id_to_class[sample_id])
