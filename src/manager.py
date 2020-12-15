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

import numpy as np
from training import Training


# the location of the samples
FILEPATH = "/home/data/birds/NEW_BIRDSONG/ALL_SPECTROGRAMS/"
EPOCHS = 100
SEED = 42
BATCH_SIZE = 16
KERNEL_SIZE = 5
NET_NAME = 'BasicNet'
GPU = 0

class cross_fold_split_and_validation:
    """
    A class that creates KFoldStratified splits of the data and then trains separate models and evaluates them with
    cross-validation.

    :param filepath: the parent directory of the labelled samples.
    :type filepath: str
    :param epochs: the max number of epochs to train each model for.
    :type epochs: int
    :param seed: the integer seed for training the model.
    :type seed: int
    :param batch_size: An int used to define the batch size of each layer.
    :type batch-size: int
    :param kernel_size: An int the defines the kernel size of the convolutional layers.
    :type kernel_size: int
    :param net: the name of the type of model to use
    :type net: str
    :param processor: The number of the GPU to use. The CPU is used if None.
    :type processor: int
    """
    def __init__(self, filepath, epochs, seed, batch_size=32, kernel_size=5, net='BasicNet', processor=None):
        """Constructor method
        """
        #initialize all the fields
        # these are just passed to the model
        self.file_path = filepath
        self.max_epochs = epochs
        self.seed = seed
        self.bs = batch_size
        self.ks = kernel_size
        self.model = net
        self.gpu = processor
        # these are used for tracking all the samples and splitting them
        self.X = OrderedDict()  # samples
        self.y = OrderedDict()  # labels
        self.file_list = []  # all of the samples
        self.folder_list = []  # all of the class_to_id
        self.ident = 0  # the number of class_to_id
        self.count = 0  # the number of total samples
        self.train_index = 0
        self.test_index = 0


class BirdDataset:
    
    # Should actually be 1:3 but broke the system:
    SAMPLE_WIDTH  = 400 # pixels
    SAMPLE_HEIGHT = 400 # pixels
    
    IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

    def __init__(self, 
                 filepath,
                 sample_width=BirdDataset.SAMPLE_WIDTH,
                 sample_height=BirdDataset.SAMPLE_WIDTH,
                 ):
        """Counts the number of class_to_id, counts the number of samples in each class,
        stores the samples with their associated labels.
        Creates two ordered dicts: X and y. These map integers
        to full species_sample_path paths of spectrogram images. Example
        Two species, one with five samples, the other with three:
            
              X: 0: species0_img_file0
                 1: species0_img_file1
                       ...
                 4: species0_img_file4
                 
                 5: species1_img_file0
                 6: species1_img_file1
                 7: species1_img_file2
                 
              y: 0: species0_folder_name
                 1: species0_folder_name
                      ...
                 4: species0_folder_name
                 5: species1_folder_name
                 6: species1_folder_name
                 7: species1_folder_name
        """
        assert os.path.exists(filepath)
        # Define the transform to be done on all images
        transform_img = transforms.Compose([
            transforms.Resize((BirdDataset.SAMPLE_WIDTH, BirdDataset.SAMPLE_HEIGHT)),  
            transforms.ToTensor()])
        
        super(datasets.ImageFolder, self).__init__(filepath,
                                                   transform=transform_img,
                                                   target_transform=None,
                                                   loader=datasets.folder.default_loader,
                                                   target_transform=None,
                                                   is_valid_file=lambda file_path : Path(filepath.lower()).suffix in BirdDataset.IMG_EXTENSIONS
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
        
        # The class_name --> class id map: create the 
        # IDs on the fly; class names are the folder names
        # without leading path:
        
        self.class_to_id = OrderedDict({os.path.basename(class_name) : class_id 
                            for class_id, class_name 
                             in enumerate(os.listdir(filepath))})
        self.sample_id_to_path = OrderedDict({})
        self.sample_id_to_class = OrderedDict({}) 
        
        # First sample ID:
        sample_id_start = 0
        
        # Go through the samples in each folder (i.e. class):
        
        for sample_folder in filepath:
            class_name    = os.path.basename(sample_folder)
            class_id      = self.class_to_id[class_name]
            
            # List of full paths to each sample of current class:
            folder_content  = os.listdir(sample_folder)
            # IDs we will assign to the samples in this folder:
            sample_id_range = range(sample_id_start, len(folder_content)) 
            
            # Create sample id --> filename map for just the 
            # samples in this folder:
            sample_id_map = OrderedDict({sample_id : folder_content[sample_id]
                                            for sample_id in sample_id_range
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
            
    def __len__(self):
        return len(self.sample_id_to_path)
    
    
    def __getitem__(self, sample_id):

        img = datasets.folder.default_loader(self.sample_id_to_path[sample_id])
        return (img, self.sample_id_to_path[sample_id])
    

    def run(self):
        """
        Splits the train and test data using the StratifiedKFold function from sklearn
        and then trains and evaluetes the model for each split.
        """
        # Perform stratified K Fold split
        skf = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)

        # enumerate splits and train models
        num = 0

        # Stratified k-fold needs only the labels 
        # in an array; the corresponding samples each 
        # have the same index as the one for each 
        # y-split (https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold)
        
        for train_index, test_index in skf.split(np.zeros(len(self.X)), self.y.values()):
            num += 1
            print("running split:", num)
            # gets the train and test data for this split
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            # create and train the model with this split of data
            birdsong = Training(X_train,
                                y_train,
                                X_test,
                                y_test,
                                self.max_epochs, self.bs, self.ks, self.seed, self.model,
                                self.gpu)
            birdsong.train()

if __name__ == '__main__':
    """The main method for the manager module.
    """
    if len(sys.argv) > 1:
        cfsv = cross_fold_split_and_validation(sys.argv[1], EPOCHS, SEED, BATCH_SIZE, KERNEL_SIZE, NET_NAME, GPU)
    else:
        cfsv = cross_fold_split_and_validation(FILEPATH, EPOCHS, SEED, BATCH_SIZE, KERNEL_SIZE, NET_NAME, GPU)

    cfsv.count_and_label()
    cfsv.run()
