'''
Created on Sep 4, 2021

@author: paepcke
'''
from pathlib import Path
import random

import torch
import torchvision

from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor


class BinaryDataset(torch.utils.data.Dataset):
    '''
    Pytorch type dataset that therefore implements __len__
    and __getitem__ methods. Handles both image (spectrogram),
    and audio (mp3, wav) data.
    
    Instances of this class are intended for binary classifiers
    in situations where there might potentially be many target 
    classes. For example: recordings of many bird species that
    are arranged in an torchvision image folder directory tree:
    
        
        species root directory:

	        subdirectory species1:
	             sample1
	             sample2
	             ...
	        subdirectory species2:
	             sample1
	             sample2
	             ...
	
	where the name of each subdirectory is a target class
	name. 
	
    When instantiating this BinaryDataset class, clients 
    passes the root directory, and the name of one subdirectory.
    The dataset then treats all samples under directories other
    than that given subdirectory as a single class of negative
    examples denoted with label (integer) 0. Samples in the 
    singled out subdirectory are treated as being of label 1.
    
    As required by many pytorch dataset class consumers, 
    __getitem__() returns a 2-tuple: a sample tensor, and 
    a label tensor. The latter being in {tensor(0), tensor(1)}

    The class handles both audio and image files.

    '''
    
    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self,
                 species_dirs_root, 
                 target_species,
                 transforms=None,
                 random_seed=None
                 ):
        '''
        The species_dirs_root should contain subdirectories,
        each of which holds the spectrogram snippets of one
        species.
        
        The target_species should be a species
        name, such as VASEG, WTROS, etc.
        
        :param species_dirs_root:
        :type species_dirs_root:
        :param target_species:
        :type target_species:
        :param transforms:
        :type transforms:
        :param random_seed: set random.seed for repeatability
        :type random_seed: {None | int}
        '''

        if random_seed is not None:
            random.seed = random_seed

        self.target_species = target_species
        self.samples_root   = species_dirs_root

        self.transforms = transforms
        
        root_p = Path(species_dirs_root)
        target_species_p = root_p.joinpath(target_species)
        
        # Iterator of species names
        # other than the target species:
        other_dir_it  = filter(lambda dir_p: dir_p != target_species_p,
                               root_p.iterdir())
        # List of full sample paths from the
        # set of other species:
        other_species_paths = []
        for other_species_dir_p in other_dir_it:
            other_species_paths.extend(list(other_species_dir_p.iterdir()))
        
        # Randomize the order of the paths:
        random.shuffle(other_species_paths)
            
        # Random list of target species sample paths:
        target_species_paths = list(target_species_p.iterdir())
        random.shuffle(target_species_paths)
        
        # Now have a randomly ordered list of 
        # 'other' samples, and the same for target samples.
        # Create a combined list of 2-tuples:
        #   (sample_path, {0|1})
        # where 0 if sample is from one of the other species,
        # and 1 if target species:
        
        other_tuples  = zip(other_species_paths, [0]*len(other_species_paths))
        target_tuples = zip(target_species_paths, [1]*len(target_species_paths))
        
        # Combine the tuples, shuffle them, and 
        # the result is the dataset:
        
        self.data = list(other_tuples) + list(target_tuples)
        random.shuffle(self.data)

    #------------------------------------
    # split_generator
    #-------------------
    
    def split_generator(self, num_splits=5, test_percentage=20):
        '''
        A generator providing successive random
        splits into train and test indices. Indices
        are 0..len(data). So successive returns when
        self.data were to contain 10 entries might be:
        
                Train            Test
            ([0,1,3,4,5,6,7,9], [2,8])
            ([0,1,2,4,6,7,8,9], [5,3])
                ...
        
        :param num_splits: number of train/test indices
            array pairs to generate
        :type num_splits: int
        :param test_percentage: percentage of all indices
            to be removed as test indices
        :type test_percentage: {int | float}
        :returns repeated array pairs of train/test indices
        :rtype ((int),(int))
        '''
        index_set = set(range(len(self.data)))
        test_sample_size = int(len(index_set) * test_percentage / 100.)
        
        for _i in range(num_splits):
            test_set  = set(random.sample(index_set, test_sample_size))
            train_set = index_set - test_set
            yield (list(train_set), list(test_set))

    #------------------------------------
    # __len__ 
    #-------------------
    
    def __len__(self):
        return len(self.data)

    #------------------------------------
    # __getitem__
    #-------------------
    
    def __getitem__(self, int_key):
        '''
        Retrieve the requested tuple of
        file-path (a PosixPath instance), and 
        either 0 or 1, depending on the sample's
        class. 
        
        If the sample being accessed is an image, then 
        load the respective spectrogram, and return
        a tuple: the spectro as tensor, and the label
        as tensor.
        
        If the sample is a sound, return the audio 
        amplitudes and the label as tensors.
        
        :param int_key: requested index into the data
        :type int_key: int
        :return a 2-tuple containting a spectrogram and
            the class label: 0 if specto is not a positive
            example of the target class, else 1.
        :rtype (torch.Tensor, torch.Tensor)
        '''
        
        file_p, bin_class = self.data[int_key]
        
        if file_p.suffix in FileUtils.IMG_EXTENSIONS:
            # Load the image file:
            img_RGBA, _metadata = SoundProcessor.load_spectrogram(str(file_p), to_nparray=False)

            # If RGBA, remove a transparency (alpha) channel:
            try:
                img_RGBA.getchannel('A')
            except ValueError:
                # Already was RGB:
                img_RGB = img_RGBA 
            else:
                # Get rid of alpha channel
                img_RGB = img_RGBA.convert('RGB')

            if self.transforms is not None:
                res_tensor = self.transforms(img_RGB)
            else:
                res_tensor = torchvision.transforms.ToTensor()(img_RGB)
        else:
            data_arr, _sample_rate = SoundProcessor.load_audio(str(file_p))
            res_tensor = torch.tensor(data_arr, dtype=float)
        

        return res_tensor, torch.tensor(float(bin_class))
