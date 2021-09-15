'''
Created on Sep 4, 2021

@author: paepcke
'''
from pathlib import Path
import random
from enum import Enum

import torch
import torchvision

from birdsong.utils.utilities import FileUtils
from data_augmentation.sound_processor import SoundProcessor


class BalancingStrategy(Enum):
    '''
    Choice between strategies for balancing a
    dataset 
    '''
    UNDERSAMPLE = 0
    OVERSAMPLE  = 1

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
    
    In addition the required __len__ and __getitem__ methods, 
    instances provide attributes:
    
            <inst>.data
                which is a list of tuples: 
                (sample-path : {0|1}, where 0 and 1 denote
                membership in the focal species.
            <inst>.focal_indices
            <inst>.others_indices

    The <inst>.focal_indices are dataset indices that access
    focal class samples. Analogously for <inst>.others_indices 
    '''
    
    #------------------------------------
    # Constructor 
    #-------------------


    def __init__(self,
                 species_dirs_root, 
                 focal_species,
                 transforms=None,
                 random_seed=None,
                 balancing_strategy=None,
                 balancing_ratio=1.0
                 ):
        '''
        The species_dirs_root should contain subdirectories,
        each of which holds the spectrogram snippets of one
        species.
        
        The focal_species should be a species
        name, such as VASEG, WTROS, etc.
        
        If balancing in the number samples of the
        focal species compared to all other samples is
        desired, set balancing_strategy to an element
        of the BalancingStrategy enum (OVERSAMPLE, UNDERSAMPLE).
        If balancing_strategy is specified, can also specify
        the target ratio
         
                num-minority-samples
                --------------------
                num-majority-samples
                
        If balancing_strategy is None, balancing_ratio is
        ignored.
        
        :param species_dirs_root:
        :type species_dirs_root:
        :param focal_species:
        :type focal_species:
        :param transforms:
        :type transforms:
        :param random_seed: set random.seed for repeatability
        :type random_seed: {None | int}
        :param balancing_strategy: if non-None, balance the
            dataset to reflect balancing_ratio between the
            minority and majority classes
        :type balancing_strategy: BalancingStrategy
        :param balancing_ratio: final desired ratio between
            the number of samples in the minority over majority
            classes.
        :type balancing_ratio: float
        '''

        if random_seed is not None:
            random.seed = random_seed

        self.focal_species = focal_species
        self.samples_root   = species_dirs_root

        self.transforms = transforms
        
        root_p = Path(species_dirs_root)
        target_species_p = root_p.joinpath(focal_species)
        
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

        # For convenience in class balancing,
        # collect the indices of focal, and of
        # others samples:

        self.focal_indices  = [i
                               for i, (_sample_path, is_focal_class)
                               in enumerate(self.data)
                               if is_focal_class
                               ]
        # The indices of non-focal samples
        # are the set difference between all 
        # data and the focal indices:
        self.others_indices = list(set(range(len(self.data))) - set(self.focal_indices))
        
        if balancing_strategy:
            balancer = ClassBalancer(self, balancing_strategy, balancing_ratio)
            self.original_data = self.data
            self.data = [self.original_data[idx]
                         for idx 
                         in balancer.new_data
                         ]

    #------------------------------------
    # split_generator
    #-------------------
    
    def split_generator(self, 
                        num_splits=5, 
                        test_percentage=20):
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

# -------------------- Class ClassBalancer -------------- 

class ClassBalancer:
    '''
    Given a dataset and a focus class, find 
    the number of samples in the focus class,
    and the sum of samples in all other classes.
    
    Determines which of the two sample sets is 
    a minority class. This will often be the focus
    class in one-against-all-others cases.
    
    Using techniques such as undersampling and oversampling,
    generates a list of datset indices. The indices 
    can be used by BinaryDataset during train/split.
    '''
    
    #------------------------------------
    # Constructor
    #-------------------
    
    def __init__(self, 
                 dataset, 
                 balancing_strategy, 
                 minority_to_majority_target=1.0):
        '''
        Analyzes the dataset's minority to majority 
        ratio. Makes a new set of data sample indices
        available in self.new_data, but does not make
        any changes in the dataset itself.
                
        The balancing_strategy specifies how class balancing
        will be accomplished. The value must be a member of
        the BalancingStrategy enum. 
        
        The minority_to_majority_target is a target ratio
        for how many samples the (initially) minority class should
        in the end contain relative to the (initially) majority class.
        I.e. the number is:
        
                number minority samples after balancing
                ----------------------------------------
                number majority samples after balancing
        
        It is legal to specify a number greater than 1.0,
        turning the minority class into a majority.
        
        Example balancing_strategy OVERSAMPLING
            Initial minority samples: 100
            Initial majority samples: 200
            
            minority_to_majority_target: 1.0
               final minority samples: 200
               final majority samples: 200
               
            minority_to_majority_target: 0.5
               final minority samples: 100
               final majority samples: 200
               
            minority_to_majority_target: 0.8
               final minority samples: 160
               final majority samples: 200
               
        Example balancing_strategy UNDERSAMPLING
            Initial minority samples: 100
            Initial majority samples: 200
            
            minority_to_majority_target: 1.0
               final minority samples: 100
               final majority samples: 100
               
            minority_to_majority_target: 0.5
               final minority samples: 100
               final majority samples: 200
               
            minority_to_majority_target: 0.8
               final minority samples: 100
               final majority samples: 125

        :param dataset: the dataset over which balancing is
            to be accomplished. The dataset itself is not
            modified.
        :type dataset: BinaryDataset
        :param balancing_strategy: which of the BalancingStrategy 
            methods to deploy
        :type balancing_strategy: BalancingStrategy
        :param minority_to_majority_target: target ratio minority to majority class
        :type minority_to_majority_target: float
        '''

        if type(balancing_strategy) != BalancingStrategy:
            raise TypeError(f"Balancing strategy must be a member of the enum BalancingStrategy, not {type(balancing_strategy)}")
        
        self.balancing_strategy = balancing_strategy
        self.target_ratio = minority_to_majority_target
        
        self._tally_sample_composition(dataset)

        if balancing_strategy == BalancingStrategy.UNDERSAMPLE:
            self.new_data = self._balance_undersample()
        elif balancing_strategy == BalancingStrategy.OVERSAMPLE:
            self.new_data = self._balance_oversample()
        else:
            raise NotImplementedError(f"Strategy {balancing_strategy} is not implemented")

    #------------------------------------
    # _balance_oversample
    #-------------------
    
    def _balance_oversample(self):
        '''
        From the list of indices self.minority,
        randomly choose self.minority_oversampling_needed
        indices, and return a list containing the 
        oversampled minority, mixed with all of self.majority
        in unspecified order
        
        :return list of indices to use from dataset
            such that classes are balanced to the 
            ratio given in the initialization (minority_to_majority_target)
        :rtype (int)
        '''
        # If the samples needed are more than
        # the samples available, random.sample would
        # throw ValueError: Sample larger than population or is negative
        # In that case, repeat sampling at the population
        # size till the oversampling goal is reached:

        if self.minority_oversampling_needed <= len(self.minority):
            oversamples = random.sample(self.minority, self.minority_oversampling_needed)
        else:
            # Replicate all of minority as often
            # as needed: round up, and snip excess
            # after:
            replicas = int(0.5 + self.minority_oversampling_needed / len(self.minority))
            oversamples = (self.minority * replicas)[:self.minority_oversampling_needed] 


        new_data = self.minority + oversamples + self.majority
        random.shuffle(new_data)
        return new_data

    #------------------------------------
    # _balance_undersample
    #-------------------
    
    def _balance_undersample(self):
        '''
        From the list of indices self.majority,
        randomly cull self.majority_undersampling_to_cull
        indices, and return a list containing the 
        undersampled majority, and the unchanged minority
        sample indices in unspecified order.

        :return list of indices to use from dataset
            such that classes are balanced to the 
            ratio given in the initialization (minority_to_majority_target)
        :rtype (int)
        '''
        undersamples = random.sample(self.majority, self.majority_undersampling_to_cull)
        remaining_majority = list(set(self.majority) - set(undersamples))
        new_data = self.minority + remaining_majority
        random.shuffle(new_data)
        return new_data

    #------------------------------------
    # tally_sample_composition
    #-------------------
    
    def _tally_sample_composition(self, dataset):
        '''
        Determine which is the minority class, and
        which the majority. Initializes self.minority and self.majority.
        The self.minority is a list of dataset indices that would
        retrieve members of the minority class. Analogously
        for self.majority.
        
        Example outcome:
            self.minority: [0,1,5,10]
            self.majority: [2,3,4,5,6,7,8,9]
            
        Also initializes:
        
           o self.minority_oversampling_needed:
             The number of minority samples to use multiple
             times in case minority oversampling is used to
             balance.
             
           o self.majority_undersampling_to_cull
             The number of majority samples to cull
             in case majority oversampling is used to
             balance.
        
        :param dataset: dataset from which to obtain sample counts
        :type dataset: BinaryDataset
        '''
        if len(dataset.focal_indices) <= len(dataset.others_indices):
            self.minority = dataset.focal_indices
            self.majority = dataset.others_indices
        else:
            self.minority = dataset.others_indices
            self.majority = dataset.focal_indices

        # Determine how many minority samples would
        # be needed after balancing if oversampling:
        minority_oversampling_wanted = len(self.majority) * self.target_ratio
        # Still needed: the target number minus what's already available
        self.minority_oversampling_needed = int(minority_oversampling_wanted - len(self.minority))

        # For the majority undersampling case:
        majority_undersampling_wanted   = len(self.minority) / self.target_ratio
        self.majority_undersampling_to_cull  = int(len(self.majority) - majority_undersampling_wanted)
