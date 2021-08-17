from csv import DictReader
from enum import Enum
from fnmatch import fnmatch
import os
from pathlib import Path
import random
import shutil
import warnings

from experiment_manager.neural_net_config import NeuralNetConfig
import librosa
from matplotlib import MatplotlibDeprecationWarning
from natsort import natsorted
from torch import cuda
import torch

import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import pandas as pd

from birdsong.utils.species_name_converter import SpeciesNameConverter, DIRECTION, ConversionError

#-------------------------- Enum for Policy When Output Files Exist -----------
class WhenAlreadyDone(Enum):
    ASK = 0
    OVERWRITE = 1
    SKIP = 2

#---------- Enum for how much to equalize num samples per species -----------

# How many augmentations to create 
# for each species. Measured against
# the number of samples of the species 
# with the most number of available
# samples.
# Meaning: 
#    TENTH: all species will have at least a 				  
#           10th of samples in the most populous 
#           species 	  
#   MEDIAN: all species will have at least the median
#    	    number of samples in the species populations
#      MAX: all species will end up with the number of
#           species of the most populously represented species
#           in the training set: 
class AugmentationGoals(Enum):
    TENTH   = 0
    MEDIAN  = 1
    MAX     = 2
    NUMBER  = 3 # not implemented yet: Arbitrary number of samples to create
    
#---------------- Enum AudAugMethod ---------------

# Audio augmentation methods:

# The value of each element is the 
# substring used in augmentation files to 
# indicate the type of aug used:

class AudAugMethod(Enum):
    ADD_NOISE   = '_bgd'
    TIME_SHIFT  = '-shift'
    VOLUME      = '-volume'

#---------------- Enum ImgAugMethod ---------------

# Spectrogram image augmentation methods:

class ImgAugMethod(Enum):
    NOISE       = '-noise'
    FMASK       = '-fmask'
    TMASK       = '-tmask'
    #ORIGINAL    = 'original'


# ------------------- Class Interval -------------

class Interval(dict):
    '''
    Instances store intervals of ints or
    floats. Supports a dict API for retrieving
    the inclusive low bound, and the exclusive
    high bound.
    '''
    def __init__(self, low_val, high_val):
        super().__init__()
        self['low_val']  = low_val
        self['high_val'] = high_val
        
    def contains(self, num):
        return num >= self['low_val'] and num < self['high_val']

#------------------------------ Utility  -------------

class Utils:

    # If multiple cores are available,
    # only use some percentage of them to
    # be nice:
    
    MAX_PERC_OF_CORES_TO_USE = 50

    #------------------------------------
    # noise_multiplier
    #-------------------

    @classmethod
    def noise_multiplier(cls, orig_recording, noise):
        MIN_SNR, MAX_SNR = 3, 30  # min and max sound to noise ratio (in dB)
        snr = random.uniform(MIN_SNR, MAX_SNR)
        noise_rms = np.sqrt(np.mean(noise**2))
        orig_rms  = np.sqrt(np.mean(orig_recording**2))
        desired_rms = orig_rms / (10 ** (float(snr) / 20))
        return desired_rms / noise_rms
    
    #------------------------------------
    # create_folder 
    #-------------------
    
    @classmethod
    def create_folder(cls, dir_path, overwrite_policy):
        '''
        Given the path to a directory:
            o If the dir does not exist, create it
            o If dir exists and overwrite_policy is WhenAlreadyDone.SKIP,
              and return True
            o If dir exists and overwrite_policy is WhenAlreadyDone.ASK,
              ask for permission to replace the dir
                - if permission denied, existing dir remains, and return False
                - else tree under dir is cleared, and return True
              If dir exists and overwrite_policy is WhenAlreadyDone.OVERWRITE,
              clear the directory and return True
        
        Returns True if a new, empty directory was created.
        Return of False means the dir existed, and was left
        untouched. 
                
        :param dir_path: path of directory to create 
        :type dir_path: str
        :return True is empty dir was created, else False
        :rtype: bool
        
        '''
        if not isinstance(overwrite_policy, WhenAlreadyDone):
            raise TypeError(f"Overwrite policy must be a member of WhenAlreadyDone, not {type(overwrite_policy)}") 
        if os.path.exists(dir_path):
            if overwrite_policy == WhenAlreadyDone.OVERWRITE:
                shutil.rmtree(dir_path)
                os.mkdir(dir_path)
                return True
            elif overwrite_policy == WhenAlreadyDone.SKIP:
                return True
            # Need to ask client for permission to replace:
            ans = input(f"{dir_path} already exists. Replace? [y/N]:  ")
            if ans.lower() in ["y", "yes"]:
                shutil.rmtree(dir_path)
                os.mkdir(dir_path)
                return True
            # Did not create a new dir
            return False
        else:
            os.mkdir(dir_path)
            return True

    #------------------------------------
    # sample_compositions_by_species 
    #-------------------

    @classmethod
    def sample_compositions_by_species(cls, path, augmented=False):
        '''
        Return a dict with frequencies of sound or spectrogram
        files in path by species. Two cases:
        
           augmented is False:
               return like
               
                         num_samples
               DYSMEN_S            2
               HENLES_S            6

           augmented is True:
           
                                    add_bg  time_shift  volume  mask  noise original
               AMADEC                   17          12      19     0      1         0     
               ARRAUR_C                  0           0       0     0      2         0
               Automolusexsertus        17          17      17     0      3         0
               Brotogerisjugularis       0           0       0     0      6         0
               CORALT_C                 11          11      11     0      10        0           

        :param path: root of species subdirs
        :type path: src
        :param augmented: whether or not to differentiate between
            augmented and not augmented files
        :type augmented: bool
        :return: dataframe with distribution
        :rtype: pandas.DataFrame
        '''

        # initialize dict - usage num_samples_in['CORALT_S'] = 64
        num_samples_in = {} 
        for species in os.listdir(path):
            if augmented:
                aug_type_dict = {"add_bg":0, 
                                 "time_shift":0,
                                 "volume":0, 
                                 "mask":0,
                                 "noise":0,
                                 "original":0}
                for sample_name in os.listdir(os.path.join(path, species)):
                    if "_bgd" in sample_name: aug_type_dict["add_bg"] += 1
                    elif "-shift" in sample_name: aug_type_dict["time_shift"] += 1
                    elif "-volume" in sample_name: aug_type_dict["volume"] += 1
                    
                    # For Spectrograms:
                    elif "-noise" in sample_name: aug_type_dict["noise"] += 1
                    elif "fmask" in sample_name or "tmask" in sample_name: aug_type_dict["mask"] += 1
                    else: aug_type_dict["original"] += 1
                num_samples_in[species]= aug_type_dict
            else:
                # Get like: 
                #     num_samples
                #     DYSMEN_S            2
                #     HENLES_S            6

                num_samples_in[species] = {"num_samples":len(os.listdir(os.path.join(path, species)))}
        df = pd.DataFrame.from_dict(num_samples_in, orient='index').sort_index()
        return df
    
    #------------------------------------
    # find_species_names 
    #-------------------
    
    @classmethod
    def find_species_names(cls, species_root):
        '''
        Given the root of a directory in which only
        subdirectories exist that are named after a 
        bird species, return a list of those species
        names.
        
        Expectation for species_root:
        
                    species_root
                    
            brd1_dir   some_file   brd2_dir, brd3_dir ...
              A   N   Y   T    H    I    N    G
              
        I.e. there may be files in species_root. But anything
        that is a directory must have a species name.
        
        :param species_root: root of bird species related subdirectories
        :type species_root: str
        :return: list of species names
        :rtype: [str]
        '''
        try:
            species_names = [dir_name 
                             for dir_name 
                             in os.listdir(species_root)
                             if os.path.isdir(os.path.join(species_root, dir_name))
                             ]
        except FileNotFoundError:
            return []
        return species_names

    #------------------------------------
    # find_total_recording_length
    #-------------------

    @classmethod
    def find_total_recording_length(cls, species_dir_path):
        total_duration = 0
        for recording in os.listdir(species_dir_path):
            y, sr = librosa.load(os.path.join(species_dir_path, recording))
            total_duration += librosa.get_duration(y, sr)
        return total_duration

    #------------------------------------
    # recording_lengths_by_species
    #-------------------

    
    @classmethod
    def recording_lengths_by_species(cls, path):
        num_samples_in = {} # initialize dict - usage num_samples_in['CORALT_S'] = 64
        for species in os.listdir(path):
            rec_len = cls.find_total_recording_length(os.path.join(path, species))
            num_samples_in[species] = {"total_recording_length": rec_len} 
        return pd.DataFrame.from_dict(num_samples_in, orient='index').sort_index()

    #------------------------------------
    # count_max_augs 
    #-------------------

    @classmethod
    def count_max_augs(cls, distribution):
        count = 0
        for val in distribution:
            if val != 0:
                count += 1
        return count

    #------------------------------------
    # find_in_tree_gen
    #-------------------

    @classmethod
    def find_in_tree_gen(cls, root_dir, pattern='*', entry_type='file'):
        '''
        
        Generator version of find_in_dir_tree(). For new
        code use this method, which does not generate a
        comprehensive list first.
        
        Similates part of the bash shell find command.
        Only bash patters are supported, not Python regex.
        For the find -type values, only file and directory 
        are supported as types. Though abbreviations
        'f' and 'd' are fine.
        
        Case of strings are case-normalized using os.path.normcase().
        Change fnmatch to fnmatchcase() in the function to
        perform case-sensitive searches.
        
        Examples:
           
           # Successively yield absolute file names in /tmp
              for f in find_in_dir_tree('/tmp', entry_type='file')
              
           # Successively yield absolute paths to text files in /tmp
              for f in find_in_dir_tree('/tmp', pattern="*.txt")
              
           # Successively yield absolute paths to directories in /tmp that start 
           # with 'Kfold'
              for d in find_in_dir_tree('/tmp', pattern="Kfold", entry_type='directory')
              for d in find_in_dir_tree('/tmp', pattern="Kfold", entry_type='d')

        :param root_dir: directory in which to start recursive descent
        :type root_dir: str
        :param pattern: pattern that file or dir must match
        :type pattern: str
        :param entry_type: whether file or directories are included
        :type entry_type: str
        :return: on first call, returns the generator, then, outputs
            upon each call
        :rtype: {Iterator | str}
        '''

        if pattern is None and entry_type is None:
            raise ValueError("At least one of pattern and/or type must be non-None")
        if entry_type not in [None, 'file', 'f', 'directory', 'd']:
            raise ValueError(f"Entry type must be one of {[None, 'file', 'f', 'directory', 'd']}, not {entry_type}")
    
        if entry_type in [None, 'f', 'file']:
            entry_type = 'file'
        else:
            entry_type = 'directory'
    
        # Walk through directory
        for dName, _sdName, fList in os.walk(root_dir):
    
            if entry_type == 'directory' and fnmatch(dName, pattern):
                yield dName
                continue
        
            for fname in fList:
                if entry_type == 'directory' and os.path.isdir(fname):
                    yield fname
                    continue
                # Match search string?
                if fnmatch(fname, pattern): 
                    yield os.path.join(dName, fname)

    #------------------------------------
    # find_in_dir_tree 
    #-------------------

    @classmethod
    def find_in_dir_tree(cls, root_dir, pattern='*', entry_type='file'):
        '''
        Similates part of the bash shell find command.
        Only bash patters are supported, not Python regex.
        For the find -type values, only file and directory 
        are supported as types. Though abbreviations
        'f' and 'd' are fine.
        
        Case of strings are case-normalized using os.path.normcase().
        Change fnmatch to fnmatchcase() in the function to
        perform case-sensitive searches.
        
        Examples:
           
           # Return  list of absolute file names in /tmp
              find_in_dir_tree('/tmp', entry_type='file')
              
           # Return list of text files in /tmp
              find_in_dir_tree('/tmp', pattern="*.txt")
              
           # Return list of directories in /tmp that start 
           # with 'Kfold'
              find_in_dir_tree('/tmp', pattern="Kfold", entry_type='directory')
              find_in_dir_tree('/tmp', pattern="Kfold", entry_type='d')

        See also: find_in_tree_gen() for a generator version of this method
                  (Should have implemented this method here as a generator
                   to start with, and only have one version...hindsight)

        :param root_dir: directory in which to start recursive descent
        :type root_dir: str
        :param pattern: pattern that file or dir must match
        :type pattern: str
        :param entry_type: whether file or directories are included
        :type entry_type: str
        :return: list of file or directory names
        :rtype: [str]
        '''
    
        file_list = []
        if pattern is None and entry_type is None:
            raise ValueError("At least one of pattern and/or type must be non-None")
        if entry_type not in [None, 'file', 'f', 'directory', 'd']:
            raise ValueError(f"Entry type must be one of {[None, 'file', 'f', 'directory', 'd']}, not {entry_type}")
    
        if entry_type in [None, 'f', 'file']:
            entry_type = 'file'
        else:
            entry_type = 'directory'
    
        # Walk through directory
        for dName, _sdName, fList in os.walk(root_dir):
    
            if entry_type == 'directory' and fnmatch(dName, pattern):
                file_list.append(dName)
                continue
        
            for fname in fList:
                if entry_type == 'directory' and os.path.isdir(fname):
                    file_list.append(fname)
                    continue
                # Match search string?
                if fnmatch(fname, pattern): 
                    file_list.append(os.path.join(dName, fname))
    
        return file_list

    #------------------------------------
    # compute_num_augs_per_species
    #-------------------

    @classmethod
    def compute_num_augs_per_species(cls, aug_volumes, sample_distrib_df):
        '''
        Return a dict mapping species name to 
        number of samples that should be available after
        augmentation. 
        
        The aug_volumes arg is either a dict mapping species name
        to an AugmentationGoals (TENTH, MEDIAN, MAX), or just
        an individual AugmentationGoals. The augmentation volume 
        indicates to which degree the least populated species should 
        be filled with augmented samples.  

        The sample_distrib_df is a dataframe whose row labels are 
        species names and the single column's values are numbers
        of available samples for training/validation/test for the
        respective row's species:
        
                        'num_samples'
               'bird1'   2
               'bird2'   6
        
        :param aug_volumes: augmentation goal for each species (median of 
            most popolous species, same as maximally populated, or 1/10th).
        :type aug_volumes: {AugmentationGoals | {str : AugmentationGoals}}
        :param sample_distrib_df: distribution of initially available
            sample numbers for each species
        :type sample_distrib_df: pandas.DataFrame
        :return: dict mapping each species to the 
            number of samples that need to be created.
        :rtype: {str : int}
        '''
        
        # For convenience:
        df = sample_distrib_df
        species_names = df.index
        
        # Find row index of species with maximum number of recordings:
        max_recs_idx = df['num_samples'].argmax()
        
        # Pull that row out of the df, getting a series:
        #  num_samples    6
        #  Name: HENLES_S, dtype: int6
        row_of_max = df.iloc[[max_recs_idx]].iloc[0]
        _max_species = row_of_max.name
        max_num_samples = row_of_max.num_samples
        
        # Compute the various methods (tenth, median, etc), 
        # relative to the max num of recordings: 
        # Plus 1 to avoid asking for 0 augmentations:
        tenth_max_num_samples = 1 + max_num_samples//10
        
        # Note: corner case for following statement:
        #   one species with no samples, another with 
        #   1 sample. Median is 0. That's fine, b/c
        #   we cannot augment the 0-samples anyway:
        median_num_samples = int(df['num_samples'].median())
        
        # Mapping goal names to number of recordings
        # that should be available after audio augmentation:
        volumes = {AugmentationGoals.TENTH  : tenth_max_num_samples,
                   AugmentationGoals.MEDIAN : median_num_samples,
                   AugmentationGoals.MAX    : max_num_samples
                   }
        
        aug_requirements = {}
        if isinstance(aug_volumes, AugmentationGoals):
            # Caller passed one aug volume for all species,
            # rather than a dict w/ species-by-species.
            # Create a dict with all species having that
            # one goal. That way all cases can be treated 
            # the same below:
            aug_volumes = {species_name : aug_volumes
                           for species_name
                           in species_names 
                           }
        # Do the computation:
        # Have dict of species-name : AugmentationGoals:
        for species in species_names:
            # ******* TODO: Ability to specify number manually:
            aug_requirement = aug_volumes[species]
            end_goal = volumes[aug_requirement]
            curr_num_recordings = df.loc[species, 'num_samples']
            aug_requirements[species] = max(0,end_goal - curr_num_recordings)
        
        return aug_requirements 

    #------------------------------------
    # orig_file_name
    #-------------------
    
    @classmethod
    def orig_file_name(cls, augmented_fname):
        '''
        Given the name of an augmented file, like Amaziliadecora1061880-volume-10.wav,
        return the original file from which the augmentation was created:
        Amaziliadecora1061880.wav
        
        :param augmented_fname: file name to decipher
        :type augmented_fname: src
        :return: the original file name
        :rtype: str
        '''

        path_elements = Utils.path_elements(augmented_fname)
        # Even if given name is already the orginal
        # name, the following will do the right thing:
        # return that name:
        orig_stem = path_elements['fname'].split('-')[0]
        orig_name = os.path.join(path_elements['root'], orig_stem+path_elements['suffix'])
        return orig_name

    #------------------------------------
    # listdir_abs
    #-------------------
    
    @classmethod
    def listdir_abs(cls, dir_path):
        '''
        Given a directory, return its content files. 
        But, in contrast to os.listdir(), the returned
        files are all absolute. Note that this is not
        a recursive walk; only the direct children of
        dir_path are examined.
        
        :param dir_path: absolute path for directory to list 
        :type dir_path: src
        :returns: list of absolute paths to all contained files
        :rtype: (str)
        '''
        abs_path_list = [os.path.join(dir_path, fname)
                         for fname
                         in os.listdir(dir_path)
                         ]
        return abs_path_list

    #------------------------------------
    # path_elements 
    #-------------------
    
    @classmethod
    def path_elements(cls, path):
        '''
        Given a path, return a dict of its elements:
        root, fname, and suffix. The method is almost
        like Path.parts or equivalent os.path method.
        But the 'root' may be absolute, or relative.
        And fname is provided without extension.
        
          foo/bar/blue.txt ==> {'root' : 'foo/bar',
                                'fname': 'blue',
                                'suffix: '.txt'
                                }

          /foo/bar/blue.txt ==> {'root' : '/foo/bar',
                                'fname': 'blue',
                                'suffix: '.txt'
                                }

          blue.txt ==> {'root' : '',
                        'fname': 'blue',
                        'suffix: '.txt'
                        }
        
        :param path: path to dissect
        :type path: str
        :return: dict with file elements
        :rtype: {str : str}
        '''
        
        p = Path(path)
        
        f_els = {}
        
        # Separate the dir from the fname:
        # From foo/bar/blue.txt  get ('foo', 'bar', 'blue.txt')
        # From /foo/bar/blue.txt get ('/', 'foo', 'bar', 'blue.txt')
        # From blue.txt          get ('blue.txt',)
        
        elements = p.parts
        if len(elements) == 1:
            # just blue.txt
            f_els['root']   = ''
            nm              = elements[0]
            f_els['fname']  = Path(nm).stem
            f_els['suffix'] = Path(nm).suffix
        else:
            # 
            f_els['root']     = os.path.join(*list(elements[:-1]))
            f_els['fname']    = p.stem
            f_els['suffix']   = p.suffix
        
        return f_els

    #------------------------------------
    # method_from_fname 
    #-------------------
    
    @classmethod
    def method_from_fname(cls, aug_name):

        # For audio:
        for meth_enum_el in AudAugMethod:
            if meth_enum_el.value in aug_name: return meth_enum_el.name
        # For Spectrograms:
            
        for meth_enum_el in ImgAugMethod:
            if meth_enum_el.value in aug_name: return meth_enum_el.name

    #------------------------------------
    # unique_fname 
    #-------------------
    
    @classmethod
    def unique_fname(cls, out_dir, fname):
        '''
        Returns a file name unique in the
        given directory. I.e. NOT globally unique.
        Keeps adding '_<i>' to end of file name.

        :param out_dir: directory for which fname is to 
            be uniquified
        :type out_dir: str
        :param fname: unique fname without leading path
        :type fname: str
        :return: either new, or given file name such that
            the returned name is unique within out_dir
        :rtype: str
        '''
        
        full_path   = os.path.join(out_dir, fname)
        fname_dict  = Utils.path_elements(full_path)
        i = 1

        while True:
            try:
                new_path = os.path.join(fname_dict['root'], fname_dict['fname']+fname_dict['suffix'])
                with open(new_path, 'r') as _fd:
                    # Succeed in opening, so file exists.
                    fname_dict['fname'] += f'_{i}'
                    i += 1
            except:
                # Couldn't open, so doesn't exist:
                return new_path

    #------------------------------------
    # is.audio_file
    #-------------------
    
    @classmethod
    def is_audio_file(cls, fpath):
        '''
        Very primitive check whether given file is
        and audio file. Just check the extension.
        
        :param fpath: path to check
        :type fpath: str
        :return True if extension is .wav or .mp3, else False
        :rtype bool
        '''
        return Path(fpath).suffix in ('.wav', '.mp3', '.ogg')
    
    #------------------------------------
    # user_confirm
    #-------------------
    
    @classmethod
    def user_confirm(cls, prompt_for_yes_no, default='Y'):
        resp = input(f"{prompt_for_yes_no} (default {default}): ")
        if resp in ('y','Y','yes','Yes', default):
            return True
        else:
            return False

    #------------------------------------
    # binary_in_interval_search 
    #-------------------

    @classmethod
    def binary_in_interval_search(cls, intervals, interval_start, low_key, high_key):
        '''
        Given a *sorted* list L of dict-like instances,
        and a number N, return the index of the instance
        in L that contains N. If none if the intervals
        contains N, return -1.
        
        The result is obtained through binary search.
        
        Note:
            o The intervals do not need to overlap. I.e.
                there may be holes
            o Like Python ranges, intervals are closed on
                the left, and open at the right. I.e. 
                an interval (4,6) includes 4, but not 6.
            o Unlike Python ranges, N may be a float, and so
                do the interval bounds.
            o One option for an acceptable dict-like for
                the elements in L are instances of Interval.
                But any dict-API supporting structure is fine.
        
        :param intervals: list of intervals for which to
            test membership
        :type intervals: [Interval]
        :param interval_start: number for which an enclosing
             interval is to be found
        :type interval_start: {int | float}
        :param low_key: key with which to retrieve an interval's
            low bound
        :type low_key: str
        :param high_key: key with which to retrieve an interval's
            high bound
        :type high_key: str
        :return index into intervals, or -1
        :rtype int
        '''
        
        if type(interval_start) != float:
            interval_start = float(interval_start)
        
        low = 0
        high = len(intervals) - 1
        mid = 0
        mids_tried = []
     
        while True:
    
            mid = (high - low) // 2
            if mid in mids_tried:
                return -1
            else:
                mids_tried.append(mid)
            
            # Does this mid-element's interval
            # contain interval_start?
            if intervals[mid][low_key] <= interval_start and \
               intervals[mid][high_key] > interval_start:
                # Found an entry with an interval in
                # which interval_start lies:
                return mid
            
            # Do possible intervals lie to the left?
            elif intervals[mid][high_key] <= interval_start:
                # Left interval is below bounds:
                low = mid + 1
            
            # If lower  is smaller, ignore right half
            # If upper of intervals el is greater
            # than interval_start, ignore left half
            
            elif intervals[mid][low_key] > interval_start:
                high = mid - 1


    #------------------------------------
    # set_seed  
    #-------------------

    @classmethod
    def set_seed(cls, seed):
        '''
        Set the seed across all different necessary platforms
        to allow for comparison of different models and runs
        
        :param seed: random seed to set for all random num generators
        :type seed: int
        '''
        torch.manual_seed(seed)
        cuda.manual_seed_all(seed)
        # Not totally sure what these two do!
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        random.seed(seed)

    #------------------------------------
    # random_list_percentage 
    #-------------------
    
    @classmethod
    def random_list_percentage(cls, lst, percentage):
        num_items_to_get = int(len(lst)*percentage/100)
        res_lst = random.sample(lst, num_items_to_get)
        return res_lst

    #------------------------------------
    # add_pyplot_manager_to_fig
    #-------------------
    
    @classmethod
    def add_pyplot_manager_to_fig(cls, fig):
        '''
        Sometimes, when pyplot figures are passed around,
        they lose their association with their canvas manager.
        This method creates a dummy manager, then adds that
        manager to the passed-in Figure instance in place.
        
        Use when fig.show() throws:
           AttributeError: Figure.show works only for figures managed 
                by pyplot, normally created by pyplot.figure()
        
        After calling this method, fig.show() will work. Nothing is
        returned.
        
        :param fig: figure to which a canvas manager must be added
        :type fig: maptlotlib.pyplot.Figure
        '''
        
        # Create a dummy figure and use its
        # manager to display "fig"  
        dummy = plt.figure()
        dummy.set_size_inches(fig.get_size_inches())
        new_manager = dummy.canvas.manager
        new_manager.canvas.figure = fig
        fig.set_canvas(new_manager.canvas)

    #------------------------------------
    # read_raven_selection_table 
    #-------------------
    
    @classmethod
    def read_raven_selection_table(cls, tbl_path):
        '''
        Given the path to a Raven selection table
        .csv file, return a list of dicts. Each dict
        contains the information in one selection table
        row, plus two additional entries: time_interval,
        and freq_interval; these are added for convenience:
        
        Selection           row number <int>
        View                not used
        Channel             not used
        Begin Time (s)      begin of vocalization in fractional seconds <float>
        End Time (s)        end of vocalization in fractional seconds <float>
        Low Freq (Hz)       lowest frequency within the lassoed vocalization <float>
        High Freq (Hz)      highest frequency within the lassoed vocalization <float>
        species             four-letter species name <str>
        type                {song, call, call-1, call-trill} <str>
        number              not used
        mix                 comma separated list of other audible species [<str>]
        
        time_interval       Inteval instance from start and end times
        freq_interval       Inteval instance from start and end frequencies
        
        Values are converted to appropriate types as above.
        Output is suitable for SnippetSelectionTableMapper.match_snippets()
        
        The list will be sorted by ascending the 'Begin Time (s)'
        value.
        
        In the incoming selection tables, the header 'species' is sometimes called
        any of 'species', 'Especie', 'Specie', 'SPECIE'

        :param tbl_path: full path to Raven selection table
        :type tbl_path: src
        :return: list of dicts, each dict reflecting the content
            of one selection table row
        :rtype [str : Any]
        '''
        with open(tbl_path, 'r') as sel_tbl_fd:
            reader = DictReader(sel_tbl_fd, delimiter='\t')
            sel_dict_list = [row_dict for row_dict in reader]
        
        
        # Coerce types and unify keys:
        
        species_spellings = ['species', 'Especie', 'Specie', 'SPECIE', 'specie']
        type_spellings    = ['Type', 'tipo', 'Tipo', 'TYPE', 'TIPPO']
        mix_spellings     = ['Mix', 'MIX', 'mezcla', 'Mezcla', 'MEZCLA']
        number_spellings  = ['Number', 'NUMBER', 'numero', 'Numero', 'NUMERO']
        
        for sel_dict in sel_dict_list:
            sel_dict['Selection'] = str(sel_dict['Selection'])
            sel_dict['Begin Time (s)'] = float(sel_dict['Begin Time (s)'])
            sel_dict['End Time (s)'] = float(sel_dict['End Time (s)'])
            sel_dict['Low Freq (Hz)'] = float(sel_dict['Low Freq (Hz)'])
            sel_dict['High Freq (Hz)'] = float(sel_dict['High Freq (Hz)'])
            
            
            # Go through the column names that human labelers
            # add manually, rather than being built-into Raven.
            # Eradicate any spelling differences:

            col_keys = list(sel_dict.keys())

            # Make sure there is a 'type' column,
            # the one that says 'call', 'song,' etc.
            # The col name could have different spellings:
            
            if 'type' not in col_keys:
                # Maybe there is no col called 'type'...
                sel_dict['type'] = ''
                # ... but we'll check for different spellings:
                for type_col_name in type_spellings:
                    try:
                        type_val = sel_dict[type_col_name]
                        sel_dict['type'] = type_val
                        del sel_dict[type_col_name]
                        break
                    except KeyError:
                        # That wasn't the spelling used
                        continue

            # Same for 'mix' column:
            if 'mix' not in col_keys:
                # Maybe there is no col called 'mix'...
                sel_dict['mix'] = ''
                # ...but we'll check for different spellings:
                for mix_col_name in mix_spellings:
                    try:
                        mix_val = sel_dict[mix_col_name]
                        sel_dict['mix'] = mix_val
                        del sel_dict[mix_col_name]
                        break
                    except KeyError:
                        # That wasn't the spelling used
                        continue

            # Same for 'mix' column:
            if 'number' not in col_keys:
                # Maybe there is no col called 'number'...
                sel_dict['number'] = '1'
                # ...but we'll check for different spellings:
                for number_col_name in number_spellings:
                    try:
                        number_val = sel_dict[number_col_name]
                        sel_dict['number'] = number_val
                        del sel_dict[number_col_name]
                        break
                    except KeyError:
                        # That wasn't the spelling used
                        continue

            # Make the four-letter species names upper case:
            try:
                sel_dict['species'] = sel_dict['species'].upper()
            except KeyError:
                dict_keys = list(sel_dict.keys())
                success = False
                for key_alternative in species_spellings:
                    if key_alternative in dict_keys:
                        # Found the column name used for species in this tbl:
                        sel_dict['species'] = sel_dict[key_alternative].upper()
                        del sel_dict[key_alternative]
                        success = True
                        break
                if not success:
                    raise KeyError(f"Could not find species column selection table {tbl_path}")

            # Turn the comma-separated list of
            # overlapping vocalizations into
            # a (possibly empty) list of strings:
            try:
                sel_dict['mix'] = [] if (sel_dict['mix'] is None or len(sel_dict['mix']) == 0) \
                                     else sel_dict['mix'].split(',')
            except KeyError:
                sel_dict['mix'] = []
            # Clean out spurious white space:
            sel_dict['mix'] = [mix_list_entry.strip().upper() 
                               for mix_list_entry
                               in sel_dict['mix']
                               ]
            # Convert all mix 4-letter codes into 5-letter codes:
            new_mix = []
            for mix_species in sel_dict['mix']:
                new_mix.append(cls._four_to_five_from_sel_row_dict(mix_species, sel_dict, tbl_path))
            sel_dict['mix'] = new_mix
            
            # Same for the single (4-letter) species:
            try:
                sel_dict['species'] = cls._four_to_five_from_sel_row_dict(sel_dict['species'], 
                                                                          sel_dict, 
                                                                          tbl_path)
            except ValueError:
                # Likely the species was entered as something 
                # like 'Motorcycle' or 'thunder' or 'NO BIRD':
                sel_dict['species'] = 'NOISG'

            sel_dict['time_interval'] = Interval(sel_dict['Begin Time (s)'],
                                                 sel_dict['End Time (s)'])
            sel_dict['freq_interval'] = Interval(sel_dict['Low Freq (Hz)'],
                                                 sel_dict['High Freq (Hz)'])

            
        # Make sure the list is sorted by 
        # ascending start time:
        sel_dict_list_sorted = natsorted(sel_dict_list, 
                                         key=lambda row_dict: row_dict['Begin Time (s)'])
        return sel_dict_list_sorted

    #------------------------------------
    # _four_to_five_from_sel_row_dict
    #-------------------
    
    @classmethod
    def _four_to_five_from_sel_row_dict(cls, four_code, sel_row_dict, tbl_path):
        '''
        Given a 4-letter species code, use SpeciesNameConverter to
        a convert to 5-letter coded. That conversion failing with a
        ConversionError means that the species needs to be split into
        song/call. In that case, checks whether the 'Type' entry in the
        given sel_row_dict contains 'SONG' or 'CALL'. If so, accomplish
        the code conversion by adding 'S' or 'C'. Else raise a ConversionError.
        
        The sel_row_dict is a dict containing all info from one row
        in one selection table. 
        
        The tbl_path is used to provide good error
        messages.

        :param four_code: 4-letter species code to convert to 5-letter code
        :type four_code: str
        :param sel_row_dict: map generated from one row in one Raven selection table
        :type sel_row_dict: {str : {str | [str]}}
        :param tbl_path: full path to selection table from which 
            sel_row_dict was generated
        :type tbl_path: str
        :return 5-letter species code
        :rtype str
        :raise ConversionError if 5-letter code unknown for 4-letter code,
            or when given species must be split into SONG/CALL, but the
            'Type' entry in sel_row_dict does not contain 'SONG' or 'CALL.
        '''
        try: 
            five_code = SpeciesNameConverter()[four_code, DIRECTION.FOUR_FIVE]
        except ConversionError:
            # Species is split by song/call, so get the type:
            species_type = sel_row_dict['type'].upper()
            if species_type not in ['SONG', 'CALL']:
                raise ConversionError(f"Selection table {tbl_path}, selection# {sel_row_dict['Selection']} species {four_code} needs song/call info")
            # We have the required info:
            five_code = four_code + ('S' if species_type == 'SONG' else 'C')

        return five_code


    #------------------------------------
    # read_configuration 
    #-------------------

    @classmethod
    def read_configuration(cls, conf_file):
        '''
        Parses config file that describes training parameters,
        various file paths, and how many GPUs different machines have.
        Syntax follows Python's configfile package, which includes
        sections, and attr/val pairs in each section.
        
        Expected sections:

           o Paths: various file paths for the application
           o Training: holds batch sizes, number of epochs, etc.
           o Parallelism: holds number of GPUs on different machines
        
        For Parallelism, expect entries like:
        
           foo.bar.com  = 4
           127.0.0.1    = 5
           localhost    = 3
           172.12.145.1 = 6
           
        Method identifies which of the entries is
        'localhost' by comparing against local hostname.
        Though 'localhost' or '127.0.0.1' may be provided.
        
        Returns a dict of dicts: 
            config[section-names][attr-names-within-section]
            
        Types of standard entries, such as epochs, batch_size,
        etc. are coerced, so that, e.g. config['Training']['epochs']
        will be an int. Clients may add non-standard entries.
        For those the client must convert values from string
        (the type in which values are stored by default) to the
        required type. This can be done the usual way: int(...),
        or using one of the configparser's retrieval methods
        getboolean(), getint(), and getfloat():
        
            config['Training'].getfloat('learning_rate')
        
        :param other_gpu_config_file: path to configuration file
        :type other_gpu_config_file: str
        :return: a dict of dicts mirroring the config file sections/entries
        :rtype: dict[dict]
        :raises ValueErr
        :raises TypeError
        '''
        
        if conf_file is None:
            return self.init_defaults()
        
        config = NeuralNetConfig(conf_file)
        
        if len(config.sections()) == 0:
            # Config file exists, but empty:
            return(self.init_defaults(config))
    
        # Do type conversion also in other entries that 
        # are standard:
        
        types = {'epochs' : int,
                 'batch_size' : int,
                 'kernel_size' : int,
                 'sample_width' : int,
                 'sample_height' : int,
                 'seed' : int,
                 'pytorch_comm_port' : int,
                 'num_pretrained_layers' : int,
                 
                 'root_train_test_data': str,
                 'net_name' : str,
                 }
        for section in config.sections():
            for attr_name in config[section].keys():
                try:
                    str_val = config[section][attr_name]
                    required_type = types[attr_name]
                    config[section][attr_name] = required_type(str_val)
                except KeyError:
                    # Current attribute is not standard;
                    # users of the corresponding value need
                    # to do their own type conversion when
                    # accessing this configuration entry:
                    continue
                except TypeError:
                    raise ValueError(f"Config file error: {section}.{attr_name} should be convertible to {required_type}")
    
        return config


    #------------------------------------
    # time_delta_str 
    #-------------------

    @classmethod
    def time_delta_str(cls, time_delta, granularity=2):
        '''
        Takes the difference between two datetime times:
        
               start_time = datetime.datetime.now()
               <some time elapses>
               end_time = datetime.datetime.now()
               
               delta = end_time - start_time
               time_delta_str(delta
        
        Depending on granularity, returns a string like:
        
            Granularity:
                      1  '160.0 weeks'
                      2  '160.0 weeks, 4.0 days'
                      3  '160.0 weeks, 4.0 days, 6.0 hours'
                      4  '160.0 weeks, 4.0 days, 6.0 hours, 42.0 minutes'
                      5  '160.0 weeks, 4.0 days, 6.0 hours, 42.0 minutes, 13.0 seconds'
        
            For smaller time deltas, such as 10 seconds,
            does not include leading zero times. For
            any granularity:
            
                          '10.0 seconds'

            If duration is less than second, returns '< 1sec>'
            
        :param time_delta: time difference to turn into a string
        :type time_delta: datetime.timedelta
        :param granularity: time granularity down to which to compute
        :type granularity: int
        :return: printable string of the time duration in readable form
        :rtype: str
        
        '''
        intervals = (
            ('weeks', 604800),  # 60 * 60 * 24 * 7
            ('days', 86400),    # 60 * 60 * 24
            ('hours', 3600),    # 60 * 60
            ('minutes', 60),
            ('seconds', 1),
            )
        secs = time_delta.total_seconds()
        result = []
        for name, count in intervals:
            value = secs // count
            if value:
                secs -= value * count
                if value == 1:
                    name = name.rstrip('s')
                result.append("{} {}".format(value, name))
        dur_str = ', '.join(result[:granularity])
        if len(dur_str) == 0:
            dur_str = '< 1sec'
        return dur_str

    #------------------------------------
    # pad_series
    #-------------------
    
    @classmethod
    def pad_series(cls, the_series, side, length):
        '''
        Given a series and a length, pad the
        series with its first (or last) element
        until length is reached. 
        
        The side argument must be in {'left', 'right'}.
        If len(the_series) >= length is already satisfied,
        returns the_series unchanged.
        
        :param the_series: series to pad
        :type the_series: pd.Series
        :param side: either 'left' or 'right'
        :type side: str
        :param length: target length
        :type length: int
        :return new, padded series, or orginal series
        :rtype: pd.Series
        '''
        
        if side not in ['left', 'right']:
            raise ValueError(f"The side parameter must be 'left' or 'right', not {side}")
        
        if len(the_series) >= length:
            return the_series
        
        num_pads_needed = length - len(the_series)
        if side == 'right':
            pad_val = the_series.iloc[-1]
            # Append padding to the_series
            new_series = the_series.append(pd.Series([pad_val]*num_pads_needed), 
                                           ignore_index=True)
        else:
            # Pad on the left:
            pad_val = the_series.iloc[0]
            # Append the_series to padding:
            new_series = pd.Series([pad_val]*num_pads_needed).append(the_series, 
                                                                     ignore_index=True)
        return new_series


# -------------------- Class ProcessWithoutWarnings ----------

class ProcessWithoutWarnings(mp.Process):
    '''
    Subclass of Process to use when creating
    multiprocessing jobs. Accomplishes two
    items in addition to the parent:
    
       o Blocks printout of various deprecation warnings connected
           with matplotlib and librosa
       o Adds ability for CreateSpectrogram and SpectrogramChopper 
           instances to return a result.
    '''
    
    #def run(self, *args, **kwargs):
    def run(self):

        # Don't show the annoying deprecation
        # librosa.display() warnings about renaming
        # 'basey' to 'base' to match matplotlib: 
        warnings.simplefilter("ignore", category=MatplotlibDeprecationWarning)
        
        # Hide the UserWarning: PySoundFile failed. Trying audioread instead.
        warnings.filterwarnings(action="ignore",
                                message="PySoundFile failed. Trying audioread instead.",
                                category=UserWarning, 
                                module='', 
                                lineno=0)

        return super().run()
    
    @property
    def ret_val(self):
        try:
            return self._ret_val
        except NameError:
            return None
        
    @ret_val.setter
    def ret_val(self, new_val):
        if not type(new_val) == mp.sharedctypes.Synchronized:
            raise TypeError(f"The ret_val instance var must be multiprocessing shared C-type, not {new_val}")
        self._ret_val = new_val

