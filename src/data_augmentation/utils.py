from enum import Enum
from fnmatch import fnmatch
import os
from pathlib import Path
import random
import shutil

import librosa

import numpy as np
import pandas as pd


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
    GAUSS       = '-gauss'
    FMASK       = 'fmask'
    TMASK       = 'tmask'
    ORIGINAL    = 'original'

#------------------------------ Utility  -------------

class Utils:

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
           
			                        add_bg  time_shift  volume  mask  gauss original
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
                                 "gauss":0,
                                 "original":0}
                for sample_name in os.listdir(os.path.join(path, species)):
                    if "_bgd" in sample_name: aug_type_dict["add_bg"] += 1
                    elif "-shift" in sample_name: aug_type_dict["time_shift"] += 1
                    elif "-volume" in sample_name: aug_type_dict["volume"] += 1
                    
                    # For Spectrograms:
                    elif "-gauss" in sample_name: aug_type_dict["gauss"] += 1
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
    # find_in_dir_tree 
    #-------------------

    @classmethod
    def find_in_dir_tree(cls, root_dir, pattern='*', entry_type='file'):
        '''
        Similates part of the bash shell find command.
        Only bash patters are supported, not Python regex.
        For the find -type values, only file and directory 
        are supported as file types. Though abbreviations
        'f' and 'd' are fine.
        
        Case of strings are case-normalized using os.path.normcase().
        Change fnmatch to fnmatchcase() in the function to
        perform case-sensitive searches.
        
        Examples:
           
           # Return list of files in /tmp
              find_in_dir_tree('/tmp', entry_type='file')
              
           # Return list of text files in /tmp
              find_in_dir_tree('/tmp', pattern="*.txt")
              
           # Return list of directories in /tmp that start 
           # with 'Kfold'
              find_in_dir_tree('/tmp', pattern="Kfold", entry_type='directory')
              find_in_dir_tree('/tmp', pattern="Kfold", entry_type='d')
              
        
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
        files are all absolute.
        
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
        fname_dict = Utils.path_elements(full_path)

        while True:
            try:
                new_path = os.path.join(fname_dict['root'], fname_dict['fname']+fname_dict['suffix'])
                with open(new_path, 'r') as _fd:
                    # Succeed in opening, so file exists.
                    fname_dict['fname'] += '_'
            except:
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
        return Path(fpath).suffix in ('.wav', '.mp3')
    
    #------------------------------------
    # user_confirm
    #-------------------
    
    @classmethod
    def user_confirm(cls, prompt_for_yes_no, default='Y'):
        resp = input(f"{prompt_for_yes_no} (default {default}): ")
        if resp in ('y','Y','yes','Yes', ''):
            return True
        else:
            return False
